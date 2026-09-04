// Package infer is Layer A inference: GGUF via llama.cpp exec, GPT-2 generate in-process.
package infer

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/shaneburrell/quikaitools/internal/backend"
)

// LlamaBinary names we look for on PATH (or QUIKAITOOLS_LLAMA).
// Do not include generic names like "main" (PATH hijack on shared boxes).
//
// Non-goal: llama-server (the HTTP daemon) is documented here only. This
// package never execs llama-server; generation is one-shot via llama-cli /
// llama-completion (or QUIKAITOOLS_LLAMA).
var LlamaBinaryCandidates = []string{
	"llama-cli",
	"llama-completion",
}

// LookLlama returns the first llama.cpp binary found.
func LookLlama() (string, error) {
	if env := strings.TrimSpace(os.Getenv("QUIKAITOOLS_LLAMA")); env != "" {
		if st, err := os.Stat(env); err == nil && !st.IsDir() {
			return env, nil
		}
		return "", fmt.Errorf("QUIKAITOOLS_LLAMA=%s not found", env)
	}
	for _, name := range LlamaBinaryCandidates {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("llama.cpp binary not found (tried %s); set QUIKAITOOLS_LLAMA or install llama-cli", strings.Join(LlamaBinaryCandidates, ", "))
}

// ExpectedLlamaLabel is what doctor prints for a profile.
func ExpectedLlamaLabel(p backend.Profile) string {
	return p.InferGGUF
}

// StrictCheck returns warnings/errors for doctor --strict.
func StrictCheck(p backend.Profile) (ok bool, lines []string) {
	ok = true
	bin, err := LookLlama()
	if err != nil {
		ok = false
		lines = append(lines, "strict: "+err.Error())
		lines = append(lines, fmt.Sprintf("strict: expected GGUF engine %s", p.InferGGUF))
	} else {
		lines = append(lines, "strict: llama="+bin)
		lines = append(lines, "strict: gguf_engine="+p.InferGGUF)
	}
	if p.Kind == backend.KindCPU {
		lines = append(lines, "strict: warning: profile is cpu — GPU boxes should not land here")
	}
	if p.Kind == backend.KindV100 || p.Kind == backend.KindCUDA {
		if !hasNVIDIA() {
			ok = false
			lines = append(lines, "strict: nvidia-smi not available but profile is "+string(p.Kind))
		}
	}
	if p.Kind == backend.KindHalo {
		if !fileExists("/dev/kfd") {
			lines = append(lines, "strict: warning: /dev/kfd missing (Halo ROCm/HIP may be unavailable; Vulkan may still work)")
		}
	}
	return ok, lines
}

func hasNVIDIA() bool {
	_, err := exec.LookPath("nvidia-smi")
	return err == nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// GenerateGGUFOptions configures llama.cpp completion.
type GenerateGGUFOptions struct {
	Model   string
	Prompt  string
	Tokens  int
	Profile backend.Profile
	// Timeout bounds the llama.cpp process. Zero means 5 minutes.
	Timeout time.Duration
}

// GenerateGGUF runs llama-cli (or QUIKAITOOLS_LLAMA) against a .gguf file.
func GenerateGGUF(opt GenerateGGUFOptions) (string, error) {
	if opt.Model == "" {
		return "", fmt.Errorf("infer: --gguf model path required")
	}
	if _, err := os.Stat(opt.Model); err != nil {
		return "", fmt.Errorf("infer: model: %w", err)
	}
	bin, err := LookLlama()
	if err != nil {
		return "", err
	}
	if opt.Tokens <= 0 {
		opt.Tokens = 32
	}
	prompt := opt.Prompt
	if prompt == "" {
		prompt = "Hello"
	}
	timeout := opt.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	args := []string{"-m", opt.Model, "-n", fmt.Sprintf("%d", opt.Tokens), "-p", prompt, "--no-display-prompt"}
	// ngl for GPU offload when available
	switch opt.Profile.Kind {
	case backend.KindMac, backend.KindV100, backend.KindCUDA, backend.KindHalo:
		args = append(args, "-ngl", "99")
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	// After the context kills the process, do not wait forever on children
	// that still hold the stdout/stderr pipes open.
	cmd.WaitDelay = 2 * time.Second
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "", fmt.Errorf("infer: llama.cpp timed out after %s", timeout)
		}
		return "", fmt.Errorf("llama.cpp: %w\n%s", err, strings.TrimSpace(stderr.String()))
	}
	out := strings.TrimSpace(stdout.String())
	if out == "" {
		// some builds write only to stderr
		out = strings.TrimSpace(stderr.String())
	}
	return out, nil
}

// FindGGUF walks dir for the first *.gguf file.
func FindGGUF(dir string) (string, error) {
	var found string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		if strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			found = path
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil && !errors.Is(err, filepath.SkipAll) {
		return "", err
	}
	if found == "" {
		return "", fmt.Errorf("no .gguf under %s", dir)
	}
	return found, nil
}
