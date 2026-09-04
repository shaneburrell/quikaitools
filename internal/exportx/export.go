// Package exportx merges LoRA adapters and prepares GGUF/Ollama export helpers.
package exportx

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
)

// defaultModelfile is used when no template path is given (or the path is unreadable).
const defaultModelfile = `FROM {{gguf}}
PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
`

// MergeGPT2LoRA merges LoRA adapters into base GPT-2 and writes a HF-like dir.
func MergeGPT2LoRA(modelDir, adapterDir, outDir string) error {
	base, err := gpt2.LoadDir(modelDir)
	if err != nil {
		return err
	}
	if err := peft.LoadAndMerge(base, adapterDir); err != nil {
		return err
	}
	var missing []string
	for _, name := range []string{"vocab.json", "merges.txt"} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err != nil {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("exportx: missing in source: %s", strings.Join(missing, ", "))
	}
	if err := gpt2.WriteDir(outDir, base); err != nil {
		return err
	}
	for _, name := range []string{"tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"} {
		src := filepath.Join(modelDir, name)
		dst := filepath.Join(outDir, name)
		b, err := os.ReadFile(src)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return err
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			return err
		}
	}
	dstCfg := filepath.Join(outDir, "config.json")
	if _, err := os.Stat(dstCfg); err != nil {
		srcCfg := filepath.Join(modelDir, "config.json")
		if b, err := os.ReadFile(srcCfg); err == nil {
			if err := os.WriteFile(dstCfg, b, 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}

// WriteModelfile writes an Ollama Modelfile pointing at ggufPath.
// Existing outPath is overwritten (force=true). See WriteModelfileOpts.
func WriteModelfile(templatePath, ggufPath, outPath string) error {
	return WriteModelfileOpts(templatePath, ggufPath, outPath, true)
}

// WriteModelfileOpts writes an Ollama Modelfile. If force is false and out
// already exists, it returns an error instead of overwriting.
func WriteModelfileOpts(templatePath, gguf, out string, force bool) error {
	if !force {
		if _, err := os.Stat(out); err == nil {
			return fmt.Errorf("exportx: %s exists (pass force to overwrite)", out)
		}
	}
	b, err := os.ReadFile(templatePath)
	if err != nil {
		b = []byte(defaultModelfile)
	}
	text := string(b)
	lines := strings.Split(text, "\n")
	replaced := false
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "FROM ") {
			lines[i] = "FROM " + gguf
			replaced = true
			break
		}
	}
	text = strings.Join(lines, "\n")
	text = strings.ReplaceAll(text, "{{GGUF}}", gguf)
	text = strings.ReplaceAll(text, "{{gguf}}", gguf)
	if !replaced && !strings.Contains(text, gguf) {
		text = "FROM " + gguf + "\n" + text
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	return os.WriteFile(out, []byte(text), 0o644)
}

// ConvertGGUF looks for llama.cpp convert tools; returns skip reason if unavailable.
// QuikAITools does not vendor Python; skip is success when tools are missing.
//
// Quantize writes to outGGUF+".tmp" and os.Rename onto outGGUF on success.
// Failure removes both temp files and never leaves a partial outGGUF.
// The converter is invoked with $PYTHON, else python3, else python.
func ConvertGGUF(modelDir, outGGUF, quant string) (skipped string, err error) {
	if quant == "" {
		quant = "Q4_K_M"
	}
	convert := findOnPath("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
	quantize := findOnPath("llama-quantize", "quantize")
	if convert == "" {
		return "no convert_hf_to_gguf.py on PATH — supply a GGUF manually or install llama.cpp convert tools", nil
	}
	if err := os.MkdirAll(filepath.Dir(outGGUF), 0o755); err != nil {
		return "", err
	}
	tmpF16 := outGGUF + ".f16.gguf"
	tmpQ := outGGUF + ".tmp"
	cleanup := func() {
		_ = os.Remove(tmpF16)
		_ = os.Remove(tmpQ)
	}
	py := findPython()
	cmd := exec.Command(py, convert, modelDir, "--outfile", tmpF16, "--outtype", "f16")
	out, err := cmd.CombinedOutput()
	if err != nil {
		cleanup()
		return "", fmt.Errorf("convert: %w\n%s", err, string(out))
	}
	if quantize == "" {
		if err := os.Rename(tmpF16, tmpQ); err != nil {
			cleanup()
			return "", err
		}
		if err := os.Rename(tmpQ, outGGUF); err != nil {
			cleanup()
			return "", err
		}
		return "quantizer missing — left F16 GGUF at " + outGGUF, nil
	}
	cmd = exec.Command(quantize, tmpF16, tmpQ, quant)
	out, err = cmd.CombinedOutput()
	_ = os.Remove(tmpF16)
	if err != nil {
		cleanup()
		return "", fmt.Errorf("quantize: %w\n%s", err, string(out))
	}
	if err := os.Rename(tmpQ, outGGUF); err != nil {
		cleanup()
		return "", err
	}
	return "", nil
}

func findPython() string {
	if p := strings.TrimSpace(os.Getenv("PYTHON")); p != "" {
		return p
	}
	if p := findOnPath("python3", "python"); p != "" {
		return p
	}
	return "python3"
}

func findOnPath(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	return ""
}
