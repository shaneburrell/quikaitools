package infer

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/backend"
)

func TestStrictCheckCPU(t *testing.T) {
	p, err := backend.Detect(backend.KindCPU, func() backend.HostInfo {
		return backend.HostInfo{GOOS: "linux"}
	})
	if err != nil {
		t.Fatal(err)
	}
	ok, lines := StrictCheck(p)
	if len(lines) == 0 {
		t.Fatal("expected lines")
	}
	// llama may or may not be present; either way we get lines
	_ = ok
}

func TestExpectedLlamaLabel(t *testing.T) {
	p, _ := backend.Detect(backend.KindMac, func() backend.HostInfo { return backend.HostInfo{GOOS: "darwin"} })
	if ExpectedLlamaLabel(p) != "llamacpp-metal" {
		t.Fatalf("%s", ExpectedLlamaLabel(p))
	}
}

func TestFindGGUFMissing(t *testing.T) {
	dir := t.TempDir()
	if _, err := FindGGUF(dir); err == nil {
		t.Fatal("expected error")
	}
}

func TestGenerateGGUFFakeBinary(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script PATH stub")
	}
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	script := filepath.Join(dir, "llama-cli")
	body := "#!/bin/sh\necho \"$@\" >" + argsFile + "\necho \"fake output\"\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("QUIKAITOOLS_LLAMA", "")

	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("gg"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := GenerateGGUF(GenerateGGUFOptions{Model: model, Prompt: "hi", Tokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	if out != "fake output" {
		t.Fatalf("got %q", out)
	}
	args, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	s := string(args)
	for _, want := range []string{"-m", model, "-p", "hi", "-n", "8"} {
		if !strings.Contains(s, want) {
			t.Fatalf("args %q missing %q", s, want)
		}
	}
}

func TestGenerateGGUFMissingBinaryMentionsEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	t.Setenv("QUIKAITOOLS_LLAMA", "")
	model := filepath.Join(dir, "x.gguf")
	if err := os.WriteFile(model, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := GenerateGGUF(GenerateGGUFOptions{Model: model, Prompt: "hi", Tokens: 1})
	if err == nil || !strings.Contains(err.Error(), "QUIKAITOOLS_LLAMA") {
		t.Fatalf("want QUIKAITOOLS_LLAMA in error, got %v", err)
	}
}
