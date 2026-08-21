package infer

import (
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
