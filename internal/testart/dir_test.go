package testart

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDirIsUnderGitignoredArtifacts(t *testing.T) {
	dir := Dir(t)
	if !strings.HasSuffix(filepath.ToSlash(dir), "testdata/artifacts") {
		t.Fatalf("dir = %s", dir)
	}
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		t.Fatalf("stat: %v", err)
	}
	path := WriteFile(t, "testart-smoke.txt", []byte("ok\n"))
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestEnvOverride(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("QUIKAITOOLS_ARTIFACTS", tmp)
	if Dir(t) != tmp {
		t.Fatalf("got %s want %s", Dir(t), tmp)
	}
	WriteFile(t, "nested/out.txt", []byte("x"))
	if _, err := os.Stat(filepath.Join(tmp, "nested", "out.txt")); err != nil {
		t.Fatal(err)
	}
}
