// Package testart writes generated test output under testdata/artifacts
// (gitignored). Tests should keep fixtures in testdata outside artifacts/.
package testart

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Dir is testdata/artifacts at the module root. It is created if missing.
func Dir(t testing.TB) string {
	t.Helper()
	if env := os.Getenv("QUIKAITOOLS_ARTIFACTS"); env != "" {
		if err := os.MkdirAll(env, 0o755); err != nil {
			t.Fatalf("artifacts dir: %v", err)
		}
		return env
	}
	dir := filepath.Join(moduleRoot(t), "testdata", "artifacts")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("artifacts dir: %v", err)
	}
	return dir
}

// Path is Dir plus name (and optional extra path elements).
func Path(t testing.TB, name string, extra ...string) string {
	t.Helper()
	parts := append([]string{Dir(t), name}, extra...)
	return filepath.Join(parts...)
}

// WriteFile writes data to testdata/artifacts/name.
func WriteFile(t testing.TB, name string, data []byte) string {
	t.Helper()
	path := Path(t, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func moduleRoot(t testing.TB) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("go.mod not found from testart package")
	return ""
}
