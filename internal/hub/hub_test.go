package hub

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPullAndSkipExisting(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		_, _ = w.Write([]byte("ok-bytes"))
	}))
	defer srv.Close()

	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0

	dir, err := c.Pull("org/tiny", []string{"config.json"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil || string(b) != "ok-bytes" {
		t.Fatalf("got %q err=%v", b, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json.ok")); err != nil {
		t.Fatalf("expected .ok marker: %v", err)
	}
	if _, err := c.Pull("org/tiny", []string{"config.json"}); err != nil {
		t.Fatal(err)
	}
	if hits != 1 {
		t.Fatalf("hits=%d want 1 (second pull cached)", hits)
	}
}

func TestValidateRepoRejectsTraversal(t *testing.T) {
	for _, bad := range []string{"", "../../../.ssh", "org/../../etc", "org", "org/name/extra", "org\\name", "/abs/path"} {
		if err := ValidateRepo(bad); err == nil {
			t.Fatalf("expected reject for %q", bad)
		}
	}
	if err := ValidateRepo("hf-internal-testing/tiny-random-gpt2"); err != nil {
		t.Fatal(err)
	}
}

func TestPullRejectsBadRepo(t *testing.T) {
	c := New(t.TempDir())
	if _, err := c.Pull("../../../.ssh", []string{"config.json"}); err == nil {
		t.Fatal("expected error")
	}
}

func TestPullRequired404Fails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "tokenizer_config.json") {
			http.NotFound(w, r)
			return
		}
		if strings.Contains(r.URL.Path, "model.safetensors") {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	if _, err := c.Pull("org/tiny", []string{"config.json", "model.safetensors", "tokenizer_config.json"}); err == nil {
		t.Fatal("expected required model.safetensors 404 to fail")
	}
	// optional tokenizer_config alone with success on config should work
	if _, err := c.Pull("org/tiny2", []string{"config.json", "tokenizer_config.json"}); err != nil {
		t.Fatal(err)
	}
}

func TestPullSendsAuthorization(t *testing.T) {
	t.Setenv("HF_TOKEN", "tok123")
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	if _, err := c.Pull("org/tiny", []string{"config.json"}); err != nil {
		t.Fatal(err)
	}
	if got != "Bearer tok123" {
		t.Fatalf("Authorization=%q", got)
	}
}

func TestPullFollowsRedirect(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/resolve/") {
			http.Redirect(w, r, "/blob/config.json", http.StatusFound)
			return
		}
		_, _ = w.Write([]byte("redirected-ok"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	dir, err := c.Pull("org/tiny", []string{"config.json"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil || string(b) != "redirected-ok" {
		t.Fatalf("got %q err=%v", b, err)
	}
}

func TestPullTruncatedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		_, _ = w.Write([]byte("tiny"))
	}))
	defer srv.Close()
	root := t.TempDir()
	c := New(root)
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	_, err := c.Pull("org/tiny", []string{"config.json"})
	if err == nil {
		t.Fatal("expected truncated-body error")
	}
	dest := filepath.Join(c.ModelDir("org/tiny"), "config.json")
	if _, stErr := os.Stat(dest); !os.IsNotExist(stErr) {
		t.Fatalf("partial dest left: %v", stErr)
	}
	if _, stErr := os.Stat(dest + ".tmp"); !os.IsNotExist(stErr) {
		t.Fatalf("partial .tmp left: %v", stErr)
	}
	if _, stErr := os.Stat(dest + ".ok"); !os.IsNotExist(stErr) {
		t.Fatalf(".ok marker after failed download: %v", stErr)
	}
}

func TestPull401MentionsToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("nope"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	_, err := c.Pull("org/gated", []string{"config.json"})
	if err == nil || !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Fatalf("want HF_TOKEN in error, got %v", err)
	}
}

func TestPullRetries503(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("after-retry"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	dir, err := c.Pull("org/tiny", []string{"config.json"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil || string(b) != "after-retry" {
		t.Fatalf("got %q err=%v", b, err)
	}
	if hits != 2 {
		t.Fatalf("hits=%d want 2", hits)
	}
}

func TestNewEnvCompatibility(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "https://mirror.example/")
	t.Setenv("QUIKAITOOLS_CACHE", "")
	t.Setenv("HUGGINGFACE_HUB_CACHE", "")
	t.Setenv("HF_HOME", filepath.Join(t.TempDir(), "hfhome"))
	c := New("")
	if c.Base != "https://mirror.example" {
		t.Fatalf("Base=%q", c.Base)
	}
	if !strings.HasSuffix(c.Cache, filepath.Join("hfhome", "hub")) {
		t.Fatalf("Cache=%q want $HF_HOME/hub", c.Cache)
	}

	t.Setenv("HUGGINGFACE_HUB_CACHE", "/tmp/hf-hub-cache")
	c = New("")
	if c.Cache != "/tmp/hf-hub-cache" {
		t.Fatalf("HUGGINGFACE_HUB_CACHE precedence: %q", c.Cache)
	}
	t.Setenv("QUIKAITOOLS_CACHE", "/tmp/q-cache")
	c = New("")
	if c.Cache != "/tmp/q-cache" {
		t.Fatalf("QUIKAITOOLS_CACHE precedence: %q", c.Cache)
	}
	c = New("/explicit")
	if c.Cache != "/explicit" {
		t.Fatalf("explicit cache: %q", c.Cache)
	}
}

func TestPullONNXRequiresWeight(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".onnx") {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	c := New(t.TempDir())
	c.Base = srv.URL
	c.HTTP = srv.Client()
	c.Backoff = 0
	_, err := c.Pull("org/onnx", []string{"config.json", "onnx/model.onnx", "model.onnx", "README.md"})
	if err == nil || !strings.Contains(err.Error(), "no model weights") {
		t.Fatalf("want weight error, got %v", err)
	}
}
