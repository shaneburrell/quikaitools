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

	dir, err := c.Pull("org/tiny", []string{"config.json"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil || string(b) != "ok-bytes" {
		t.Fatalf("got %q err=%v", b, err)
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
	if _, err := c.Pull("org/tiny", []string{"config.json", "model.safetensors", "tokenizer_config.json"}); err == nil {
		t.Fatal("expected required model.safetensors 404 to fail")
	}
	// optional tokenizer_config alone with success on config should work
	if _, err := c.Pull("org/tiny2", []string{"config.json", "tokenizer_config.json"}); err != nil {
		t.Fatal(err)
	}
}
