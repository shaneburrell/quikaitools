// Package hub downloads files from the Hugging Face Hub (no Python).
package hub

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultFiles are enough to LoRA a GPT-2-style checkpoint.
var DefaultFiles = []string{
	"config.json",
	"model.safetensors",
	"vocab.json",
	"merges.txt",
	"tokenizer_config.json",
}

// OptionalFiles may 404 without failing the pull.
var OptionalFiles = map[string]bool{
	"tokenizer_config.json": true,
	"tokenizer.json":        true,
	"vocab.txt":             true,
	"onnx/model.onnx":       true,
	"model.onnx":            true,
	"README.md":             true,
	"merges.txt":            true, // some tokenizers are vocab-only
}

// Client downloads Hub files.
type Client struct {
	Base  string
	HTTP  *http.Client
	Cache string
}

// New returns a client that caches under dir (or QUIKAITOOLS_CACHE / ~/.cache/quikaitools).
func New(cacheDir string) *Client {
	if cacheDir == "" {
		cacheDir = os.Getenv("QUIKAITOOLS_CACHE")
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "quikaitools")
	}
	return &Client{
		Base:  "https://huggingface.co",
		HTTP:  &http.Client{Timeout: 2 * time.Minute},
		Cache: cacheDir,
	}
}

// ModelDir is the local folder for a repo id (org/name).
func (c *Client) ModelDir(repo string) string {
	return filepath.Join(c.Cache, "models", filepath.FromSlash(repo))
}

// Pull downloads files into ModelDir(repo). Existing files are kept.
// Optional files (see OptionalFiles) may 404; required files must download.
func (c *Client) Pull(repo string, files []string) (string, error) {
	return c.PullOpts(repo, files, true)
}

// PullOpts is Pull with control over optional-file 404 skipping.
func (c *Client) PullOpts(repo string, files []string, skipOptional404 bool) (string, error) {
	if err := ValidateRepo(repo); err != nil {
		return "", err
	}
	if len(files) == 0 {
		files = DefaultFiles
	}
	dir, err := c.safeModelDir(repo)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	got := 0
	for _, name := range files {
		dest := filepath.Join(dir, name)
		if rel, err := filepath.Rel(dir, dest); err != nil || strings.HasPrefix(rel, "..") {
			return dir, fmt.Errorf("hub: refused path %q", name)
		}
		if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
			got++
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return dir, err
		}
		url := fmt.Sprintf("%s/%s/resolve/main/%s", strings.TrimRight(c.Base, "/"), repo, name)
		if err := c.download(url, dest); err != nil {
			optional := OptionalFiles[name]
			if skipOptional404 && optional && strings.Contains(err.Error(), "HTTP 404") {
				continue
			}
			return dir, fmt.Errorf("%s: %w", name, err)
		}
		got++
	}
	if got == 0 {
		return dir, fmt.Errorf("hub: no files downloaded for %s", repo)
	}
	return dir, nil
}

// ValidateRepo accepts Hugging Face org/name only (no path traversal).
func ValidateRepo(repo string) error {
	if repo == "" {
		return fmt.Errorf("hub: empty repo")
	}
	if strings.Contains(repo, "..") || strings.Contains(repo, `\`) || filepath.IsAbs(repo) {
		return fmt.Errorf("hub: invalid repo %q", repo)
	}
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("hub: repo must be org/name, got %q", repo)
	}
	for _, p := range parts {
		for _, r := range p {
			ok := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '-' || r == '_'
			if !ok {
				return fmt.Errorf("hub: invalid repo %q", repo)
			}
		}
	}
	return nil
}

func (c *Client) safeModelDir(repo string) (string, error) {
	root := filepath.Clean(filepath.Join(c.Cache, "models"))
	dir := filepath.Clean(filepath.Join(root, filepath.FromSlash(repo)))
	rel, err := filepath.Rel(root, dir)
	if err != nil || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return "", fmt.Errorf("hub: repo escapes cache: %q", repo)
	}
	return dir, nil
}

func (c *Client) download(url, dest string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "quikaitools/0.1")
	res, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("HTTP %d for %s: %s", res.StatusCode, url, strings.TrimSpace(string(body)))
	}
	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(f, res.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dest)
}
