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
func (c *Client) Pull(repo string, files []string) (string, error) {
	if repo == "" {
		return "", fmt.Errorf("hub: empty repo")
	}
	if len(files) == 0 {
		files = DefaultFiles
	}
	dir := c.ModelDir(repo)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	for _, name := range files {
		dest := filepath.Join(dir, name)
		if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
			continue
		}
		url := fmt.Sprintf("%s/%s/resolve/main/%s", strings.TrimRight(c.Base, "/"), repo, name)
		if err := c.download(url, dest); err != nil {
			if name == "tokenizer_config.json" && strings.Contains(err.Error(), "HTTP 404") {
				continue
			}
			return dir, fmt.Errorf("%s: %w", name, err)
		}
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
