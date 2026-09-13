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
//
// merges.txt is listed so a default pull attempts it, but it is also in
// OptionalFiles: a 404 is skipped. tiny-random-gpt2 ships merges.txt; some
// vocab-only tokenizers do not.
var DefaultFiles = []string{
	"config.json",
	"model.safetensors",
	"vocab.json",
	"merges.txt",
	"tokenizer_config.json",
}

// OptionalFiles may 404 without failing the pull.
//
// merges.txt is optional even though it appears in DefaultFiles (see above).
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
	// Backoff is the initial retry delay (500ms, then 1s, then 2s).
	// Zero skips sleeps so tests can exercise 429/5xx retries instantly.
	Backoff time.Duration
}

// New returns a client that caches under dir, else the first of
// QUIKAITOOLS_CACHE, HUGGINGFACE_HUB_CACHE, $HF_HOME/hub, ~/.cache/quikaitools.
// Base is HF_ENDPOINT (trailing "/" stripped) or https://huggingface.co.
func New(cacheDir string) *Client {
	return &Client{
		Base:    defaultBase(),
		HTTP:    &http.Client{Timeout: 2 * time.Minute},
		Cache:   defaultCacheDir(cacheDir),
		Backoff: 500 * time.Millisecond,
	}
}

func defaultBase() string {
	if v := strings.TrimSpace(os.Getenv("HF_ENDPOINT")); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://huggingface.co"
}

func defaultCacheDir(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if v := os.Getenv("QUIKAITOOLS_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "quikaitools")
}

func hubToken() string {
	if t := strings.TrimSpace(os.Getenv("HF_TOKEN")); t != "" {
		return t
	}
	return strings.TrimSpace(os.Getenv("HUGGINGFACE_HUB_TOKEN"))
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
	gotWeight := 0
	wantedWeight := false
	for _, name := range files {
		if isWeightFile(name) {
			wantedWeight = true
		}
		dest := filepath.Join(dir, name)
		if rel, err := filepath.Rel(dir, dest); err != nil || strings.HasPrefix(rel, "..") {
			return dir, fmt.Errorf("hub: refused path %q", name)
		}
		// Cache hit: not a .tmp leftover, size>0, and either dest+".ok" exists
		// (verified download) or the file is a legacy download (marker absent).
		if cacheValid(dest) {
			got++
			if isWeightFile(name) {
				gotWeight++
			}
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
		if isWeightFile(name) {
			gotWeight++
		}
	}
	if got == 0 {
		return dir, fmt.Errorf("hub: no files downloaded for %s", repo)
	}
	if wantedWeight && gotWeight == 0 {
		return dir, fmt.Errorf("hub: no model weights downloaded for %s", repo)
	}
	return dir, nil
}

func isWeightFile(name string) bool {
	n := strings.ToLower(name)
	return strings.HasSuffix(n, ".safetensors") ||
		strings.HasSuffix(n, ".onnx") ||
		strings.HasSuffix(n, ".gguf") ||
		strings.HasSuffix(n, ".bin")
}

// cacheValid reports whether dest is a usable cache hit.
//
// A hit is valid only if dest does not end in ".tmp", the file exists with
// size>0, and either a sidecar dest+".ok" is present (written after a verified
// download) or the file is a legacy download (marker absent AND size>0).
func cacheValid(dest string) bool {
	if strings.HasSuffix(dest, ".tmp") {
		return false
	}
	st, err := os.Stat(dest)
	if err != nil || st.Size() <= 0 {
		return false
	}
	if _, err := os.Stat(dest + ".ok"); err == nil {
		return true
	}
	// Legacy download from a client that did not write the .ok marker.
	return true
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
	var last error
	for attempt := 0; attempt < 4; attempt++ {
		retry, err := c.downloadOnce(url, dest)
		if err == nil {
			return nil
		}
		last = err
		if !retry || attempt == 3 {
			return err
		}
		if c.Backoff > 0 {
			time.Sleep(c.Backoff << uint(attempt)) // 500ms, 1s, 2s
		}
	}
	return last
}

func (c *Client) downloadOnce(url, dest string) (retry bool, err error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return false, err
	}
	req.Header.Set("User-Agent", "quikaitools/0.5.1")
	if tok := hubToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	res, err := c.HTTP.Do(req)
	if err != nil {
		return false, err
	}
	defer func() { _ = res.Body.Close() }()

	switch {
	case res.StatusCode == http.StatusUnauthorized || res.StatusCode == http.StatusForbidden:
		return false, fmt.Errorf("HTTP %d for %s: repo may be private or gated; set HF_TOKEN", res.StatusCode, url)
	case res.StatusCode == http.StatusTooManyRequests || res.StatusCode >= 500:
		body, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return true, fmt.Errorf("HTTP %d for %s: %s", res.StatusCode, url, strings.TrimSpace(string(body)))
	case res.StatusCode != http.StatusOK:
		body, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return false, fmt.Errorf("HTTP %d for %s: %s", res.StatusCode, url, strings.TrimSpace(string(body)))
	}

	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return false, err
	}
	n, copyErr := io.Copy(f, res.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return false, copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return false, closeErr
	}
	if res.ContentLength >= 0 && n != res.ContentLength {
		_ = os.Remove(tmp)
		return false, fmt.Errorf("hub: truncated download: got %d bytes, Content-Length %d", n, res.ContentLength)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return false, err
	}
	// Sidecar written only after a verified download. See cacheValid.
	_ = os.WriteFile(dest+".ok", []byte("ok\n"), 0o644)
	return false, nil
}
