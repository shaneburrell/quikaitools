package train_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shaneburrell/quikaitools/internal/train"
)

func TestMessagesJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.jsonl")
	body := `{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	text, err := train.MessagesToText(path)
	if err != nil || text == "" {
		t.Fatalf("%q %v", text, err)
	}
	total, issues, err := train.ValidateMessagesJSONL(path)
	if err != nil || total != 1 || len(issues) != 0 {
		t.Fatalf("total=%d issues=%v err=%v", total, issues, err)
	}
}

// A bare non-JSON token used to make the decoder loop forever; it must be
// reported as one issue on the right line and the scan must terminate.
func TestValidateMessagesJSONLMalformedTerminates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.jsonl")
	body := "not-json\n" +
		`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}` + "\n" +
		"\n" +
		`{"messages":[{"role":"user"}]}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var total int
	var issues []string
	var err error
	go func() {
		total, issues, err = train.ValidateMessagesJSONL(path)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("ValidateMessagesJSONL did not terminate on malformed input")
	}
	if err != nil {
		t.Fatal(err)
	}
	if total != 2 {
		t.Fatalf("total=%d want 2 (blank line skipped, bad token not counted)", total)
	}
	if len(issues) != 2 || !strings.HasPrefix(issues[0], "line 1: invalid json") || !strings.HasPrefix(issues[1], "line 4:") {
		t.Fatalf("issues=%v", issues)
	}
}

func TestMessagesToPromptCompletions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.jsonl")
	body := `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	pairs, err := train.MessagesToPromptCompletions(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 {
		t.Fatalf("pairs=%d", len(pairs))
	}
	if pairs[0].Completion != "world" {
		t.Fatalf("completion=%q", pairs[0].Completion)
	}
	if pairs[0].Prompt != "sys\n\nhello\n\n" {
		t.Fatalf("prompt=%q", pairs[0].Prompt)
	}
}
