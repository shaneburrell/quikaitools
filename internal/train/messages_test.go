package train_test

import (
	"os"
	"path/filepath"
	"testing"

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
