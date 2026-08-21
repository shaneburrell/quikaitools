package chatfmt_test

import (
	"strings"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/chatfmt"
)

func TestChatML(t *testing.T) {
	out, err := chatfmt.Apply("chatml", []chatfmt.Message{
		{Role: "system", Content: "Be careful."},
		{Role: "user", Content: "Hi"},
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "<|im_start|>system") || !strings.Contains(out, "<|im_start|>assistant\n") {
		t.Fatalf("got %q", out)
	}
}

func TestParseAndPrompt(t *testing.T) {
	msgs, err := chatfmt.ParseMessagesJSON(`[{"role":"user","content":"a"}]`)
	if err != nil || len(msgs) != 1 {
		t.Fatalf("%v %#v", err, msgs)
	}
	out, err := chatfmt.Apply("raw", msgs, "b")
	if err != nil || out != "a\nb" {
		t.Fatalf("out=%q err=%v", out, err)
	}
}
