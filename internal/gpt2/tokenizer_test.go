package gpt2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSplitAndDecodeRoundTripSpace(t *testing.T) {
	tk := &Tokenizer{
		TokenToID: map[string]int{"h": 1, "i": 2, "Ġ": 3, "<|endoftext|>": 0},
		IDToToken: []string{"<|endoftext|>", "h", "i", "Ġ"},
		Rank:      map[[2]string]int{},
		EOSID:     0,
		UnkID:     0,
	}
	ids := tk.Encode("hi hi")
	if len(ids) < 2 {
		t.Fatalf("ids=%v", ids)
	}
	if got := tk.Decode([]int{1, 2, 3, 1}); got == "" {
		t.Fatal("empty decode")
	}
}

func TestNewRandomInner(t *testing.T) {
	m := NewRandom(Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 8}, 3)
	if m.Cfg.Inner() != 32 || m.Cfg.HeadDim() != 4 {
		t.Fatalf("inner=%d head=%d", m.Cfg.Inner(), m.Cfg.HeadDim())
	}
}

func TestBytesToUnicode(t *testing.T) {
	if byteToUnicode[0x20] != 'Ġ' {
		t.Fatalf("0x20 -> %U want U+0120 Ġ", byteToUnicode[0x20])
	}
	if byteToUnicode[0x0A] != 'Ċ' {
		t.Fatalf("0x0A -> %U want U+010A Ċ", byteToUnicode[0x0A])
	}
	if byteToUnicode[0x21] != '!' {
		t.Fatalf("0x21 -> %q want !", byteToUnicode[0x21])
	}
	if byteToUnicode[0xFF] != 'ÿ' {
		t.Fatalf("0xFF -> %q want ÿ", byteToUnicode[0xFF])
	}
	if len(unicodeToByte) != 256 {
		t.Fatalf("decoder size %d want 256", len(unicodeToByte))
	}
	seen := make(map[rune]bool, 256)
	for b := 0; b < 256; b++ {
		r := byteToUnicode[b]
		if seen[r] {
			t.Fatalf("not injective: byte %d and another map to %U", b, r)
		}
		seen[r] = true
		got, ok := unicodeToByte[r]
		if !ok || got != byte(b) {
			t.Fatalf("decoder[%U]=%d ok=%v want %d", r, got, ok, b)
		}
	}
}

func TestPretokenize(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"a  b", []string{"a", " ", " b"}},
		{"it's 42!\n", []string{"it", "'s", " 42", "!", "\n"}},
	}
	for _, tc := range cases {
		got := pretokenize(tc.in)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("pretokenize(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
	got := pretokenize("héllo  wörld\tx")
	if !containsToken(got, "\t") {
		t.Fatalf("tab not preserved: %q", got)
	}
	want := []string{"héllo", " ", " wörld", "\t", "x"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("pretokenize(héllo…)=%q want %q", got, want)
	}
}

func TestPretokenizePromptCompletionBoundary(t *testing.T) {
	prompt := "sys\n\nhello\n\n"
	completion := "world"
	joint := pretokenize(prompt + completion)
	var split []string
	split = append(split, pretokenize(prompt)...)
	split = append(split, pretokenize(completion)...)
	if reflect.DeepEqual(joint, split) {
		t.Fatalf("expected pretok of prompt+completion to differ from concat, joint=%q split=%q", joint, split)
	}
}

func containsToken(toks []string, want string) bool {
	for _, t := range toks {
		if t == want {
			return true
		}
	}
	return false
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	tk := loadSynthetic(t, nil)
	texts := []string{
		"Hello world",
		"café",
		"日本語",
		"emoji 🎉",
		"line1\nline2\n",
		"tab\there",
		"  leading and trailing  ",
		"it's 42!\n",
		"héllo  wörld\tx",
	}
	for _, s := range texts {
		got := tk.Decode(tk.Encode(s))
		if got != s {
			t.Errorf("roundtrip %q -> %q ids=%v", s, got, tk.Encode(s))
		}
	}
}

func TestBPERankOrder(t *testing.T) {
	tk := loadSynthetic(t, []string{"h e", "he l"})
	ids := tk.Encode("hello")
	got := make([]string, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(tk.IDToToken) {
			t.Fatalf("bad id %d", id)
		}
		got[i] = tk.IDToToken[id]
	}
	want := []string{"hel", "l", "o"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("hello -> %q want %q (ids=%v)", got, want, ids)
	}
}

func TestTinyRandomGPT2Golden(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	dir := filepath.Join(home, ".cache", "quikaitools", "models", "hf-internal-testing", "tiny-random-gpt2")
	if _, err := os.Stat(filepath.Join(dir, "vocab.json")); err != nil {
		t.Skip("tiny-random-gpt2 not cached")
	}
	tk, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	s := "Hello, café.\n日本語 and a newline end.\n"
	if got := tk.Decode(tk.Encode(s)); got != s {
		t.Fatalf("roundtrip %q -> %q", s, got)
	}
	if len(tk.IDToToken) == 50257 {
		ids := tk.Encode("Hello world")
		if len(ids) != 2 {
			t.Fatalf("Encode(Hello world)=%v want 2 tokens", ids)
		}
	}
}

func loadSynthetic(t *testing.T, merges []string) *Tokenizer {
	t.Helper()
	dir := t.TempDir()
	vocab := map[string]int{}
	for b := 0; b < 256; b++ {
		vocab[string(byteToUnicode[b])] = b
	}
	next := 256
	vocab["<|endoftext|>"] = next
	next++
	seen := map[string]bool{}
	var mergeLines []string
	for _, m := range merges {
		parts := strings.Split(m, " ")
		if len(parts) != 2 {
			t.Fatalf("bad merge %q", m)
		}
		merged := parts[0] + parts[1]
		if !seen[merged] {
			vocab[merged] = next
			next++
			seen[merged] = true
		}
		mergeLines = append(mergeLines, m)
	}
	raw, err := json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "vocab.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	body := "#version: 0.2\n"
	if len(mergeLines) > 0 {
		body += strings.Join(mergeLines, "\n") + "\n"
	}
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}
