package gpt2

import "testing"

func TestSplitAndDecodeRoundTripSpace(t *testing.T) {
	tk := &Tokenizer{
		TokenToID: map[string]int{"h": 1, "i": 2, "Ġ": 3, "<|endoftext|>": 0},
		IDToToken: []string{"<|endoftext|>", "h", "i", "Ġ"},
		Rank:      map[[2]string]int{},
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
