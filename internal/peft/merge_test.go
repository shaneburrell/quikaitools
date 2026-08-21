package peft_test

import (
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
	"github.com/shaneburrell/quikaitools/internal/testart"
)

func TestMergeIntoBase(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 32, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
	base := gpt2.NewRandom(cfg, 2)
	before := append([]float32(nil), base.Blocks[0].AttnW...)
	m := peft.Wrap(base, peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	// Force non-zero B so merge changes weights
	for i := range m.Attn[0].B {
		m.Attn[0].B[i] = 0.01
	}
	m.MergeIntoBase()
	changed := false
	for i := range before {
		if base.Blocks[0].AttnW[i] != before[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("expected merge to change AttnW")
	}
	dir := testart.Path(t, "merged-gpt2")
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
}
