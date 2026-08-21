package peft

import (
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestLoRAInitIsIdentityDelta(t *testing.T) {
	a := newAdapter("t", 4, 6, 2)
	x := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	y := a.apply(x, 2, 1)
	for _, v := range y {
		if v != 0 {
			t.Fatalf("zero-B LoRA should be 0, got %v", y)
		}
	}
}

func TestWrapCounts(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 2, NPositions: 16, VocabSize: 16, NInner: 16}, 2)
	m := Wrap(base, Config{Rank: 2, Alpha: 4})
	if len(m.Attn) != 2 || len(m.FC) != 2 {
		t.Fatalf("adapters attn=%d fc=%d", len(m.Attn), len(m.FC))
	}
	if m.TrainableParams() != 2*(8*2+2*24)+2*(8*2+2*16) {
		t.Fatalf("params %d", m.TrainableParams())
	}
}
