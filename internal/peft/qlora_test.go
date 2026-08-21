package peft

import (
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestQLoRATrain(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 9)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	m.EnableQLoRA()
	if !m.useQ || len(m.qAttn) == 0 {
		t.Fatal("qlora not enabled")
	}
	loss := m.StepLoss([]int{1, 2, 3, 4, 5, 6, 7, 8})
	if loss != loss {
		t.Fatalf("nan loss")
	}
	packed := 0
	for _, q := range m.qAttn {
		packed += len(q.Packed)
	}
	if packed == 0 {
		t.Fatal("no packed weights")
	}
}

func TestQuantizeRoundTripShape(t *testing.T) {
	w := make([]float32, 4*12)
	for i := range w {
		w[i] = float32(i%7) * 0.1
	}
	q := QuantizeConv1D(w, 4, 12)
	x := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	y := q.MatMul(x, 2)
	if len(y) != 24 {
		t.Fatalf("len=%d", len(y))
	}
}
