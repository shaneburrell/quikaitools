package train

import (
	"fmt"
	"math"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
	"github.com/shaneburrell/quikaitools/internal/testart"
)

func TestRunLoRARandomLossFinite(t *testing.T) {
	losses, m, err := RunLoRARandom("the cat sat on the mat the cat sat", 4)
	if err != nil {
		t.Fatal(err)
	}
	if m.TrainableParams() == 0 {
		t.Fatal("no lora params")
	}
	line := ""
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("step %d loss=%v", i, l)
		}
		if i > 0 {
			line += " "
		}
		line += fmt.Sprintf("%.4f", l)
	}
	testart.WriteFile(t, "lora-random-loss.txt", []byte(line+"\n"))
}

func TestRunLoRARandomQLoRA(t *testing.T) {
	losses, m, err := RunLoRARandomOpts("the cat sat on the mat the cat sat", 3, true)
	if err != nil {
		t.Fatal(err)
	}
	if m.TrainableParams() == 0 {
		t.Fatal("no lora params")
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("step %d loss=%v", i, l)
		}
	}
}

func TestAccumAdamSteps(t *testing.T) {
	m := peft.Wrap(gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 5), peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8}
	accum := 2
	micro := 6
	var accumN int
	for step := 0; step < micro; step++ {
		zero := accumN == 0
		_ = m.AccumulateLoss(ids, zero)
		accumN++
		if accumN >= accum {
			m.ScaleGrads(1 / float32(accumN))
			m.Step()
			accumN = 0
		}
	}
	if accumN > 0 {
		m.ScaleGrads(1 / float32(accumN))
		m.Step()
	}
	want := micro / accum
	if m.AdamSteps() != want {
		t.Fatalf("AdamSteps=%d want %d", m.AdamSteps(), want)
	}
}
