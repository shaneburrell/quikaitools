package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/backend"
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
		if _, err := m.AccumulateLoss(ids, zero); err != nil {
			t.Fatal(err)
		}
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

func TestRunLoRASeedDeterministic(t *testing.T) {
	dir := t.TempDir()
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 3)
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	data := filepath.Join(dir, "data.txt")
	if err := os.WriteFile(data, []byte("abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz"), 0o644); err != nil {
		t.Fatal(err)
	}
	run := func(seed int64) []float32 {
		t.Helper()
		losses, err := RunLoRA(LoRAOptions{
			ModelDir: dir, DataPath: data,
			Steps: 6, SeqLen: 8, Rank: 2, Alpha: 4, LR: 1e-2,
			Seed: seed, Profile: backend.KindCPU,
		})
		if err != nil {
			t.Fatal(err)
		}
		return losses
	}
	a := run(42)
	b := run(42)
	if len(a) != len(b) {
		t.Fatalf("len %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed step %d: %v vs %v", i, a[i], b[i])
		}
	}
	c := run(99)
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seeds produced identical losses")
	}
}
