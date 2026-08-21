package peft

import (
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestSaveLoadGenerate(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	_ = m.StepLoss([]int{1, 2, 3, 4, 5, 6})
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	base2 := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	loaded, err := Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := filepath.Abs(filepath.Join(dir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
	out := loaded.Generate([]int{1, 2, 3}, 4)
	if len(out) != 7 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestQLoRASaveLoadEnablesQuant(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	m.EnableQLoRA()
	_ = m.StepLoss([]int{1, 2, 3, 4, 5, 6})
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	base2 := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	loaded, err := Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.useQ || len(loaded.qAttn) == 0 {
		t.Fatal("expected qlora restored on Load")
	}
}

func TestBaseOnlyGenerate(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := BaseOnly(base)
	out := m.Generate([]int{1, 2}, 3)
	if len(out) != 5 {
		t.Fatalf("len=%d", len(out))
	}
	if m.TrainableParams() != 0 {
		t.Fatal("base-only should have no adapters")
	}
}
