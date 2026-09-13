package peft

import (
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestSaveLoadGenerate(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	if _, err := m.StepLoss([]int{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
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
	if _, err := m.StepLoss([]int{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
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

func TestGenerateEmptyPromptAndZeroPositions(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := BaseOnly(base)
	out := m.Generate(nil, 4)
	if len(out) != 5 {
		t.Fatalf("empty prompt len=%d want 5 (seed + 4)", len(out))
	}
	sampled := m.Sample(nil, 3, SampleOptions{Seed: 1, EOS: -1})
	if len(sampled) != 4 {
		t.Fatalf("empty sample len=%d want 4", len(sampled))
	}
	base.Cfg.NPositions = 0
	out = m.Generate([]int{1, 2}, 2)
	if len(out) != 4 {
		t.Fatalf("n_positions=0 len=%d want 4", len(out))
	}
}
