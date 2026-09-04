package peft

import (
	"slices"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestSampleTemperatureZeroMatchesGenerate(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4})
	prompt := []int{1, 2, 3}
	g := m.Generate(prompt, 5)
	s := m.Sample(prompt, 5, SampleOptions{Temperature: 0, EOS: -1})
	if !slices.Equal(g, s) {
		t.Fatalf("generate=%v sample=%v", g, s)
	}
}

func TestSampleSeedReproducible(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4})
	prompt := []int{1, 2}
	a := m.Sample(prompt, 8, SampleOptions{Temperature: 1, Seed: 42, EOS: -1})
	b := m.Sample(prompt, 8, SampleOptions{Temperature: 1, Seed: 42, EOS: -1})
	if !slices.Equal(a, b) {
		t.Fatalf("same seed differed: %v vs %v", a, b)
	}
}

func TestSampleEOSStopsEarly(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 7)
	m := Wrap(base, Config{Rank: 2, Alpha: 4})
	prompt := []int{1, 2, 3}
	first := m.Generate(prompt, 1)
	eos := first[len(first)-1]
	out := m.Sample(prompt, 8, SampleOptions{Temperature: 0, EOS: eos})
	if len(out) != len(prompt)+1 {
		t.Fatalf("len=%d want %d (%v)", len(out), len(prompt)+1, out)
	}
	if out[len(out)-1] != eos {
		t.Fatalf("last=%d want EOS %d", out[len(out)-1], eos)
	}
}
