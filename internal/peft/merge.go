package peft

import "github.com/shaneburrell/quikaitools/internal/gpt2"

// MergeIntoBase folds LoRA deltas into base Conv1D weights (c_attn / c_fc) in place.
// After merge, adapters are no longer needed for forward.
func (m *Model) MergeIntoBase() {
	scale := m.Cfg.Scale()
	for i := range m.Base.Blocks {
		if i < len(m.Attn) && m.Attn[i] != nil {
			mergeLinear(m.Base.Blocks[i].AttnW, m.Attn[i], scale)
		}
		if i < len(m.FC) && m.FC[i] != nil {
			mergeLinear(m.Base.Blocks[i].FcW, m.FC[i], scale)
		}
	}
}

// W += scale * (A @ B) with A [in,r], B [r,out], W [in,out] row-major.
func mergeLinear(w []float32, a *Adapter, scale float32) {
	delta := matmul(a.A, a.In, a.Rank, a.B, a.Rank, a.Out)
	for i := range w {
		if i < len(delta) {
			w[i] += delta[i] * scale
		}
	}
}

// LoadAndMerge loads adapters and merges into a copy-friendly base (mutates base).
func LoadAndMerge(base *gpt2.Model, adapterDir string) error {
	m, err := Load(base, adapterDir)
	if err != nil {
		return err
	}
	m.MergeIntoBase()
	return nil
}
