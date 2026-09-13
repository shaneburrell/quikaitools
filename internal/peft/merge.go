package peft

import "github.com/shaneburrell/quikaitools/internal/gpt2"

// MergeIntoBase folds LoRA deltas into base Conv1D weights (c_attn / c_fc) in place.
// After merge, adapters are no longer needed for forward.
//
// QLoRA trains against 4-bit packed weights, so merge dequantizes those
// packed tensors first. Adding the adapter onto the original FP32 tensors
// would emit a checkpoint that does not match QLoRA inference.
func (m *Model) MergeIntoBase() {
	scale := m.Cfg.Scale()
	for i := range m.Base.Blocks {
		if m.useQ {
			if i < len(m.qAttn) {
				m.Base.Blocks[i].AttnW = m.qAttn[i].Dequant()
			}
			if i < len(m.qFC) {
				m.Base.Blocks[i].FcW = m.qFC[i].Dequant()
			}
		}
		if i < len(m.Attn) && m.Attn[i] != nil {
			mergeLinear(m.Base.Blocks[i].AttnW, m.Attn[i], scale)
		}
		if i < len(m.FC) && m.FC[i] != nil {
			mergeLinear(m.Base.Blocks[i].FcW, m.FC[i], scale)
		}
	}
	m.useQ = false
	m.qAttn = nil
	m.qFC = nil
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

// LoadAndMerge loads adapters (legacy adapter.json or HF PEFT files) and merges into base.
func LoadAndMerge(base *gpt2.Model, adapterDir string) error {
	m, err := Load(base, adapterDir)
	if err != nil {
		return err
	}
	m.MergeIntoBase()
	return nil
}
