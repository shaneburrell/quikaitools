package peft

import "math"

// QuantLinear is a 4-bit packed frozen weight [in, out] with per-column scale.
type QuantLinear struct {
	In, Out int
	// Packed nibbles: two weights per byte, length (In*Out+1)/2
	Packed []byte
	Scale  []float32 // [Out]
}

// QuantizeConv1D packs W [in, out] to uint4 with absmax scale per output column.
func QuantizeConv1D(w []float32, in, out int) QuantLinear {
	q := QuantLinear{In: in, Out: out, Packed: make([]byte, (in*out+1)/2), Scale: make([]float32, out)}
	for j := 0; j < out; j++ {
		amax := float32(0)
		for i := 0; i < in; i++ {
			v := float32(math.Abs(float64(w[i*out+j])))
			if v > amax {
				amax = v
			}
		}
		if amax < 1e-8 {
			amax = 1
		}
		q.Scale[j] = amax / 7
		for i := 0; i < in; i++ {
			idx := i*out + j
			n := int8(math.Round(float64(w[idx] / q.Scale[j])))
			if n > 7 {
				n = 7
			}
			if n < -8 {
				n = -8
			}
			u := uint8(n + 8) // 0..15
			byteIdx := idx / 2
			if idx%2 == 0 {
				q.Packed[byteIdx] = (q.Packed[byteIdx] & 0xF0) | u
			} else {
				q.Packed[byteIdx] = (q.Packed[byteIdx] & 0x0F) | (u << 4)
			}
		}
	}
	return q
}

// MatMul dequants on the fly: y = x @ W_hat, x [rows,in], out [rows,out].
func (q QuantLinear) MatMul(x []float32, rows int) []float32 {
	out := make([]float32, rows*q.Out)
	for r := 0; r < rows; r++ {
		xr := x[r*q.In : (r+1)*q.In]
		dst := out[r*q.Out : (r+1)*q.Out]
		for j := 0; j < q.Out; j++ {
			var sum float32
			s := q.Scale[j]
			for i := 0; i < q.In; i++ {
				idx := i*q.Out + j
				byteIdx := idx / 2
				var u uint8
				if idx%2 == 0 {
					u = q.Packed[byteIdx] & 0x0F
				} else {
					u = (q.Packed[byteIdx] >> 4) & 0x0F
				}
				w := (float32(int8(u) - 8)) * s
				sum += xr[i] * w
			}
			dst[j] = sum
		}
	}
	return out
}

// Dequant expands packed weights to FP32 [in, out] matching Conv1D layout.
func (q QuantLinear) Dequant() []float32 {
	w := make([]float32, q.In*q.Out)
	for j := 0; j < q.Out; j++ {
		s := q.Scale[j]
		for i := 0; i < q.In; i++ {
			idx := i*q.Out + j
			byteIdx := idx / 2
			var u uint8
			if idx%2 == 0 {
				u = q.Packed[byteIdx] & 0x0F
			} else {
				u = (q.Packed[byteIdx] >> 4) & 0x0F
			}
			w[idx] = (float32(int8(u) - 8)) * s
		}
	}
	return w
}

// EnableQLoRA replaces AttnW/FcW matmuls with 4-bit packed weights (base frozen).
func (m *Model) EnableQLoRA() {
	m.useQ = true
	m.qAttn = make([]QuantLinear, len(m.Base.Blocks))
	m.qFC = make([]QuantLinear, len(m.Base.Blocks))
	d := m.Base.Cfg.NEmbd
	inn := m.Base.Cfg.Inner()
	for i, blk := range m.Base.Blocks {
		m.qAttn[i] = QuantizeConv1D(blk.AttnW, d, 3*d)
		m.qFC[i] = QuantizeConv1D(blk.FcW, d, inn)
	}
}

// IsQLoRA reports whether 4-bit base packing is enabled.
func (m *Model) IsQLoRA() bool { return m.useQ }
