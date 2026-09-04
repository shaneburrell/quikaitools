package safetensors

import "math"

func float32frombitsImpl(u uint32) float32 {
	return math.Float32frombits(u)
}

// float16ToFloat32 converts IEEE-754 binary16 bits to float32.
// Subnormals, signed zeros, infinities, and NaN payloads are preserved.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// F16 subnormal: 2^{-14} * (frac/2^{10}). Normalize into F32.
		exp32 := uint32(127 - 14)
		for frac&0x0400 == 0 {
			frac <<= 1
			exp32--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | (exp32 << 23) | (frac << 13))
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (frac << 13))
	default:
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (frac << 13))
	}
}

// bfloat16ToFloat32 expands BF16 by shifting into the F32 high half.
func bfloat16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}
