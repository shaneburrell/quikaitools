package gpt2

import "math"

// causalAttn: qkv is [T, 3D]. Returns [T, D] and caches for backward.
func CausalAttn(qkv []float32, t, d, nHead int) (out []float32, cache AttnCache) {
	hd := d / nHead
	q := make([]float32, t*d)
	k := make([]float32, t*d)
	v := make([]float32, t*d)
	for i := 0; i < t; i++ {
		copy(q[i*d:(i+1)*d], qkv[i*3*d:i*3*d+d])
		copy(k[i*d:(i+1)*d], qkv[i*3*d+d:i*3*d+2*d])
		copy(v[i*d:(i+1)*d], qkv[i*3*d+2*d:i*3*d+3*d])
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	scores := make([]float32, nHead*t*t)
	for h := 0; h < nHead; h++ {
		for i := 0; i < t; i++ {
			qi := q[i*d+h*hd : i*d+h*hd+hd]
			for j := 0; j <= i; j++ {
				kj := k[j*d+h*hd : j*d+h*hd+hd]
				var dot float32
				for u := 0; u < hd; u++ {
					dot += qi[u] * kj[u]
				}
				scores[(h*t+i)*t+j] = dot * scale
			}
			for j := i + 1; j < t; j++ {
				scores[(h*t+i)*t+j] = -1e9
			}
		}
	}
	attn := SoftmaxRows(scores, nHead*t, t)
	out = make([]float32, t*d)
	for h := 0; h < nHead; h++ {
		for i := 0; i < t; i++ {
			dst := out[i*d+h*hd : i*d+h*hd+hd]
			for j := 0; j <= i; j++ {
				w := attn[(h*t+i)*t+j]
				vj := v[j*d+h*hd : j*d+h*hd+hd]
				for u := 0; u < hd; u++ {
					dst[u] += w * vj[u]
				}
			}
		}
	}
	cache = AttnCache{Q: q, K: k, V: v, Attn: attn, T: t, D: d, Heads: nHead, HD: hd, Scale: scale}
	return out, cache
}

type AttnCache struct {
	Q, K, V, Attn   []float32
	T, D, Heads, HD int
	Scale           float32
}

func (c AttnCache) Backward(dout []float32) (dqkv []float32) {
	t, d, nHead, hd := c.T, c.D, c.Heads, c.HD
	dQ := make([]float32, t*d)
	dK := make([]float32, t*d)
	dV := make([]float32, t*d)
	dAttn := make([]float32, nHead*t*t)
	for h := 0; h < nHead; h++ {
		for i := 0; i < t; i++ {
			do := dout[i*d+h*hd : i*d+h*hd+hd]
			for j := 0; j <= i; j++ {
				w := c.Attn[(h*t+i)*t+j]
				vj := c.V[j*d+h*hd : j*d+h*hd+hd]
				var dot float32
				for u := 0; u < hd; u++ {
					dV[j*d+h*hd+u] += w * do[u]
					dot += do[u] * vj[u]
				}
				dAttn[(h*t+i)*t+j] = dot
			}
		}
	}
	// softmax bwd per row
	dScores := make([]float32, nHead*t*t)
	for r := 0; r < nHead*t; r++ {
		s := c.Attn[r*t : (r+1)*t]
		ds := dAttn[r*t : (r+1)*t]
		var sum float32
		for j := 0; j < t; j++ {
			sum += s[j] * ds[j]
		}
		for j := 0; j < t; j++ {
			dScores[r*t+j] = s[j] * (ds[j] - sum)
		}
	}
	for h := 0; h < nHead; h++ {
		for i := 0; i < t; i++ {
			qi := c.Q[i*d+h*hd : i*d+h*hd+hd]
			for j := 0; j <= i; j++ {
				g := dScores[(h*t+i)*t+j] * c.Scale
				kj := c.K[j*d+h*hd : j*d+h*hd+hd]
				for u := 0; u < hd; u++ {
					dQ[i*d+h*hd+u] += g * kj[u]
					dK[j*d+h*hd+u] += g * qi[u]
				}
			}
		}
	}
	dqkv = make([]float32, t*3*d)
	for i := 0; i < t; i++ {
		copy(dqkv[i*3*d:i*3*d+d], dQ[i*d:(i+1)*d])
		copy(dqkv[i*3*d+d:i*3*d+2*d], dK[i*d:(i+1)*d])
		copy(dqkv[i*3*d+2*d:i*3*d+3*d], dV[i*d:(i+1)*d])
	}
	return dqkv
}
