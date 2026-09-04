package gpt2

import "math"

// MatMul is a @ b with shapes (aR,aC) x (bR,bC).
func MatMul(a []float32, aR, aC int, b []float32, bR, bC int) []float32 {
	if aC != bR {
		panic("matmul shape")
	}
	out := make([]float32, aR*bC)
	for i := 0; i < aR; i++ {
		row := a[i*aC : (i+1)*aC]
		dst := out[i*bC : (i+1)*bC]
		for k := 0; k < aC; k++ {
			av := row[k]
			if av == 0 {
				continue
			}
			br := b[k*bC : (k+1)*bC]
			for j := 0; j < bC; j++ {
				dst[j] += av * br[j]
			}
		}
	}
	return out
}

func AddBias(m []float32, rows, cols int, bias []float32) {
	for i := 0; i < rows; i++ {
		row := m[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			row[j] += bias[j]
		}
	}
}

func GELUNew(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	inner := c * (float64(x) + 0.044715*float64(x)*float64(x)*float64(x))
	return float32(0.5 * float64(x) * (1 + math.Tanh(inner)))
}

func GELUNewDeriv(x float32) float32 {
	const c = 0.7978845608028654
	xf := float64(x)
	inner := c * (xf + 0.044715*xf*xf*xf)
	th := math.Tanh(inner)
	dinner := c * (1 + 3*0.044715*xf*xf)
	return float32(0.5*(1+th) + 0.5*xf*(1-th*th)*dinner)
}

func LayerNorm(x []float32, rows, cols int, w, b []float32, eps float32) (out, mean, rstd []float32) {
	out = make([]float32, len(x))
	mean = make([]float32, rows)
	rstd = make([]float32, rows)
	for i := 0; i < rows; i++ {
		row := x[i*cols : (i+1)*cols]
		var m float32
		for _, v := range row {
			m += v
		}
		m /= float32(cols)
		var v float32
		for _, z := range row {
			d := z - m
			v += d * d
		}
		rs := float32(1 / math.Sqrt(float64(v/float32(cols)+eps)))
		mean[i], rstd[i] = m, rs
		dst := out[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			dst[j] = (row[j]-m)*rs*w[j] + b[j]
		}
	}
	return out, mean, rstd
}

func LayerNormBwd(dy, x []float32, rows, cols int, w []float32, mean, rstd []float32) (dx, dw, db []float32) {
	dx = make([]float32, len(x))
	dw = make([]float32, cols)
	db = make([]float32, cols)
	for i := 0; i < rows; i++ {
		row := x[i*cols : (i+1)*cols]
		drow := dy[i*cols : (i+1)*cols]
		m, rs := mean[i], rstd[i]
		dhat := make([]float32, cols)
		for j := 0; j < cols; j++ {
			hat := (row[j] - m) * rs
			db[j] += drow[j]
			dw[j] += drow[j] * hat
			dhat[j] = drow[j] * w[j]
		}
		var sumDhat, sumDhatHat float32
		for j := 0; j < cols; j++ {
			hat := (row[j] - m) * rs
			sumDhat += dhat[j]
			sumDhatHat += dhat[j] * hat
		}
		inv := rs / float32(cols)
		for j := 0; j < cols; j++ {
			hat := (row[j] - m) * rs
			dx[i*cols+j] = inv * (float32(cols)*dhat[j] - sumDhat - hat*sumDhatHat)
		}
	}
	return dx, dw, db
}

func SoftmaxRows(logits []float32, rows, cols int) []float32 {
	out := make([]float32, len(logits))
	for i := 0; i < rows; i++ {
		row := logits[i*cols : (i+1)*cols]
		m := row[0]
		for _, v := range row[1:] {
			if v > m {
				m = v
			}
		}
		var sum float32
		dst := out[i*cols : (i+1)*cols]
		for j, v := range row {
			e := float32(math.Exp(float64(v - m)))
			dst[j] = e
			sum += e
		}
		inv := 1 / sum
		for j := range dst {
			dst[j] *= inv
		}
	}
	return out
}

func Transpose(a []float32, r, c int) []float32 {
	out := make([]float32, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}
