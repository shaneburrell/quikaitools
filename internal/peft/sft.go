package peft

import (
	"math"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

type blockTape struct {
	x, ln1, qkv, attn, proj, x2, ln2, fc, act, mlp []float32
	ln1Mean, ln1Rstd, ln2Mean, ln2Rstd             []float32
	attnCache                                      gpt2.AttnCache
}

// Step runs next-token SFT on tokens (length T). Returns mean CE loss.
func (m *Model) StepLoss(tokens []int) float32 {
	if len(tokens) < 2 {
		return 0
	}
	m.ZeroGrad()
	loss, dx := m.forwardBackward(tokens)
	_ = dx
	m.Step()
	return loss
}

func (m *Model) forwardBackward(tokens []int) (float32, []float32) {
	base := m.Base
	cfg := base.Cfg
	t := len(tokens)
	d := cfg.NEmbd
	eps := cfg.LayerNormEps
	scale := m.Cfg.Scale()

	x := make([]float32, t*d)
	for i, id := range tokens {
		if id < 0 || id >= cfg.VocabSize {
			id = 0
		}
		copy(x[i*d:(i+1)*d], base.WTE[id*d:(id+1)*d])
		addVec(x[i*d:(i+1)*d], base.WPE[i*d:(i+1)*d])
	}

	tapes := make([]blockTape, len(base.Blocks))
	for li, blk := range base.Blocks {
		tp := &tapes[li]
		tp.x = append([]float32(nil), x...)
		tp.ln1, tp.ln1Mean, tp.ln1Rstd = gpt2.LayerNorm(x, t, d, blk.LN1.W, blk.LN1.B, eps)
		qkv := gpt2.MatMul(tp.ln1, t, d, blk.AttnW, d, 3*d)
		gpt2.AddBias(qkv, t, 3*d, blk.AttnB)
		addVec(qkv, m.Attn[li].apply(tp.ln1, t, scale))
		tp.qkv = qkv
		att, cache := gpt2.CausalAttn(qkv, t, d, cfg.NHead)
		tp.attn, tp.attnCache = att, cache
		proj := gpt2.MatMul(att, t, d, blk.ProjW, d, d)
		gpt2.AddBias(proj, t, d, blk.ProjB)
		tp.proj = proj
		x2 := append([]float32(nil), x...)
		addVec(x2, proj)
		tp.x2 = x2
		tp.ln2, tp.ln2Mean, tp.ln2Rstd = gpt2.LayerNorm(x2, t, d, blk.LN2.W, blk.LN2.B, eps)
		fc := gpt2.MatMul(tp.ln2, t, d, blk.FcW, d, cfg.Inner())
		gpt2.AddBias(fc, t, cfg.Inner(), blk.FcB)
		addVec(fc, m.FC[li].apply(tp.ln2, t, scale))
		tp.fc = fc
		act := make([]float32, len(fc))
		for i, v := range fc {
			act[i] = gpt2.GELUNew(v)
		}
		tp.act = act
		mlp := gpt2.MatMul(act, t, cfg.Inner(), blk.FcProjW, cfg.Inner(), d)
		gpt2.AddBias(mlp, t, d, blk.FcProjB)
		tp.mlp = mlp
		x = append([]float32(nil), x2...)
		addVec(x, mlp)
	}

	lnf, lnfMean, lnfRstd := gpt2.LayerNorm(x, t, d, base.LNF.W, base.LNF.B, eps)
	// logits = lnf @ WTE^T  → [T, V]
	wteT := gpt2.Transpose(base.WTE, cfg.VocabSize, d)
	logits := gpt2.MatMul(lnf, t, d, wteT, d, cfg.VocabSize)
	probs := gpt2.SoftmaxRows(logits, t, cfg.VocabSize)

	var loss float32
	dLogits := make([]float32, t*cfg.VocabSize)
	nPred := t - 1
	for i := 0; i < nPred; i++ {
		target := tokens[i+1]
		if target < 0 || target >= cfg.VocabSize {
			target = 0
		}
		p := probs[i*cfg.VocabSize+target]
		if p < 1e-12 {
			p = 1e-12
		}
		loss -= float32(math.Log(float64(p)))
		row := dLogits[i*cfg.VocabSize : (i+1)*cfg.VocabSize]
		copy(row, probs[i*cfg.VocabSize:(i+1)*cfg.VocabSize])
		row[target] -= 1
	}
	loss /= float32(nPred)
	invN := 1 / float32(nPred)
	for i := range dLogits {
		dLogits[i] *= invN
	}

	// dLnf = dLogits @ WTE
	dLnf := gpt2.MatMul(dLogits, t, cfg.VocabSize, base.WTE, cfg.VocabSize, d)
	dx, _, _ := gpt2.LayerNormBwd(dLnf, x, t, d, base.LNF.W, lnfMean, lnfRstd)

	for li := len(base.Blocks) - 1; li >= 0; li-- {
		blk := base.Blocks[li]
		tp := tapes[li]
		inn := cfg.Inner()
		// x_out = x2 + mlp  → dx2 += dx, dmlp = dx
		dx2 := append([]float32(nil), dx...)
		dmlp := dx
		// mlp = act @ FcProjW
		fcProjT := gpt2.Transpose(blk.FcProjW, inn, d)
		dAct := gpt2.MatMul(dmlp, t, d, fcProjT, d, inn)
		dfc := make([]float32, len(dAct))
		for i, v := range tp.fc {
			dfc[i] = dAct[i] * gpt2.GELUNewDeriv(v)
		}
		fcWT := gpt2.Transpose(blk.FcW, d, inn)
		dln2 := gpt2.MatMul(dfc, t, inn, fcWT, inn, d)
		addVec(dln2, m.FC[li].backward(tp.ln2, dfc, t, scale))
		dx2b, _, _ := gpt2.LayerNormBwd(dln2, tp.x2, t, d, blk.LN2.W, tp.ln2Mean, tp.ln2Rstd)
		addVec(dx2, dx2b)

		// x2 = x + proj
		dxIn := append([]float32(nil), dx2...)
		dproj := dx2
		projT := gpt2.Transpose(blk.ProjW, d, d)
		datt := gpt2.MatMul(dproj, t, d, projT, d, d)
		dqkv := tp.attnCache.Backward(datt)
		attnWT := gpt2.Transpose(blk.AttnW, d, 3*d)
		dln1 := gpt2.MatMul(dqkv, t, 3*d, attnWT, 3*d, d)
		addVec(dln1, m.Attn[li].backward(tp.ln1, dqkv, t, scale))
		dxb, _, _ := gpt2.LayerNormBwd(dln1, tp.x, t, d, blk.LN1.W, tp.ln1Mean, tp.ln1Rstd)
		addVec(dxIn, dxb)
		dx = dxIn
	}
	_ = lnfRstd
	return loss, dx
}

func addVec(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
