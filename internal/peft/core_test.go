package peft

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func tinyBase(seed int64) *gpt2.Model {
	return gpt2.NewRandom(gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 64, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}, seed)
}

func TestLossDecreasesOnRepeatedSequence(t *testing.T) {
	base := tinyBase(1)
	// NewRandom WTE is ~0.02, so frozen unembed logits stay near 0 and CE
	// cannot leave log(V). Scale the head so hidden-state steering is visible.
	for i := range base.WTE {
		base.WTE[i] *= 15
	}
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	ids := []int{1, 2, 1, 2, 1, 2, 1, 2}
	var first, last float32
	for step := 0; step < 40; step++ {
		loss, err := m.StepLoss(ids)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss
		}
		last = loss
	}
	if last >= 0.7*first {
		t.Fatalf("loss did not drop enough: first=%.4f last=%.4f", first, last)
	}
}

func TestAdapterBackwardFiniteDiff(t *testing.T) {
	a := newAdapter("fd", 3, 2, 1)
	for i := range a.A {
		a.A[i] = 0.15 * float32(i+1)
	}
	for i := range a.B {
		a.B[i] = 0.25 * float32(i+1)
	}
	x := []float32{0.4, -0.2, 0.7, 0.1, 0.5, -0.3}
	const rows = 2
	const scale float32 = 0.5
	const eps = 1e-3

	y := a.apply(x, rows, scale)
	dy := make([]float32, len(y))
	for i := range dy {
		dy[i] = 1
	}
	a.zeroGrad()
	a.backward(x, dy, rows, scale)

	lossOf := func() float64 {
		z := a.apply(x, rows, scale)
		var s float64
		for _, v := range z {
			s += float64(v)
		}
		return s
	}
	check := func(name string, w, g []float32) {
		t.Helper()
		for i := range w {
			orig := w[i]
			w[i] = orig + eps
			lp := lossOf()
			w[i] = orig - eps
			lm := lossOf()
			w[i] = orig
			num := (lp - lm) / (2 * float64(eps))
			an := float64(g[i])
			den := math.Max(math.Abs(an), 1e-6)
			if math.Abs(num-an)/den > 1e-2 {
				t.Fatalf("%s[%d]: numeric=%.6f analytic=%.6f", name, i, num, an)
			}
		}
	}
	check("A", a.A, a.dA)
	check("B", a.B, a.dB)
}

func TestLayerNormBwdFiniteDiff(t *testing.T) {
	x := []float32{0.5, -0.3, 1.2, 0.1}
	w := []float32{1.1, 0.9, 1.0, 1.2}
	b := []float32{0.01, -0.02, 0, 0.03}
	const epsLN float32 = 1e-5
	const rows, cols = 1, 4
	out, mean, rstd := gpt2.LayerNorm(x, rows, cols, w, b, epsLN)
	dy := make([]float32, len(out))
	for i := range dy {
		dy[i] = 1
	}
	dx, dw, db := gpt2.LayerNormBwd(dy, x, rows, cols, w, mean, rstd)

	const eps = 1e-3
	lossX := func(xx []float32) float64 {
		o, _, _ := gpt2.LayerNorm(xx, rows, cols, w, b, epsLN)
		var s float64
		for _, v := range o {
			s += float64(v)
		}
		return s
	}
	rel := func(num, an float64) float64 {
		return math.Abs(num-an) / math.Max(math.Abs(an), 1e-6)
	}
	for i := range x {
		orig := x[i]
		x[i] = orig + eps
		lp := lossX(x)
		x[i] = orig - eps
		lm := lossX(x)
		x[i] = orig
		num := (lp - lm) / (2 * float64(eps))
		if rel(num, float64(dx[i])) > 1e-2 {
			t.Fatalf("dx[%d]: numeric=%.6f analytic=%.6f", i, num, dx[i])
		}
	}
	lossW := func(ww []float32) float64 {
		o, _, _ := gpt2.LayerNorm(x, rows, cols, ww, b, epsLN)
		var s float64
		for _, v := range o {
			s += float64(v)
		}
		return s
	}
	for i := range w {
		orig := w[i]
		w[i] = orig + eps
		lp := lossW(w)
		w[i] = orig - eps
		lm := lossW(w)
		w[i] = orig
		num := (lp - lm) / (2 * float64(eps))
		if rel(num, float64(dw[i])) > 1e-2 {
			t.Fatalf("dw[%d]: numeric=%.6f analytic=%.6f", i, num, dw[i])
		}
	}
	_ = db
}

func TestMergeForwardLogitsMatch(t *testing.T) {
	base := tinyBase(2)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	for i := 0; i < 3; i++ {
		if _, err := m.StepLoss([]int{1, 2, 3, 4, 5, 6}); err != nil {
			t.Fatal(err)
		}
	}
	toks := []int{1, 2, 3, 4}
	lora := append([]float32(nil), m.ForwardLogits(toks)...)
	m.MergeIntoBase()
	merged := BaseOnly(m.Base)
	got := merged.ForwardLogits(toks)
	if len(lora) != len(got) {
		t.Fatalf("len %d vs %d", len(lora), len(got))
	}
	for i := range lora {
		if math.Abs(float64(lora[i]-got[i])) > 1e-4 {
			t.Fatalf("logit[%d]: lora=%.6f merged=%.6f", i, lora[i], got[i])
		}
	}
}

func TestResumeRestoresAdamAndContinues(t *testing.T) {
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8}
	base := tinyBase(3)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	const n = 8
	var losses []float32
	for i := 0; i < n; i++ {
		loss, err := m.StepLoss(ids)
		if err != nil {
			t.Fatal(err)
		}
		losses = append(losses, loss)
	}
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "optimizer.json")); err != nil {
		t.Fatal(err)
	}
	base2 := tinyBase(3)
	loaded, err := Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.AdamSteps() != n {
		t.Fatalf("AdamSteps=%d want %d", loaded.AdamSteps(), n)
	}
	cont, err := loaded.StepLoss(ids)
	if err != nil {
		t.Fatal(err)
	}
	last := losses[len(losses)-1]
	if cont > last*1.8+0.25 {
		t.Fatalf("loss spiked after resume: last=%.4f continued=%.4f", last, cont)
	}
}

func TestForwardLogitsErrTooLong(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 16, NInner: 16}, 1)
	m := Wrap(base, Config{Rank: 2, Alpha: 4})
	toks := []int{1, 2, 3, 4, 5, 6}
	if _, err := m.ForwardLogitsErr(toks); err == nil {
		t.Fatal("expected error")
	}
	got := m.ForwardLogits(toks)
	want := m.ForwardLogits(toks[len(toks)-4:])
	if len(got) != len(want) {
		t.Fatalf("truncate len %d vs %d", len(got), len(want))
	}
	if _, err := m.forwardBackward(toks, nil); err == nil {
		t.Fatal("expected forwardBackward error")
	}
}

func TestAccumulateLossMaskedNone(t *testing.T) {
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 16, NInner: 16}, 1)
	m := Wrap(base, Config{Rank: 2, Alpha: 4, LR: 1e-2})
	ids := []int{1, 2, 3, 4}
	mask := []bool{false, false, false, false}
	a0 := append([]float32(nil), m.Attn[0].A...)
	loss, err := m.AccumulateLossMasked(ids, mask, true)
	if err != nil {
		t.Fatal(err)
	}
	if loss != 0 {
		t.Fatalf("loss=%v", loss)
	}
	for i := range a0 {
		if m.Attn[0].A[i] != a0[i] || m.Attn[0].dA[i] != 0 {
			t.Fatal("expected no grads when mask is empty")
		}
	}
}
