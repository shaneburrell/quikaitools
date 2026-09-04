package peft

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

// Load restores adapters from adapter.json onto base.
// optimizer.json is restored when present (ignored if absent).
func Load(base *gpt2.Model, dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "adapter.json"))
	if err != nil {
		return nil, err
	}
	type dump struct {
		Name string    `json:"name"`
		In   int       `json:"in"`
		Out  int       `json:"out"`
		Rank int       `json:"rank"`
		A    []float32 `json:"a"`
		B    []float32 `json:"b"`
	}
	var file struct {
		LoRA     Config `json:"lora"`
		QLoRA    bool   `json:"qlora"`
		Adapters []dump `json:"adapters"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, err
	}
	m := Wrap(base, file.LoRA)
	byName := map[string]*Adapter{}
	for _, a := range m.adapters() {
		byName[a.Name] = a
	}
	for _, d := range file.Adapters {
		a, ok := byName[d.Name]
		if !ok {
			return nil, fmt.Errorf("peft: unknown adapter %q", d.Name)
		}
		if len(d.A) != len(a.A) || len(d.B) != len(a.B) {
			return nil, fmt.Errorf("peft: shape mismatch for %s", d.Name)
		}
		copy(a.A, d.A)
		copy(a.B, d.B)
		a.Rank = d.Rank
	}
	if file.QLoRA {
		m.EnableQLoRA()
	}
	if err := m.restoreOptimizer(dir); err != nil {
		return nil, err
	}
	return m, nil
}

// BaseOnly wraps a frozen GPT-2 with no LoRA adapters (identity generate path).
func BaseOnly(base *gpt2.Model) *Model {
	return &Model{Base: base, Cfg: Config{Rank: 0}}
}

// ForwardLogits runs the model once and returns logits [T, V].
// Sequences longer than n_positions are truncated to the last n_positions tokens.
func (m *Model) ForwardLogits(tokens []int) []float32 {
	if npos := m.Base.Cfg.NPositions; npos > 0 && len(tokens) > npos {
		tokens = tokens[len(tokens)-npos:]
	}
	logits, _ := m.forwardLogits(tokens)
	return logits
}

// ForwardLogitsErr is ForwardLogits but returns an error instead of truncating
// when len(tokens) exceeds cfg.NPositions.
func (m *Model) ForwardLogitsErr(tokens []int) ([]float32, error) {
	if npos := m.Base.Cfg.NPositions; npos > 0 && len(tokens) > npos {
		return nil, fmt.Errorf("peft: sequence length %d exceeds n_positions %d", len(tokens), npos)
	}
	return m.forwardLogits(tokens)
}

func (m *Model) forwardLogits(tokens []int) ([]float32, error) {
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

	for li, blk := range base.Blocks {
		ln1, _, _ := gpt2.LayerNorm(x, t, d, blk.LN1.W, blk.LN1.B, eps)
		var qkv []float32
		if m.useQ {
			qkv = m.qAttn[li].MatMul(ln1, t)
		} else {
			qkv = gpt2.MatMul(ln1, t, d, blk.AttnW, d, 3*d)
		}
		gpt2.AddBias(qkv, t, 3*d, blk.AttnB)
		if li < len(m.Attn) {
			addVec(qkv, m.Attn[li].apply(ln1, t, scale))
		}
		att, _ := gpt2.CausalAttn(qkv, t, d, cfg.NHead)
		proj := gpt2.MatMul(att, t, d, blk.ProjW, d, d)
		gpt2.AddBias(proj, t, d, blk.ProjB)
		x2 := append([]float32(nil), x...)
		addVec(x2, proj)
		ln2, _, _ := gpt2.LayerNorm(x2, t, d, blk.LN2.W, blk.LN2.B, eps)
		var fc []float32
		if m.useQ {
			fc = m.qFC[li].MatMul(ln2, t)
		} else {
			fc = gpt2.MatMul(ln2, t, d, blk.FcW, d, cfg.Inner())
		}
		gpt2.AddBias(fc, t, cfg.Inner(), blk.FcB)
		if li < len(m.FC) {
			addVec(fc, m.FC[li].apply(ln2, t, scale))
		}
		act := make([]float32, len(fc))
		for i, v := range fc {
			act[i] = gpt2.GELUNew(v)
		}
		mlp := gpt2.MatMul(act, t, cfg.Inner(), blk.FcProjW, cfg.Inner(), d)
		gpt2.AddBias(mlp, t, d, blk.FcProjB)
		x = x2
		addVec(x, mlp)
	}

	lnf, _, _ := gpt2.LayerNorm(x, t, d, base.LNF.W, base.LNF.B, eps)
	wteT := gpt2.Transpose(base.WTE, cfg.VocabSize, d)
	return gpt2.MatMul(lnf, t, d, wteT, d, cfg.VocabSize), nil
}

// Generate greedily extends prompt by maxNew tokens.
func (m *Model) Generate(prompt []int, maxNew int) []int {
	if maxNew <= 0 {
		maxNew = 16
	}
	out := append([]int(nil), prompt...)
	cfg := m.Base.Cfg
	for n := 0; n < maxNew; n++ {
		ctx := out
		if len(ctx) > cfg.NPositions {
			ctx = ctx[len(ctx)-cfg.NPositions:]
		}
		logits := m.ForwardLogits(ctx)
		t := len(ctx)
		row := logits[(t-1)*cfg.VocabSize : t*cfg.VocabSize]
		out = append(out, argmax(row))
	}
	return out
}

// SampleOptions controls stochastic decoding. Temperature <= 0 is greedy
// (same as Generate). TopK <= 0 disables top-k. TopP <= 0 or >= 1 disables
// nucleus sampling. EOS < 0 disables early stop; otherwise generation stops
// after the EOS token is produced (EOS is included in the returned sequence).
type SampleOptions struct {
	Temperature float64
	TopK        int
	TopP        float64
	EOS         int
	Seed        int64
}

// Sample extends prompt by up to maxNew tokens using SampleOptions.
func (m *Model) Sample(prompt []int, maxNew int, o SampleOptions) []int {
	if maxNew <= 0 {
		maxNew = 16
	}
	out := append([]int(nil), prompt...)
	cfg := m.Base.Cfg
	rng := rand.New(rand.NewPCG(uint64(o.Seed), uint64(o.Seed)^0x9e3779b97f4a7c15)) //nolint:gosec // G404: deterministic sampling, not crypto
	for n := 0; n < maxNew; n++ {
		ctx := out
		if len(ctx) > cfg.NPositions {
			ctx = ctx[len(ctx)-cfg.NPositions:]
		}
		logits := m.ForwardLogits(ctx)
		t := len(ctx)
		row := logits[(t-1)*cfg.VocabSize : t*cfg.VocabSize]
		tok := sampleToken(row, o, rng)
		out = append(out, tok)
		if o.EOS >= 0 && tok == o.EOS {
			break
		}
	}
	return out
}

func argmax(row []float32) int {
	best := 0
	bestV := row[0]
	for i, v := range row[1:] {
		if v > bestV {
			bestV = v
			best = i + 1
		}
	}
	return best
}

func sampleToken(row []float32, o SampleOptions, rng *rand.Rand) int {
	if o.Temperature <= 0 {
		return argmax(row)
	}
	logits := append([]float32(nil), row...)
	invT := float32(1 / o.Temperature)
	for i := range logits {
		logits[i] *= invT
	}
	applyTopK(logits, o.TopK)
	probs := softmax1d(logits)
	applyTopP(probs, o.TopP)
	u := rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += float64(p)
		if u <= cum {
			return i
		}
	}
	return len(probs) - 1
}

func applyTopK(logits []float32, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return logits[idx[i]] > logits[idx[j]] })
	for _, i := range idx[k:] {
		logits[i] = float32(math.Inf(-1))
	}
}

func softmax1d(logits []float32) []float32 {
	out := make([]float32, len(logits))
	m := logits[0]
	for _, v := range logits[1:] {
		if v > m {
			m = v
		}
	}
	var sum float32
	for i, v := range logits {
		if math.IsInf(float64(v), -1) {
			out[i] = 0
			continue
		}
		e := float32(math.Exp(float64(v - m)))
		out[i] = e
		sum += e
	}
	if sum == 0 {
		return out
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

func applyTopP(probs []float32, p float64) {
	if p <= 0 || p >= 1 {
		return
	}
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return probs[idx[i]] > probs[idx[j]] })
	cutoff := 1
	var cum float64
	for i, id := range idx {
		cum += float64(probs[id])
		cutoff = i + 1
		if cum >= p {
			break
		}
	}
	var sum float32
	keep := cutoff
	for i, id := range idx {
		if i >= keep {
			probs[id] = 0
			continue
		}
		sum += probs[id]
	}
	if sum <= 0 {
		return
	}
	inv := 1 / sum
	for i := 0; i < keep; i++ {
		probs[idx[i]] *= inv
	}
}
