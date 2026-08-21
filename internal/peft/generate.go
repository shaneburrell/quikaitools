package peft

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

// Load restores adapters from adapter.json onto base.
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
	return m, nil
}

// BaseOnly wraps a frozen GPT-2 with no LoRA adapters (identity generate path).
func BaseOnly(base *gpt2.Model) *Model {
	return &Model{Base: base, Cfg: Config{Rank: 0}}
}

// ForwardLogits runs the model once and returns logits [T, V].
func (m *Model) ForwardLogits(tokens []int) []float32 {
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
	return gpt2.MatMul(lnf, t, d, wteT, d, cfg.VocabSize)
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
		best := 0
		bestV := row[0]
		for i, v := range row[1:] {
			if v > bestV {
				bestV = v
				best = i + 1
			}
		}
		out = append(out, best)
	}
	return out
}
