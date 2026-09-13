package peft

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/safetensors"
)

// Load restores adapters onto base.
// When adapter_model.safetensors and adapter_config.json are both present they
// are preferred (PEFT tensors are transposed back to Conv1D [in,r] / [r,out]);
// otherwise adapter.json is used. optimizer.json is restored when present.
func Load(base *gpt2.Model, dir string) (*Model, error) {
	st := filepath.Join(dir, "adapter_model.safetensors")
	cfgPath := filepath.Join(dir, "adapter_config.json")
	_, errST := os.Stat(st)
	_, errCfg := os.Stat(cfgPath)
	if errST == nil && errCfg == nil {
		return loadHFPEFT(base, dir)
	}
	return loadAdapterJSON(base, dir)
}

func loadAdapterJSON(base *gpt2.Model, dir string) (*Model, error) {
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

type adapterJSONSidecar struct {
	LoRA  Config `json:"lora"`
	QLoRA bool   `json:"qlora"`
}

func loadHFPEFT(base *gpt2.Model, dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return nil, err
	}
	var ac hfAdapterConfig
	if err := json.Unmarshal(raw, &ac); err != nil {
		return nil, err
	}
	if ac.PeftType != "" && ac.PeftType != "LORA" {
		return nil, fmt.Errorf("peft: unsupported peft_type %q", ac.PeftType)
	}
	if ac.R <= 0 {
		return nil, fmt.Errorf("peft: adapter_config r must be positive")
	}
	cfg := Config{Rank: ac.R, Alpha: ac.LoraAlpha}
	var qlora bool
	if side, err := os.ReadFile(filepath.Join(dir, "adapter.json")); err == nil {
		var file adapterJSONSidecar
		if err := json.Unmarshal(side, &file); err != nil {
			return nil, err
		}
		if file.LoRA.Rank != 0 && file.LoRA.Rank != ac.R {
			return nil, fmt.Errorf("peft: adapter_config r=%d does not match Config rank=%d", ac.R, file.LoRA.Rank)
		}
		if file.LoRA.Alpha != 0 && file.LoRA.Alpha != ac.LoraAlpha {
			return nil, fmt.Errorf("peft: adapter_config lora_alpha=%v does not match Config alpha=%v", ac.LoraAlpha, file.LoRA.Alpha)
		}
		cfg.LR = file.LoRA.LR
		qlora = file.QLoRA
	}
	m := Wrap(base, cfg)
	tensors, err := safetensors.LoadFile(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return nil, err
	}
	applied := 0
	for name, t := range tensors {
		layer, module, which, ok := parsePEFTKey(name)
		if !ok {
			continue
		}
		var a *Adapter
		switch module {
		case "attn.c_attn":
			if layer < 0 || layer >= len(m.Attn) {
				return nil, fmt.Errorf("peft: layer %d out of range for %s", layer, name)
			}
			a = m.Attn[layer]
		case "mlp.c_fc":
			if layer < 0 || layer >= len(m.FC) {
				return nil, fmt.Errorf("peft: layer %d out of range for %s", layer, name)
			}
			a = m.FC[layer]
		default:
			continue
		}
		if err := copyPEFTMatrix(a, which, t); err != nil {
			return nil, fmt.Errorf("peft: %s: %w", name, err)
		}
		applied++
	}
	if applied == 0 {
		return nil, fmt.Errorf("peft: no GPT-2 LoRA tensors in %s", dir)
	}
	if qlora {
		m.EnableQLoRA()
	}
	if err := m.restoreOptimizer(dir); err != nil {
		return nil, err
	}
	return m, nil
}

// copyPEFTMatrix transposes a PEFT nn.Linear tensor back into Adapter layout.
// lora_A is [r, in] → A [in, r]; lora_B is [out, r] → B [r, out].
func copyPEFTMatrix(a *Adapter, which string, t safetensors.Tensor) error {
	switch which {
	case "A":
		if len(t.Shape) != 2 || t.Shape[0] != a.Rank || t.Shape[1] != a.In {
			return fmt.Errorf("lora_A shape %v want [%d, %d]", t.Shape, a.Rank, a.In)
		}
		if len(t.Data) != a.Rank*a.In {
			return fmt.Errorf("lora_A data length %d want %d", len(t.Data), a.Rank*a.In)
		}
		copy(a.A, transpose(t.Data, a.Rank, a.In))
	case "B":
		if len(t.Shape) != 2 || t.Shape[0] != a.Out || t.Shape[1] != a.Rank {
			return fmt.Errorf("lora_B shape %v want [%d, %d]", t.Shape, a.Out, a.Rank)
		}
		if len(t.Data) != a.Out*a.Rank {
			return fmt.Errorf("lora_B data length %d want %d", len(t.Data), a.Out*a.Rank)
		}
		copy(a.B, transpose(t.Data, a.Out, a.Rank))
	default:
		return fmt.Errorf("unknown LoRA matrix %q", which)
	}
	return nil
}

// parsePEFTKey accepts base_model.model.transformer.h.{N}.{attn.c_attn|mlp.c_fc}.lora_{A|B}[.default].weight
func parsePEFTKey(name string) (layer int, module, which string, ok bool) {
	if !strings.HasPrefix(name, hfPEFTPrefix) || !strings.HasSuffix(name, ".weight") {
		return 0, "", "", false
	}
	mid := strings.TrimSuffix(strings.TrimPrefix(name, hfPEFTPrefix), ".weight")
	mid = strings.TrimSuffix(mid, ".default")
	const aSuf = ".lora_A"
	const bSuf = ".lora_B"
	switch {
	case strings.HasSuffix(mid, aSuf):
		which = "A"
		mid = strings.TrimSuffix(mid, aSuf)
	case strings.HasSuffix(mid, bSuf):
		which = "B"
		mid = strings.TrimSuffix(mid, bSuf)
	default:
		return 0, "", "", false
	}
	// mid is h.{N}.attn.c_attn or h.{N}.mlp.c_fc
	if !strings.HasPrefix(mid, "h.") {
		return 0, "", "", false
	}
	rest := strings.TrimPrefix(mid, "h.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return 0, "", "", false
	}
	n, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return 0, "", "", false
	}
	module = rest[dot+1:]
	return n, module, which, true
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
	if len(out) == 0 {
		out = []int{0}
	}
	cfg := m.Base.Cfg
	for n := 0; n < maxNew; n++ {
		ctx := clipContext(out, cfg.NPositions)
		logits := m.ForwardLogits(ctx)
		row := lastTokenLogits(logits, len(ctx), cfg.VocabSize)
		if len(row) == 0 {
			break
		}
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
	if len(out) == 0 {
		out = []int{0}
	}
	cfg := m.Base.Cfg
	rng := rand.New(rand.NewPCG(uint64(o.Seed), uint64(o.Seed)^0x9e3779b97f4a7c15)) //nolint:gosec // G404: deterministic sampling, not crypto
	for n := 0; n < maxNew; n++ {
		ctx := clipContext(out, cfg.NPositions)
		logits := m.ForwardLogits(ctx)
		row := lastTokenLogits(logits, len(ctx), cfg.VocabSize)
		if len(row) == 0 {
			break
		}
		tok := sampleToken(row, o, rng)
		out = append(out, tok)
		if o.EOS >= 0 && tok == o.EOS {
			break
		}
	}
	return out
}

// clipContext keeps the last npos tokens. npos <= 0 means no limit, matching ForwardLogits.
func clipContext(ctx []int, npos int) []int {
	if npos > 0 && len(ctx) > npos {
		return ctx[len(ctx)-npos:]
	}
	return ctx
}

func lastTokenLogits(logits []float32, t, vocab int) []float32 {
	if t <= 0 || vocab <= 0 {
		return nil
	}
	start := (t - 1) * vocab
	end := t * vocab
	if start < 0 || end > len(logits) {
		return nil
	}
	return logits[start:end]
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
