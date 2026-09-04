package gpt2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shaneburrell/quikaitools/internal/safetensors"
)

// Config is the Hugging Face GPT-2 JSON subset we need.
type Config struct {
	NEmbd        int     `json:"n_embd"`
	NHead        int     `json:"n_head"`
	NLayer       int     `json:"n_layer"`
	NPositions   int     `json:"n_positions"`
	VocabSize    int     `json:"vocab_size"`
	LayerNormEps float32 `json:"layer_norm_epsilon"`
	NInner       int     `json:"n_inner"`
}

func (c *Config) HeadDim() int { return c.NEmbd / c.NHead }

func (c *Config) Inner() int {
	if c.NInner > 0 {
		return c.NInner
	}
	return 4 * c.NEmbd
}

// Model is a frozen GPT-2 checkpoint plus optional LoRA (see peft).
type Model struct {
	Cfg    Config
	WTE    []float32 // [V, D]
	WPE    []float32 // [Tmax, D]
	LNF    struct{ W, B []float32 }
	Blocks []Block
}

// Block is one transformer layer. Weights are Conv1D [in, out].
type Block struct {
	LN1, LN2         struct{ W, B []float32 }
	AttnW, AttnB     []float32 // [D, 3D], [3D]
	ProjW, ProjB     []float32 // [D, D], [D]
	FcW, FcB         []float32 // [D, I], [I]
	FcProjW, FcProjB []float32 // [I, D], [D]
}

// LoadDir reads config.json + model.safetensors from a pulled Hub folder.
func LoadDir(dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-5
	}
	if cfg.NEmbd == 0 || cfg.NHead == 0 || cfg.NLayer == 0 {
		return nil, fmt.Errorf("gpt2: incomplete config")
	}
	if cfg.NEmbd%cfg.NHead != 0 {
		return nil, fmt.Errorf("gpt2: n_embd not divisible by n_head")
	}
	tensors, err := safetensors.LoadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	m := &Model{Cfg: cfg}
	var loadErr error
	tensor := func(name string) []float32 {
		if loadErr != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			loadErr = fmt.Errorf("gpt2: missing tensor %q in %s", name, dir)
			return nil
		}
		return t.Data
	}
	m.WTE = tensor("transformer.wte.weight")
	m.WPE = tensor("transformer.wpe.weight")
	m.LNF.W = tensor("transformer.ln_f.weight")
	m.LNF.B = tensor("transformer.ln_f.bias")
	if fc := tensors["transformer.h.0.mlp.c_fc.weight"]; len(fc.Shape) == 2 {
		m.Cfg.NInner = fc.Shape[1]
	}
	m.Blocks = make([]Block, cfg.NLayer)
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("transformer.h.%d.", i)
		b := &m.Blocks[i]
		b.LN1.W = tensor(p + "ln_1.weight")
		b.LN1.B = tensor(p + "ln_1.bias")
		b.LN2.W = tensor(p + "ln_2.weight")
		b.LN2.B = tensor(p + "ln_2.bias")
		b.AttnW = tensor(p + "attn.c_attn.weight")
		b.AttnB = tensor(p + "attn.c_attn.bias")
		b.ProjW = tensor(p + "attn.c_proj.weight")
		b.ProjB = tensor(p + "attn.c_proj.bias")
		b.FcW = tensor(p + "mlp.c_fc.weight")
		b.FcB = tensor(p + "mlp.c_fc.bias")
		b.FcProjW = tensor(p + "mlp.c_proj.weight")
		b.FcProjB = tensor(p + "mlp.c_proj.bias")
	}
	if loadErr != nil {
		return nil, loadErr
	}
	return m, nil
}

// NewRandom is a tiny randomly-initialized model for tests (no Hub).
func NewRandom(cfg Config, seed int64) *Model {
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-5
	}
	if cfg.NInner == 0 {
		cfg.NInner = 4 * cfg.NEmbd
	}
	rng := newRNG(seed)
	m := &Model{Cfg: cfg}
	m.WTE = rng.randn(cfg.VocabSize * cfg.NEmbd)
	m.WPE = rng.randn(cfg.NPositions * cfg.NEmbd)
	m.LNF.W = ones(cfg.NEmbd)
	m.LNF.B = zeros(cfg.NEmbd)
	m.Blocks = make([]Block, cfg.NLayer)
	d, inn := cfg.NEmbd, cfg.NInner
	for i := range m.Blocks {
		b := &m.Blocks[i]
		b.LN1.W, b.LN1.B = ones(d), zeros(d)
		b.LN2.W, b.LN2.B = ones(d), zeros(d)
		b.AttnW, b.AttnB = rng.randn(d*3*d), zeros(3*d)
		b.ProjW, b.ProjB = rng.randn(d*d), zeros(d)
		b.FcW, b.FcB = rng.randn(d*inn), zeros(inn)
		b.FcProjW, b.FcProjB = rng.randn(inn*d), zeros(d)
	}
	return m
}

func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
func zeros(n int) []float32 { return make([]float32, n) }

type rng struct{ s uint64 }

func newRNG(seed int64) *rng { return &rng{s: uint64(seed | 1)} }

func (r *rng) next() float32 {
	r.s ^= r.s << 13
	r.s ^= r.s >> 7
	r.s ^= r.s << 17
	u := (r.s >> 8) & 0xffffff
	return (float32(u)/float32(1<<24))*0.04 - 0.02
}

func (r *rng) randn(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.next()
	}
	return out
}
