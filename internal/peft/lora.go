// Package peft is LoRA on frozen GPT-2 Conv1D weights (c_attn and c_fc).
package peft

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

// Config is a PEFT-style LoRA setup.
type Config struct {
	Rank  int     `json:"rank"`
	Alpha float32 `json:"alpha"`
	LR    float64 `json:"lr"`
}

func (c Config) Scale() float32 {
	if c.Rank <= 0 {
		return 0
	}
	return c.Alpha / float32(c.Rank)
}

// Adapter holds A [in,r] and B [r,out] for one linear.
type Adapter struct {
	Name           string
	In, Out, Rank  int
	A, B           []float32
	dA, dB         []float32
	mA, vA, mB, vB []float32
	Step           int
}

// Model is a GPT-2 plus LoRA on every layer's c_attn and c_fc.
type Model struct {
	Base  *gpt2.Model
	Cfg   Config
	Attn  []*Adapter
	FC    []*Adapter
	useQ  bool
	qAttn []QuantLinear
	qFC   []QuantLinear
}

// Wrap attaches zero-B LoRA (A small random) so the first step is identity.
func Wrap(base *gpt2.Model, cfg Config) *Model {
	return WrapSeeded(base, cfg, 0)
}

// WrapSeeded is Wrap with a seed mixed into LoRA A initialization.
// Seed 0 uses the historical name-hash init (identical to Wrap).
func WrapSeeded(base *gpt2.Model, cfg Config, seed int64) *Model {
	if cfg.Rank <= 0 {
		cfg.Rank = 4
	}
	if cfg.Alpha == 0 {
		cfg.Alpha = float32(2 * cfg.Rank)
	}
	if cfg.LR == 0 {
		cfg.LR = 3e-3
	}
	m := &Model{Base: base, Cfg: cfg}
	d, inn := base.Cfg.NEmbd, base.Cfg.Inner()
	for i := range base.Blocks {
		m.Attn = append(m.Attn, newAdapterSeeded(fmt.Sprintf("h.%d.attn.c_attn", i), d, 3*d, cfg.Rank, seed))
		m.FC = append(m.FC, newAdapterSeeded(fmt.Sprintf("h.%d.mlp.c_fc", i), d, inn, cfg.Rank, seed))
	}
	return m
}

func newAdapter(name string, in, out, r int) *Adapter {
	return newAdapterSeeded(name, in, out, r, 0)
}

func newAdapterSeeded(name string, in, out, r int, seed int64) *Adapter {
	a := &Adapter{
		Name: name, In: in, Out: out, Rank: r,
		A: make([]float32, in*r), B: make([]float32, r*out),
		dA: make([]float32, in*r), dB: make([]float32, r*out),
		mA: make([]float32, in*r), vA: make([]float32, in*r),
		mB: make([]float32, r*out), vB: make([]float32, r*out),
	}
	// Kaiming-ish A, zeros B (standard LoRA init). Seed 0 is a no-op XOR.
	scale := float32(1 / math.Sqrt(float64(in)))
	s := uint64(len(name)*997 + in + out)
	s ^= uint64(seed)
	for i := range a.A {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		a.A[i] = (float32(s&0xffff)/65535 - 0.5) * 2 * scale
	}
	return a
}

func (a *Adapter) apply(x []float32, rows int, scale float32) []float32 {
	// y = (x @ A) @ B * scale
	xa := matmul(x, rows, a.In, a.A, a.In, a.Rank)
	yb := matmul(xa, rows, a.Rank, a.B, a.Rank, a.Out)
	for i := range yb {
		yb[i] *= scale
	}
	return yb
}

func (a *Adapter) backward(x, dy []float32, rows int, scale float32) (dx []float32) {
	// y = scale * (x @ A @ B)
	// xa = x @ A
	xa := matmul(x, rows, a.In, a.A, a.In, a.Rank)
	// dB += xa^T @ (dy * scale)
	dScaled := append([]float32(nil), dy...)
	for i := range dScaled {
		dScaled[i] *= scale
	}
	xAT := transpose(xa, rows, a.Rank)
	dB := matmul(xAT, a.Rank, rows, dScaled, rows, a.Out)
	add(a.dB, dB)
	// dA += x^T @ ((dy*scale) @ B^T)
	BT := transpose(a.B, a.Rank, a.Out)
	dXA := matmul(dScaled, rows, a.Out, BT, a.Out, a.Rank)
	xT := transpose(x, rows, a.In)
	dA := matmul(xT, a.In, rows, dXA, rows, a.Rank)
	add(a.dA, dA)
	// dx += (dy*scale) @ B^T @ A^T
	AT := transpose(a.A, a.In, a.Rank)
	dx = matmul(dXA, rows, a.Rank, AT, a.Rank, a.In)
	return dx
}

func (a *Adapter) zeroGrad() {
	clear(a.dA)
	clear(a.dB)
}

// Adam updates A and B.
func (a *Adapter) adam(lr, beta1, beta2, eps float64) {
	a.Step++
	t := float64(a.Step)
	bc1 := 1 - math.Pow(beta1, t)
	bc2 := 1 - math.Pow(beta2, t)
	update := func(w, g, m, v []float32) {
		for i := range w {
			m[i] = float32(beta1)*m[i] + float32(1-beta1)*g[i]
			v[i] = float32(beta2)*v[i] + float32(1-beta2)*g[i]*g[i]
			mh := m[i] / float32(bc1)
			vh := v[i] / float32(bc2)
			w[i] -= float32(lr) * mh / (float32(math.Sqrt(float64(vh))) + float32(eps))
		}
	}
	update(a.A, a.dA, a.mA, a.vA)
	update(a.B, a.dB, a.mB, a.vB)
}

// TrainableParams counts LoRA floats (not the frozen base).
func (m *Model) TrainableParams() int {
	n := 0
	for _, a := range m.Attn {
		n += len(a.A) + len(a.B)
	}
	for _, a := range m.FC {
		n += len(a.A) + len(a.B)
	}
	return n
}

func (m *Model) adapters() []*Adapter {
	return append(append([]*Adapter{}, m.Attn...), m.FC...)
}

func (m *Model) ZeroGrad() {
	for _, a := range m.adapters() {
		a.zeroGrad()
	}
}

// ScaleGrads multiplies accumulated LoRA grads by factor (use 1/accum).
func (m *Model) ScaleGrads(factor float32) {
	for _, a := range m.adapters() {
		for i := range a.dA {
			a.dA[i] *= factor
		}
		for i := range a.dB {
			a.dB[i] *= factor
		}
	}
}

func (m *Model) Step() {
	for _, a := range m.adapters() {
		a.adam(m.Cfg.LR, 0.9, 0.999, 1e-8)
	}
}

// Save writes adapter.json (tiny, human-readable).
func (m *Model) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	type dump struct {
		Name string    `json:"name"`
		In   int       `json:"in"`
		Out  int       `json:"out"`
		Rank int       `json:"rank"`
		A    []float32 `json:"a"`
		B    []float32 `json:"b"`
	}
	var list []dump
	for _, a := range m.adapters() {
		list = append(list, dump{a.Name, a.In, a.Out, a.Rank, a.A, a.B})
	}
	body, err := json.MarshalIndent(struct {
		LoRA     Config `json:"lora"`
		QLoRA    bool   `json:"qlora"`
		Adapters []dump `json:"adapters"`
	}{m.Cfg, m.useQ, list}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "adapter.json"), body, 0o644); err != nil {
		return err
	}
	return m.saveOptimizer(dir)
}

type optimizerDump struct {
	Name string    `json:"name"`
	MA   []float32 `json:"m_a"`
	VA   []float32 `json:"v_a"`
	MB   []float32 `json:"m_b"`
	VB   []float32 `json:"v_b"`
	Step int       `json:"step"`
}

func (m *Model) saveOptimizer(dir string) error {
	var list []optimizerDump
	for _, a := range m.adapters() {
		list = append(list, optimizerDump{a.Name, a.mA, a.vA, a.mB, a.vB, a.Step})
	}
	body, err := json.MarshalIndent(struct {
		Adapters []optimizerDump `json:"adapters"`
	}{list}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "optimizer.json"), body, 0o644)
}

func (m *Model) restoreOptimizer(dir string) error {
	raw, err := os.ReadFile(filepath.Join(dir, "optimizer.json"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	var file struct {
		Adapters []optimizerDump `json:"adapters"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return err
	}
	byName := map[string]*Adapter{}
	for _, a := range m.adapters() {
		byName[a.Name] = a
	}
	for _, d := range file.Adapters {
		a, ok := byName[d.Name]
		if !ok {
			continue
		}
		if len(d.MA) == len(a.mA) {
			copy(a.mA, d.MA)
		}
		if len(d.VA) == len(a.vA) {
			copy(a.vA, d.VA)
		}
		if len(d.MB) == len(a.mB) {
			copy(a.mB, d.MB)
		}
		if len(d.VB) == len(a.vB) {
			copy(a.vB, d.VB)
		}
		a.Step = d.Step
	}
	return nil
}

func matmul(a []float32, aR, aC int, b []float32, bR, bC int) []float32 {
	if aC != bR {
		panic("peft: matmul shape")
	}
	out := make([]float32, aR*bC)
	for i := 0; i < aR; i++ {
		for k := 0; k < aC; k++ {
			av := a[i*aC+k]
			if av == 0 {
				continue
			}
			for j := 0; j < bC; j++ {
				out[i*bC+j] += av * b[k*bC+j]
			}
		}
	}
	return out
}

func transpose(a []float32, r, c int) []float32 {
	out := make([]float32, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}

func add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func clear(s []float32) {
	for i := range s {
		s[i] = 0
	}
}
