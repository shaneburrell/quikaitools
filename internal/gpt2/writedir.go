package gpt2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/safetensors"
)

// WriteDir writes config.json, model.safetensors, vocab.json, merges.txt for tests.
func WriteDir(dir string, m *Model) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg := m.Cfg
	if cfg.NInner == 0 {
		cfg.NInner = m.Cfg.Inner()
	}
	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		return err
	}
	tensors := map[string]safetensors.Tensor{}
	put := func(name string, shape []int, data []float32) {
		tensors[name] = safetensors.Tensor{Name: name, Dtype: "F32", Shape: shape, Data: data}
	}
	d, v, tmax := cfg.NEmbd, cfg.VocabSize, cfg.NPositions
	inn := cfg.Inner()
	put("transformer.wte.weight", []int{v, d}, m.WTE)
	put("transformer.wpe.weight", []int{tmax, d}, m.WPE)
	put("transformer.ln_f.weight", []int{d}, m.LNF.W)
	put("transformer.ln_f.bias", []int{d}, m.LNF.B)
	for i, blk := range m.Blocks {
		p := fmt.Sprintf("transformer.h.%d.", i)
		put(p+"ln_1.weight", []int{d}, blk.LN1.W)
		put(p+"ln_1.bias", []int{d}, blk.LN1.B)
		put(p+"ln_2.weight", []int{d}, blk.LN2.W)
		put(p+"ln_2.bias", []int{d}, blk.LN2.B)
		put(p+"attn.c_attn.weight", []int{d, 3 * d}, blk.AttnW)
		put(p+"attn.c_attn.bias", []int{3 * d}, blk.AttnB)
		put(p+"attn.c_proj.weight", []int{d, d}, blk.ProjW)
		put(p+"attn.c_proj.bias", []int{d}, blk.ProjB)
		put(p+"mlp.c_fc.weight", []int{d, inn}, blk.FcW)
		put(p+"mlp.c_fc.bias", []int{inn}, blk.FcB)
		put(p+"mlp.c_proj.weight", []int{inn, d}, blk.FcProjW)
		put(p+"mlp.c_proj.bias", []int{d}, blk.FcProjB)
	}
	if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		return err
	}
	vocab := map[string]int{"<|endoftext|>": 0}
	for i := 1; i < cfg.VocabSize; i++ {
		// GPT-2 byte-level: use printable letters for small vocabs
		ch := string(rune('a' + (i-1)%26))
		if i > 26 {
			ch = fmt.Sprintf("t%d", i)
		}
		vocab[ch] = i
	}
	vb, err := json.Marshal(vocab)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "vocab.json"), vb, 0o644); err != nil {
		return err
	}
	// empty merges is fine for letter tokens
	return os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("#version: 0.2\n"+strings.Repeat("", 0)), 0o644)
}
