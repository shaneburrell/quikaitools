package peft_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
	"github.com/shaneburrell/quikaitools/internal/safetensors"
)

func tinyCfg() gpt2.Config {
	return gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 32, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
}

func trainedLoRA(t *testing.T, seed int64) *peft.Model {
	t.Helper()
	base := gpt2.NewRandom(tinyCfg(), seed)
	m := peft.Wrap(base, peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	for i := range m.Attn[0].B {
		m.Attn[0].B[i] = 0.03
	}
	for i := range m.FC[1].B {
		m.FC[1].B[i] = -0.02
	}
	return m
}

func logitsClose(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len %d vs %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("logit[%d]: got=%.6f want=%.6f", i, got[i], want[i])
		}
	}
}

func TestPEFTSaveLoadRoundTripNoAdapterJSON(t *testing.T) {
	m := trainedLoRA(t, 11)
	toks := []int{1, 2, 3, 4}
	want := append([]float32(nil), m.ForwardLogits(toks)...)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
	base2 := gpt2.NewRandom(tinyCfg(), 11)
	loaded, err := peft.Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	logitsClose(t, loaded.ForwardLogits(toks), want)
}

func TestPEFTSafetensorsKeysAndShapes(t *testing.T) {
	m := trainedLoRA(t, 5)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if (8+n)%8 != 0 {
		t.Fatalf("data start %d not aligned", 8+n)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	var meta map[string]string
	if err := json.Unmarshal(hdr["__metadata__"], &meta); err != nil {
		t.Fatal(err)
	}
	if meta["format"] != "pt" {
		t.Fatalf("metadata=%v", meta)
	}
	type entry struct {
		Dtype string `json:"dtype"`
		Shape []int  `json:"shape"`
	}
	want := map[string][]int{
		"base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": {2, 16},
		"base_model.model.transformer.h.0.attn.c_attn.lora_B.weight": {48, 2},
		"base_model.model.transformer.h.0.mlp.c_fc.lora_A.weight":    {2, 16},
		"base_model.model.transformer.h.0.mlp.c_fc.lora_B.weight":    {32, 2},
		"base_model.model.transformer.h.1.attn.c_attn.lora_A.weight": {2, 16},
		"base_model.model.transformer.h.1.attn.c_attn.lora_B.weight": {48, 2},
		"base_model.model.transformer.h.1.mlp.c_fc.lora_A.weight":    {2, 16},
		"base_model.model.transformer.h.1.mlp.c_fc.lora_B.weight":    {32, 2},
	}
	if len(hdr)-1 != len(want) { // minus __metadata__
		t.Fatalf("header keys %d want %d", len(hdr)-1, len(want))
	}
	for key, shape := range want {
		rawE, ok := hdr[key]
		if !ok {
			t.Fatalf("missing key %s", key)
		}
		var e entry
		if err := json.Unmarshal(rawE, &e); err != nil {
			t.Fatal(err)
		}
		if e.Dtype != "F32" || len(e.Shape) != 2 || e.Shape[0] != shape[0] || e.Shape[1] != shape[1] {
			t.Fatalf("%s: dtype=%s shape=%v want F32 %v", key, e.Dtype, e.Shape, shape)
		}
	}
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ac map[string]any
	if err := json.Unmarshal(cfgRaw, &ac); err != nil {
		t.Fatal(err)
	}
	if ac["peft_type"] != "LORA" || ac["task_type"] != "CAUSAL_LM" || ac["bias"] != "none" {
		t.Fatalf("adapter_config=%v", ac)
	}
	if ac["fan_in_fan_out"] != true || ac["inference_mode"] != false {
		t.Fatalf("adapter_config flags=%v", ac)
	}
}

func TestLegacyAdapterJSONOnlyStillLoads(t *testing.T) {
	m := trainedLoRA(t, 9)
	toks := []int{2, 3, 4}
	want := append([]float32(nil), m.ForwardLogits(toks)...)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"adapter_config.json", "adapter_model.safetensors"} {
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
	}
	base2 := gpt2.NewRandom(tinyCfg(), 9)
	loaded, err := peft.Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	logitsClose(t, loaded.ForwardLogits(toks), want)
}

func TestAdapterConfigRankMismatch(t *testing.T) {
	m := trainedLoRA(t, 3)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ac map[string]any
	if err := json.Unmarshal(raw, &ac); err != nil {
		t.Fatal(err)
	}
	ac["r"] = 8
	body, err := json.MarshalIndent(ac, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	base2 := gpt2.NewRandom(tinyCfg(), 3)
	if _, err := peft.Load(base2, dir); err == nil {
		t.Fatal("expected r mismatch error")
	}
}

func TestLoadAndMergePEFTFormat(t *testing.T) {
	m := trainedLoRA(t, 7)
	toks := []int{1, 2, 3, 5}
	want := append([]float32(nil), m.ForwardLogits(toks)...)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
	mergedBase := gpt2.NewRandom(tinyCfg(), 7)
	if err := peft.LoadAndMerge(mergedBase, dir); err != nil {
		t.Fatal(err)
	}
	got := peft.BaseOnly(mergedBase).ForwardLogits(toks)
	logitsClose(t, got, want)
}

func TestPEFTSubsetLayersStayZeroB(t *testing.T) {
	base := gpt2.NewRandom(tinyCfg(), 4)
	full := peft.Wrap(base, peft.Config{Rank: 2, Alpha: 4})
	for i := range full.Attn[0].B {
		full.Attn[0].B[i] = 0.04
	}
	dir := t.TempDir()
	if err := full.Save(dir); err != nil {
		t.Fatal(err)
	}
	all, err := safetensors.LoadFile(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	keep := "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight"
	keepB := "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight"
	subset := map[string]safetensors.Tensor{keep: all[keep], keepB: all[keepB]}
	if err := safetensors.WriteFileMeta(filepath.Join(dir, "adapter_model.safetensors"), subset, map[string]string{"format": "pt"}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
	base2 := gpt2.NewRandom(tinyCfg(), 4)
	loaded, err := peft.Load(base2, dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range loaded.Attn[1].B {
		if v != 0 {
			t.Fatal("missing layer B should stay zero")
		}
	}
	for _, v := range loaded.FC[0].B {
		if v != 0 {
			t.Fatal("missing module B should stay zero")
		}
	}
	if loaded.Attn[0].B[0] == 0 {
		t.Fatal("present adapter B should be restored")
	}
}

func TestLoadPEFTZeroKeysErrors(t *testing.T) {
	m := trainedLoRA(t, 4)
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	all, err := safetensors.LoadFile(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	var one safetensors.Tensor
	for _, tns := range all {
		one = tns
		break
	}
	renamed := map[string]safetensors.Tensor{
		"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": one,
	}
	if err := safetensors.WriteFileMeta(filepath.Join(dir, "adapter_model.safetensors"), renamed, map[string]string{"format": "pt"}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
	base := gpt2.NewRandom(tinyCfg(), 4)
	if _, err := peft.Load(base, dir); err == nil {
		t.Fatal("expected error when no GPT-2 LoRA keys match")
	}
}
