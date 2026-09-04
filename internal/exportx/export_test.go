package exportx_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/exportx"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
	"github.com/shaneburrell/quikaitools/internal/testart"
)

func TestMergeAndModelfile(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 32, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
	base := gpt2.NewRandom(cfg, 3)
	modelDir := testart.Path(t, "exportx-model")
	if err := gpt2.WriteDir(modelDir, base); err != nil {
		t.Fatal(err)
	}
	base2, err := gpt2.LoadDir(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	m := peft.Wrap(base2, peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	for i := range m.Attn[0].B {
		m.Attn[0].B[i] = 0.02
	}
	adapterDir := testart.Path(t, "exportx-adapter")
	if err := m.Save(adapterDir); err != nil {
		t.Fatal(err)
	}
	out := testart.Path(t, "exportx-merged")
	if err := exportx.MergeGPT2LoRA(modelDir, adapterDir, out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(out, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	mf := filepath.Join(t.TempDir(), "Modelfile")
	if err := exportx.WriteModelfile("", "/tmp/x.gguf", mf); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(mf)
	text := string(b)
	if !strings.Contains(text, "FROM /tmp/x.gguf") {
		t.Fatalf("%s", b)
	}
	if !strings.Contains(text, "PARAMETER temperature 0.7") {
		t.Fatalf("default template missing temperature 0.7:\n%s", text)
	}
	if !strings.Contains(text, "<|im_start|>") || !strings.Contains(text, "<|im_end|>") {
		t.Fatalf("default template missing ChatML:\n%s", text)
	}
	skip, err := exportx.ConvertGGUF(out, filepath.Join(t.TempDir(), "x.gguf"), "Q4_K_M")
	if err != nil {
		t.Fatal(err)
	}
	if skip == "" {
		t.Fatal("expected skip without convert tools")
	}
}

func TestMergeGPT2LoRAMissingTokenizer(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 32, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
	base := gpt2.NewRandom(cfg, 3)
	modelDir := t.TempDir()
	if err := gpt2.WriteDir(modelDir, base); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(modelDir, "vocab.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(modelDir, "merges.txt")); err != nil {
		t.Fatal(err)
	}
	base2, err := gpt2.LoadDir(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	m := peft.Wrap(base2, peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	adapterDir := t.TempDir()
	if err := m.Save(adapterDir); err != nil {
		t.Fatal(err)
	}
	err = exportx.MergeGPT2LoRA(modelDir, adapterDir, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "vocab.json") || !strings.Contains(err.Error(), "merges.txt") {
		t.Fatalf("want missing tokenizer list, got %v", err)
	}
}

func TestWriteModelfileOptsForce(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "Modelfile")
	if err := os.WriteFile(out, []byte("OLD"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := exportx.WriteModelfileOpts("", "/tmp/y.gguf", out, false); err == nil {
		t.Fatal("expected refuse overwrite without force")
	}
	b, _ := os.ReadFile(out)
	if string(b) != "OLD" {
		t.Fatalf("clobbered without force: %q", b)
	}
	if err := exportx.WriteModelfile("", "/tmp/y.gguf", out); err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(out)
	if !strings.Contains(string(b), "FROM /tmp/y.gguf") {
		t.Fatalf("WriteModelfile force=true should overwrite: %s", b)
	}
	fresh := filepath.Join(dir, "new", "Modelfile")
	if err := exportx.WriteModelfileOpts("", "/tmp/z.gguf", fresh, false); err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(fresh)
	if !strings.Contains(string(b), "FROM /tmp/z.gguf") {
		t.Fatalf("%s", b)
	}
}
