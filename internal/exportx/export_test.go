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
	if !strings.Contains(string(b), "FROM /tmp/x.gguf") {
		t.Fatalf("%s", b)
	}
	skip, err := exportx.ConvertGGUF(out, filepath.Join(t.TempDir(), "x.gguf"), "Q4_K_M")
	if err != nil {
		t.Fatal(err)
	}
	if skip == "" {
		t.Fatal("expected skip without convert tools")
	}
}
