package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/backend"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
)

func TestRunLoRAWithDirAccumResume(t *testing.T) {
	dir := t.TempDir()
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 3)
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	data := filepath.Join(dir, "data.txt")
	if err := os.WriteFile(data, []byte("aaaaaaaaaa bbbbbbbbbb cccccccccc"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "adapter")
	losses, err := RunLoRA(LoRAOptions{
		ModelDir: dir, DataPath: data, OutDir: out,
		Steps: 4, SeqLen: 8, Rank: 2, Alpha: 4, LR: 1e-2,
		Accum: 2, CkptEvery: 2, Profile: backend.KindCPU, EvalEvery: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) {
			t.Fatalf("nan %d", i)
		}
	}
	out2 := filepath.Join(dir, "adapter2")
	_, err = RunLoRA(LoRAOptions{
		ModelDir: dir, DataPath: data, OutDir: out2,
		Steps: 2, SeqLen: 8, Rank: 2, Resume: out, Profile: backend.KindCPU, QLoRA: true, LR: 1e-2,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = RunLoRA(LoRAOptions{
		ModelDir: dir, DataPath: data, Resume: out, Rank: 8, Profile: backend.KindCPU,
	})
	if err == nil {
		t.Fatal("expected rank mismatch error")
	}
	nomete := filepath.Join(dir, "no-meta")
	if err := os.MkdirAll(nomete, 0o755); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(out, "adapter.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nomete, "adapter.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = RunLoRA(LoRAOptions{
		ModelDir: dir, DataPath: data, Resume: nomete, Rank: 2, Profile: backend.KindCPU,
	})
	if err == nil {
		t.Fatal("expected missing meta.json error")
	}
}

func TestRunLoRAMaskPrompt(t *testing.T) {
	dir := t.TempDir()
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 3)
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	data := filepath.Join(dir, "chat.jsonl")
	body := `{"messages":[{"role":"system","content":"abc"},{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}` + "\n"
	body += `{"messages":[{"role":"user","content":"cat"},{"role":"assistant","content":"sat"}]}` + "\n"
	if err := os.WriteFile(data, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	losses, err := RunLoRA(LoRAOptions{
		ModelDir: dir, DataPath: data,
		Steps: 4, SeqLen: 8, Rank: 2, Alpha: 4, LR: 1e-2,
		MaskPrompt: true, Profile: backend.KindCPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) == 0 {
		t.Fatal("no losses")
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("step %d loss=%v", i, l)
		}
	}
}

func TestMaskPromptUsesJointEncode(t *testing.T) {
	dir := t.TempDir()
	base := gpt2.NewRandom(gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 32, VocabSize: 32, NInner: 16}, 3)
	if err := gpt2.WriteDir(dir, base); err != nil {
		t.Fatal(err)
	}
	tok, err := gpt2.LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt := "sys\n\nhello\n\n"
	completion := "world"
	joint := tok.Encode(prompt + completion)
	ids, mask := encodeMaskedSFT(tok, prompt, completion)
	if !equalInts(ids, joint) {
		t.Fatalf("masked ids=%v want joint %v", ids, joint)
	}
	if len(mask) != len(ids) {
		t.Fatalf("mask len %d ids %d", len(mask), len(ids))
	}
	masked := 0
	for _, m := range mask {
		if m {
			masked++
		}
	}
	if masked == 0 {
		t.Fatal("expected completion tokens to be masked")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
