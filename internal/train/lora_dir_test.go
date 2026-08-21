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
		Steps: 2, SeqLen: 8, Rank: 2, Resume: out, Profile: backend.KindCPU, QLoRA: true,
	})
	if err != nil {
		t.Fatal(err)
	}
}
