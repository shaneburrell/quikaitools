package train

import (
	"fmt"
	"os"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
)

// LoRAOptions is the SFT+LoRA job.
type LoRAOptions struct {
	ModelDir string
	DataPath string
	OutDir   string
	Steps    int
	SeqLen   int
	Rank     int
	Alpha    float32
	LR       float64
	Text     string // if set, skip DataPath
}

// RunLoRA loads a pulled GPT-2 folder (or uses Text + random if ModelDir empty
// only when testing via RunLoRARandom).
func RunLoRA(opt LoRAOptions) (losses []float32, err error) {
	if opt.Steps <= 0 {
		opt.Steps = 20
	}
	if opt.SeqLen <= 0 {
		opt.SeqLen = 32
	}
	base, err := gpt2.LoadDir(opt.ModelDir)
	if err != nil {
		return nil, err
	}
	tok, err := gpt2.LoadTokenizer(opt.ModelDir)
	if err != nil {
		return nil, err
	}
	text := opt.Text
	if text == "" {
		b, err := os.ReadFile(opt.DataPath)
		if err != nil {
			return nil, err
		}
		text = string(b)
	}
	ids := tok.Encode(text)
	if len(ids) < 3 {
		return nil, fmt.Errorf("train: need more tokens (got %d) — use longer --data", len(ids))
	}
	m := peft.Wrap(base, peft.Config{Rank: opt.Rank, Alpha: opt.Alpha, LR: opt.LR})
	seq := opt.SeqLen
	if seq >= len(ids) {
		seq = len(ids)
	}
	losses = make([]float32, 0, opt.Steps)
	for step := 0; step < opt.Steps; step++ {
		off := (step * 3) % (len(ids) - seq + 1)
		batch := ids[off : off+seq]
		loss := m.StepLoss(batch)
		losses = append(losses, loss)
	}
	if opt.OutDir != "" {
		if err := m.Save(opt.OutDir); err != nil {
			return losses, err
		}
	}
	return losses, nil
}

// RunLoRARandom trains LoRA on a synthetic tiny GPT-2 (CI, no Hub).
func RunLoRARandom(text string, steps int) ([]float32, *peft.Model, error) {
	cfg := gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 64, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
	base := gpt2.NewRandom(cfg, 1)
	ids := make([]int, 0, len(text))
	for _, r := range strings.ToLower(text) {
		if r >= 'a' && r <= 'z' {
			ids = append(ids, int(r-'a')%cfg.VocabSize)
		} else if r == ' ' {
			ids = append(ids, 26)
		}
	}
	for len(ids) < 8 {
		ids = append(ids, 1, 2, 3)
	}
	m := peft.Wrap(base, peft.Config{Rank: 2, Alpha: 4, LR: 1e-2})
	var losses []float32
	seq := 8
	if seq > len(ids) {
		seq = len(ids)
	}
	for step := 0; step < steps; step++ {
		losses = append(losses, m.StepLoss(ids[:seq]))
	}
	return losses, m, nil
}
