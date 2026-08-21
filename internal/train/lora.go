package train

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/backend"
	"github.com/shaneburrell/quikaitools/internal/dist"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
)

// LoRAOptions is the SFT+LoRA / QLoRA job.
type LoRAOptions struct {
	ModelDir  string
	DataPath  string
	OutDir    string
	Steps     int
	SeqLen    int
	Rank      int
	Alpha     float32
	LR        float64
	Text      string // if set, skip DataPath
	QLoRA     bool
	Accum     int
	Resume    string
	CkptEvery int
	Profile   backend.Kind
	EvalEvery int
}

// RunLoRA loads a pulled GPT-2 folder and trains LoRA (or QLoRA).
func RunLoRA(opt LoRAOptions) (losses []float32, err error) {
	if opt.Steps <= 0 {
		opt.Steps = 20
	}
	if opt.SeqLen <= 0 {
		opt.SeqLen = 32
	}
	if opt.Accum <= 0 {
		opt.Accum = 1
	}
	if opt.CkptEvery <= 0 {
		opt.CkptEvery = 10
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
		var err error
		text, err = loadTrainText(opt.DataPath)
		if err != nil {
			return nil, err
		}
	}
	ids := tok.Encode(text)
	if len(ids) < 3 {
		return nil, fmt.Errorf("train: need more tokens (got %d) — use longer --data", len(ids))
	}

	profileKind := opt.Profile
	if profileKind == "" {
		profileKind = backend.KindAuto
	}
	prof, err := backend.Detect(profileKind, nil)
	if err != nil {
		return nil, err
	}
	job := dist.NewJob(prof)
	job.AccumSteps = opt.Accum
	job.CheckpointEvery = opt.CkptEvery
	job.CheckpointDir = opt.OutDir
	job.ResumeFrom = opt.Resume
	if err := job.Validate(); err != nil {
		return nil, err
	}

	var m *peft.Model
	startStep := 0
	if opt.Resume != "" {
		m, err = peft.Load(base, opt.Resume)
		if err != nil {
			return nil, fmt.Errorf("resume: %w", err)
		}
		if meta, err := dist.LoadCheckpointMeta(opt.Resume); err == nil {
			startStep = meta.Step
		}
		if opt.QLoRA && !m.IsQLoRA() {
			m.EnableQLoRA()
		}
	} else {
		m = peft.Wrap(base, peft.Config{Rank: opt.Rank, Alpha: opt.Alpha, LR: opt.LR})
		if opt.QLoRA {
			m.EnableQLoRA()
		}
	}

	seq := opt.SeqLen
	if seq >= len(ids) {
		seq = len(ids)
	}
	losses = make([]float32, 0, opt.Steps)
	var accumLoss float32
	accumN := 0
	optSteps := 0
	recipe := "lora"
	if opt.QLoRA {
		recipe = "qlora"
	}
	syncAndMaybeCkpt := func(global int, loss float32) error {
		if accumN == 0 {
			return nil
		}
		m.ScaleGrads(1 / float32(accumN))
		m.Step()
		losses = append(losses, accumLoss/float32(accumN))
		optSteps++
		accumLoss = 0
		accumN = 0
		if opt.OutDir != "" && job.CheckpointEvery > 0 && optSteps%job.CheckpointEvery == 0 {
			ckpt := filepath.Join(opt.OutDir, fmt.Sprintf("step-%d", global))
			if err := m.Save(ckpt); err != nil {
				return err
			}
			_ = dist.SaveCheckpoint(ckpt, dist.CheckpointMeta{Step: global, Loss: loss, Recipe: recipe, Rank: opt.Rank})
		}
		return nil
	}
	for step := 0; step < opt.Steps; step++ {
		off := ((startStep + step) * 3) % (len(ids) - seq + 1)
		batch := ids[off : off+seq]
		zeroFirst := accumN == 0
		loss := m.AccumulateLoss(batch, zeroFirst)
		accumLoss += loss
		accumN++
		global := startStep + step + 1
		if accumN >= job.AccumSteps {
			if err := syncAndMaybeCkpt(global, loss); err != nil {
				return losses, err
			}
		}
		if opt.EvalEvery > 0 && (step+1)%opt.EvalEvery == 0 {
			_ = m.ForwardLogits(batch) // smoke eval forward
		}
	}
	if accumN > 0 {
		global := startStep + opt.Steps
		if err := syncAndMaybeCkpt(global, accumLoss/float32(accumN)); err != nil {
			return losses, err
		}
	}
	if opt.OutDir != "" {
		if err := m.Save(opt.OutDir); err != nil {
			return losses, err
		}
		last := float32(0)
		if len(losses) > 0 {
			last = losses[len(losses)-1]
		}
		_ = dist.SaveCheckpoint(opt.OutDir, dist.CheckpointMeta{
			Step: startStep + opt.Steps, Loss: last, Recipe: recipe, Rank: opt.Rank,
		})
	}
	return losses, nil
}

// RunLoRARandom trains LoRA on a synthetic tiny GPT-2 (CI, no Hub).
func RunLoRARandom(text string, steps int) ([]float32, *peft.Model, error) {
	return RunLoRARandomOpts(text, steps, false)
}

// RunLoRARandomOpts allows QLoRA in CI.
func RunLoRARandomOpts(text string, steps int, qlora bool) ([]float32, *peft.Model, error) {
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
	if qlora {
		m.EnableQLoRA()
	}
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
