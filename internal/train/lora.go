package train

import (
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/backend"
	"github.com/shaneburrell/quikaitools/internal/dist"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/peft"
)

// LoRAOptions is the SFT+LoRA / QLoRA job.
type LoRAOptions struct {
	ModelDir   string
	DataPath   string
	OutDir     string
	Steps      int
	SeqLen     int
	Rank       int
	Alpha      float32
	LR         float64
	Text       string // if set, skip DataPath
	QLoRA      bool
	Accum      int
	Resume     string
	CkptEvery  int
	Profile    backend.Kind
	EvalEvery  int
	Seed       int64
	MaskPrompt bool
}

type sftSample struct {
	ids  []int
	mask []bool
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

	useMask := opt.MaskPrompt && opt.Text == "" && strings.EqualFold(filepath.Ext(opt.DataPath), ".jsonl")
	var ids []int
	var samples []sftSample
	if useMask {
		pairs, err := MessagesToPromptCompletions(opt.DataPath)
		if err != nil {
			return nil, err
		}
		for _, p := range pairs {
			ids, mask := encodeMaskedSFT(tok, p.Prompt, p.Completion)
			if len(ids) < 2 {
				continue
			}
			samples = append(samples, sftSample{ids: ids, mask: mask})
		}
		if len(samples) == 0 {
			return nil, fmt.Errorf("train: need more tokens (got 0 masked samples) — use longer --data")
		}
	} else {
		text := opt.Text
		if text == "" {
			var err error
			text, err = loadTrainText(opt.DataPath)
			if err != nil {
				return nil, err
			}
		}
		ids = tok.Encode(text)
		if len(ids) < 3 {
			return nil, fmt.Errorf("train: need more tokens (got %d) — use longer --data", len(ids))
		}
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
		meta, err := dist.LoadCheckpointMeta(opt.Resume)
		if err != nil {
			return nil, fmt.Errorf("resume: %s has no meta.json: %w", opt.Resume, err)
		}
		startStep = meta.Step
		if opt.LR > 0 {
			m.Cfg.LR = opt.LR
		}
		if opt.Rank > 0 && opt.Rank != m.Cfg.Rank {
			return nil, fmt.Errorf("resume: cannot change rank from %d to %d", m.Cfg.Rank, opt.Rank)
		}
		if opt.QLoRA && !m.IsQLoRA() {
			m.EnableQLoRA()
		}
	} else {
		m = peft.WrapSeeded(base, peft.Config{Rank: opt.Rank, Alpha: opt.Alpha, LR: opt.LR}, opt.Seed)
		if opt.QLoRA {
			m.EnableQLoRA()
		}
	}

	seq := opt.SeqLen
	if !useMask && seq > len(ids) {
		seq = len(ids)
	}
	if npos := base.Cfg.NPositions; npos > 0 && seq > npos {
		seq = npos
	}
	losses = make([]float32, 0, opt.Steps)
	var accumLoss float32
	accumN := 0
	optSteps := 0
	recipe := "lora"
	if opt.QLoRA {
		recipe = "qlora"
	}
	syncAndMaybeCkpt := func(global int) error {
		if accumN == 0 {
			return nil
		}
		mean := accumLoss / float32(accumN)
		m.ScaleGrads(1 / float32(accumN))
		m.Step()
		losses = append(losses, mean)
		optSteps++
		accumLoss = 0
		accumN = 0
		if opt.OutDir != "" && job.CheckpointEvery > 0 && optSteps%job.CheckpointEvery == 0 {
			ckpt := filepath.Join(opt.OutDir, fmt.Sprintf("step-%d", global))
			if err := m.Save(ckpt); err != nil {
				return err
			}
			if err := dist.SaveCheckpoint(ckpt, dist.CheckpointMeta{Step: global, Loss: mean, Recipe: recipe, Rank: m.Cfg.Rank}); err != nil {
				return err
			}
		}
		return nil
	}
	for step := 0; step < opt.Steps; step++ {
		var batch []int
		var mask []bool
		if useMask {
			s := samples[(startStep+step)%len(samples)]
			batch = s.ids
			mask = s.mask
			if len(batch) > seq {
				batch = batch[len(batch)-seq:]
				mask = mask[len(mask)-seq:]
			}
		} else {
			nWin := len(ids) - seq + 1
			off := windowOffset(opt.Seed, startStep+step, nWin)
			batch = ids[off : off+seq]
		}
		zeroFirst := accumN == 0
		var loss float32
		if useMask {
			loss, err = m.AccumulateLossMasked(batch, mask, zeroFirst)
		} else {
			loss, err = m.AccumulateLoss(batch, zeroFirst)
		}
		if err != nil {
			return losses, err
		}
		accumLoss += loss
		accumN++
		global := startStep + step + 1
		if accumN >= job.AccumSteps {
			if err := syncAndMaybeCkpt(global); err != nil {
				return losses, err
			}
		}
		if opt.EvalEvery > 0 && (step+1)%opt.EvalEvery == 0 {
			_ = m.ForwardLogits(batch) // smoke eval forward
		}
	}
	if accumN > 0 {
		global := startStep + opt.Steps
		if err := syncAndMaybeCkpt(global); err != nil {
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
		if err := dist.SaveCheckpoint(opt.OutDir, dist.CheckpointMeta{
			Step: startStep + opt.Steps, Loss: last, Recipe: recipe, Rank: m.Cfg.Rank,
		}); err != nil {
			return losses, err
		}
	}
	return losses, nil
}

func windowOffset(seed int64, step, nWin int) int {
	if nWin <= 1 {
		return 0
	}
	if seed == 0 {
		return (step * 3) % nWin
	}
	r := rand.New(rand.NewPCG(uint64(seed), uint64(step+1))) //nolint:gosec // G404: deterministic window offset, not crypto
	return r.IntN(nWin)
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
		loss, err := m.StepLoss(ids[:seq])
		if err != nil {
			return losses, m, err
		}
		losses = append(losses, loss)
	}
	return losses, m, nil
}

// encodeMaskedSFT jointly encodes prompt+completion (same as MessagesToText)
// and masks tokens that fall in the completion, including a token that straddles
// the prompt/completion boundary.
func encodeMaskedSFT(tok *gpt2.Tokenizer, prompt, completion string) ([]int, []bool) {
	full := prompt + completion
	ids := tok.Encode(full)
	if len(ids) == 0 {
		return nil, nil
	}
	split := promptTokenCount(tok, ids, prompt, full)
	if split < 0 {
		split = 0
	}
	if split > len(ids) {
		split = len(ids)
	}
	mask := make([]bool, len(ids))
	for i := split; i < len(ids); i++ {
		mask[i] = true
	}
	return ids, mask
}

func promptTokenCount(tok *gpt2.Tokenizer, ids []int, prompt, full string) int {
	if tok.Decode(ids) == full {
		split := 0
		for n := 1; n <= len(ids); n++ {
			if len(tok.Decode(ids[:n])) <= len(prompt) {
				split = n
				continue
			}
			break
		}
		return split
	}
	// Tiny/incomplete vocabs may not round-trip through Decode. Fall back to
	// the common prefix of Encode(prompt) and the joint ids.
	pIDs := tok.Encode(prompt)
	n := 0
	for n < len(pIDs) && n < len(ids) && pIDs[n] == ids[n] {
		n++
	}
	return n
}
