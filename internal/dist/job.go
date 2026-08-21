// Package dist is Accelerate-lite: device profile, accum, checkpoint, FP16 flag.
package dist

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shaneburrell/quikaitools/internal/backend"
)

// ShardMode is FSDP/Shardy placeholder — MVP is None only.
type ShardMode string

const (
	ShardModeNone ShardMode = "none"
)

// Job is the trainer job config.
type Job struct {
	Profile       backend.Profile `json:"profile"`
	AccumSteps    int             `json:"accum_steps"`
	CheckpointDir string          `json:"checkpoint_dir"`
	CheckpointEvery int           `json:"checkpoint_every"`
	FP16          bool            `json:"fp16"`
	ShardMode     ShardMode       `json:"shard_mode"`
	ResumeFrom    string          `json:"resume_from,omitempty"`
}

// NewJob builds defaults from a profile.
func NewJob(p backend.Profile) Job {
	j := Job{
		Profile:         p,
		AccumSteps:      1,
		CheckpointEvery: 10,
		ShardMode:       ShardModeNone,
	}
	if p.MixedPrecision == "fp16" && (p.Kind == backend.KindV100 || p.Kind == backend.KindCUDA) {
		j.FP16 = true
	}
	return j
}

// CheckpointMeta is written next to adapter.json on save.
type CheckpointMeta struct {
	Step   int     `json:"step"`
	Loss   float32 `json:"loss"`
	Recipe string  `json:"recipe"`
	Rank   int     `json:"rank"`
}

// SaveCheckpoint writes meta.json under dir.
func SaveCheckpoint(dir string, meta CheckpointMeta) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "meta.json"), b, 0o644)
}

// LoadCheckpointMeta reads meta.json if present.
func LoadCheckpointMeta(dir string) (CheckpointMeta, error) {
	var m CheckpointMeta
	b, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, err
	}
	return m, nil
}

// Validate returns an error if the job is misconfigured.
func (j Job) Validate() error {
	if j.AccumSteps < 1 {
		return fmt.Errorf("dist: accum_steps must be >= 1")
	}
	if j.ShardMode != "" && j.ShardMode != ShardModeNone {
		return fmt.Errorf("dist: shard mode %q not supported in MVP (only none)", j.ShardMode)
	}
	if j.FP16 && j.Profile.Kind != backend.KindV100 && j.Profile.Kind != backend.KindCUDA {
		// allow flag but warn via notes — Mac/Halo stay FP32 in trainer
		return nil
	}
	return nil
}
