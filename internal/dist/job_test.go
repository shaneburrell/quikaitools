package dist

import (
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/backend"
)

func TestNewJobFP16OnlyOnCUDA(t *testing.T) {
	mac, _ := backend.Detect(backend.KindMac, func() backend.HostInfo { return backend.HostInfo{GOOS: "darwin"} })
	j := NewJob(mac)
	if j.FP16 {
		t.Fatal("mac should not default FP16")
	}
	v100, _ := backend.Detect(backend.KindV100, func() backend.HostInfo { return backend.HostInfo{GOOS: "linux"} })
	j2 := NewJob(v100)
	if !j2.FP16 {
		t.Fatal("v100 should FP16")
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := SaveCheckpoint(dir, CheckpointMeta{Step: 5, Loss: 1.2, Recipe: "lora", Rank: 4}); err != nil {
		t.Fatal(err)
	}
	m, err := LoadCheckpointMeta(dir)
	if err != nil || m.Step != 5 {
		t.Fatalf("%+v %v", m, err)
	}
	if filepath.Base(dir) == "" {
		t.Fatal("empty")
	}
}

func TestValidateAccum(t *testing.T) {
	j := Job{AccumSteps: 0, ShardMode: ShardModeNone}
	if err := j.Validate(); err == nil {
		t.Fatal("expected error")
	}
}
