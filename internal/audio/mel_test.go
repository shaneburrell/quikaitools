package audio

import (
	"path/filepath"
	"testing"
)

func TestWAVToneMel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tone.wav")
	if err := WriteToneWAV(path, 16000, 16000, 440); err != nil {
		t.Fatal(err)
	}
	samples, rate, err := LoadWAVMono16(path)
	if err != nil || rate != 16000 || len(samples) != 16000 {
		t.Fatalf("rate=%d len=%d err=%v", rate, len(samples), err)
	}
	rs := Resample(samples, 16000, 8000)
	if len(rs) < 7000 || len(rs) > 9000 {
		t.Fatalf("resample len=%d", len(rs))
	}
	mel := LogMel(samples[:3200], 16000, 400, 160, 40)
	if mel.NMels != 40 || mel.NFrames < 1 || len(mel.Data) != mel.NMels*mel.NFrames {
		t.Fatalf("%+v", mel)
	}
}
