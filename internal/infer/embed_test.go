package infer

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"image/color"

	"github.com/shaneburrell/quikaitools/internal/audio"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/vision"
)

func TestEmbedTextONNXStubHash(t *testing.T) {
	dir := t.TempDir()
	if err := osWrite(filepath.Join(dir, "model.onnx"), []byte("x")); err != nil {
		t.Fatal(err)
	}
	v, eng, err := EmbedText(EmbedTextOptions{ModelDir: dir, Text: "hello"})
	if err != nil || len(v) == 0 {
		t.Fatalf("%v %s %v", v, eng, err)
	}
	if eng != "onnx-stub-hash" {
		t.Fatalf("engine=%q want onnx-stub-hash", eng)
	}
}

func TestEmbedTextBagWTE(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 16, NHead: 4, NLayer: 2, NPositions: 32, VocabSize: 32, LayerNormEps: 1e-5, NInner: 32}
	m := gpt2.NewRandom(cfg, 3)
	dir := t.TempDir()
	if err := gpt2.WriteDir(dir, m); err != nil {
		t.Fatal(err)
	}
	v, eng, err := EmbedText(EmbedTextOptions{ModelDir: dir, Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if eng != "bag-of-wte" {
		t.Fatalf("engine=%q want bag-of-wte", eng)
	}
	if len(v) != cfg.NEmbd {
		t.Fatalf("dim=%d want %d", len(v), cfg.NEmbd)
	}
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if math.Abs(s-1) > 1e-4 {
		t.Fatalf("vector not L2-normalized: ||v||^2=%g", s)
	}
}

func TestEmbedTextOOV(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 8, NHead: 2, NLayer: 1, NPositions: 16, VocabSize: 2, LayerNormEps: 1e-5}
	m := gpt2.NewRandom(cfg, 1)
	dir := t.TempDir()
	if err := gpt2.WriteDir(dir, m); err != nil {
		t.Fatal(err)
	}
	vocab := map[string]int{"<|endoftext|>": 0, "a": 1, "h": 8, "e": 5, "l": 12, "o": 15}
	b, err := json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "vocab.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err = EmbedText(EmbedTextOptions{ModelDir: dir, Text: "hello"})
	if err == nil || !strings.Contains(err.Error(), "outside vocab") {
		t.Fatalf("want OOV error, got %v", err)
	}
}

func TestEmbedTextStrictStub(t *testing.T) {
	dir := t.TempDir()
	if err := osWrite(filepath.Join(dir, "model.onnx"), []byte("x")); err != nil {
		t.Fatal(err)
	}
	_, _, err := EmbedText(EmbedTextOptions{ModelDir: dir, Text: "hello", StrictStub: true})
	if err == nil || !strings.Contains(err.Error(), "onnx-stub-hash") || !strings.Contains(err.Error(), "allow-stub") {
		t.Fatalf("want stub error, got %v", err)
	}
}

func TestEmbedVisionAndTranscribe(t *testing.T) {
	dir := t.TempDir()
	img := filepath.Join(dir, "a.png")
	if err := vision.SolidPNG(img, color.RGBA{G: 255, A: 255}, 64, 64); err != nil {
		t.Fatal(err)
	}
	v, eng, err := EmbedVision(EmbedVisionOptions{ModelDir: dir, Image: img, Size: 32})
	if err != nil || len(v) != 64 || eng == "" {
		t.Fatalf("%v %s %v", len(v), eng, err)
	}
	if !strings.Contains(eng, "stub") {
		t.Fatalf("vision engine %q should contain stub", eng)
	}
	wav := filepath.Join(dir, "t.wav")
	if err := audio.WriteToneWAV(wav, 16000, 8000, 220); err != nil {
		t.Fatal(err)
	}
	text, eng2, err := Transcribe(TranscribeOptions{ModelDir: dir, Audio: wav})
	if err != nil || text == "" || eng2 == "" {
		t.Fatalf("%q %s %v", text, eng2, err)
	}
	if !strings.Contains(eng2, "stub") {
		t.Fatalf("transcribe engine %q should contain stub", eng2)
	}
	_ = gpt2.Config{}
}

func osWrite(path string, b []byte) error {
	return writeFile(path, b)
}
