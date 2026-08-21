package infer

import (
	"image/color"
	"path/filepath"
	"testing"

	"github.com/shaneburrell/quikaitools/internal/audio"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/vision"
)

func TestEmbedTextBagWTE(t *testing.T) {
	dir := t.TempDir()
	// minimal: use NewRandom saved? LoadDir needs safetensors — skip to hash path via empty onnx
	// create fake onnx marker
	if err := osWrite(filepath.Join(dir, "model.onnx"), []byte("x")); err != nil {
		t.Fatal(err)
	}
	v, eng, err := EmbedText(EmbedTextOptions{ModelDir: dir, Text: "hello"})
	if err != nil || len(v) == 0 || eng == "" {
		t.Fatalf("%v %s %v", v, eng, err)
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
	wav := filepath.Join(dir, "t.wav")
	if err := audio.WriteToneWAV(wav, 16000, 8000, 220); err != nil {
		t.Fatal(err)
	}
	text, eng2, err := Transcribe(TranscribeOptions{ModelDir: dir, Audio: wav})
	if err != nil || text == "" || eng2 == "" {
		t.Fatalf("%q %s %v", text, eng2, err)
	}
	_ = gpt2.Config{}
}

func osWrite(path string, b []byte) error {
	return writeFile(path, b)
}
