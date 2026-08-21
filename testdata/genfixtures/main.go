package main

import (
	"image/color"
	"path/filepath"

	"github.com/shaneburrell/quikaitools/internal/audio"
	"github.com/shaneburrell/quikaitools/internal/vision"
)

func main() {
	dir := "testdata/fixtures"
	_ = vision.SolidPNG(filepath.Join(dir, "red.png"), color.RGBA{R: 220, G: 40, B: 40, A: 255}, 64, 64)
	_ = audio.WriteToneWAV(filepath.Join(dir, "tone.wav"), 16000, 16000, 440)
}
