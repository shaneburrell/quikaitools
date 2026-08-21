package vision

import (
	"image/color"
	"path/filepath"
	"testing"
)

func TestPreprocessSolid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "red.png")
	if err := SolidPNG(path, color.RGBA{R: 255, A: 255}, 32, 48); err != nil {
		t.Fatal(err)
	}
	ten, err := LoadAndPreprocess(path, 16)
	if err != nil {
		t.Fatal(err)
	}
	if ten.C != 3 || ten.H != 16 || ten.W != 16 || len(ten.Data) != 3*16*16 {
		t.Fatalf("%+v len=%d", ten, len(ten.Data))
	}
}
