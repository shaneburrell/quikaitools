// Package vision is a torchvision slice: decode, resize, crop, ImageNet normalize.
package vision

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
)

// ImageNet mean/std (RGB).
var (
	Mean = [3]float32{0.485, 0.456, 0.406}
	Std  = [3]float32{0.229, 0.224, 0.225}
)

// Tensor is CHW float32 in [0,1] then normalized.
type Tensor struct {
	C, H, W int
	Data    []float32
}

// LoadAndPreprocess opens path, resizes shortest side then center-crops to size×size, normalizes.
func LoadAndPreprocess(path string, size int) (Tensor, error) {
	if size <= 0 {
		size = 224
	}
	f, err := os.Open(path)
	if err != nil {
		return Tensor{}, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return Tensor{}, fmt.Errorf("vision: decode: %w", err)
	}
	return Preprocess(img, size), nil
}

// Preprocess resizes (bilinear-ish nearest for MVP), center crops, CHW ImageNet norm.
func Preprocess(img image.Image, size int) Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	// scale so min side == size
	scale := float64(size) / float64(min(w, h))
	nw := int(math.Round(float64(w) * scale))
	nh := int(math.Round(float64(h) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	resized := image.NewRGBA(image.Rect(0, 0, nw, nh))
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			sx := b.Min.X + x*w/nw
			sy := b.Min.Y + y*h/nh
			resized.Set(x, y, img.At(sx, sy))
		}
	}
	// center crop
	x0 := (nw - size) / 2
	y0 := (nh - size) / 2
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	t := Tensor{C: 3, H: size, W: size, Data: make([]float32, 3*size*size)}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := resized.At(x0+x, y0+y).RGBA()
			rf := float32(r>>8) / 255
			gf := float32(g>>8) / 255
			bf := float32(b>>8) / 255
			idx := y*size + x
			t.Data[0*size*size+idx] = (rf - Mean[0]) / Std[0]
			t.Data[1*size*size+idx] = (gf - Mean[1]) / Std[1]
			t.Data[2*size*size+idx] = (bf - Mean[2]) / Std[2]
		}
	}
	return t
}

// SolidPNG writes a tiny RGB PNG for fixtures/tests.
func SolidPNG(path string, c color.Color, w, h int) error {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return encodePNG(f, img)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
