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
	"strings"
)

// ImageNet mean/std (RGB). Kept for callers that read the exported vars.
var (
	Mean = [3]float32{0.485, 0.456, 0.406}
	Std  = [3]float32{0.229, 0.224, 0.225}
)

// Normalization is per-channel RGB mean/std applied after converting pixels to [0,1].
type Normalization struct {
	Mean, Std [3]float32
}

// ImageNetNorm is the torchvision ImageNet normalization preset.
var ImageNetNorm = Normalization{
	Mean: [3]float32{0.485, 0.456, 0.406},
	Std:  [3]float32{0.229, 0.224, 0.225},
}

// CLIPNorm is OpenAI CLIP's RGB normalization preset.
var CLIPNorm = Normalization{
	Mean: [3]float32{0.48145466, 0.4578275, 0.40821073},
	Std:  [3]float32{0.26862954, 0.26130258, 0.27577711},
}

// ParseNormalization accepts "imagenet" or "clip" (case-insensitive).
func ParseNormalization(name string) (Normalization, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "imagenet":
		return ImageNetNorm, nil
	case "clip":
		return CLIPNorm, nil
	default:
		return Normalization{}, fmt.Errorf("vision: unknown normalization %q (want imagenet or clip)", name)
	}
}

// Options configures decode → resize → center-crop → normalize.
// Zero Size defaults to 224. Zero Norm defaults to ImageNetNorm.
// Zero Background defaults to white (1,1,1).
type Options struct {
	Size       int
	Norm       Normalization
	Background [3]float32
}

// Tensor is CHW float32 in [0,1] then normalized.
type Tensor struct {
	C, H, W int
	Data    []float32
}

// LoadAndPreprocess opens path, resizes shortest side then center-crops to size×size, ImageNet-normalizes.
func LoadAndPreprocess(path string, size int) (Tensor, error) {
	t, err := LoadAndPreprocessOpts(path, Options{
		Size:       size,
		Norm:       ImageNetNorm,
		Background: [3]float32{1, 1, 1},
	})
	if err != nil {
		return Tensor{}, err
	}
	return *t, nil
}

// LoadAndPreprocessOpts is LoadAndPreprocess with explicit size, normalization, and alpha background.
func LoadAndPreprocessOpts(path string, o Options) (*Tensor, error) {
	o = o.withDefaults()
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("vision: decode: %w", err)
	}
	t := preprocess(img, o)
	return &t, nil
}

// Preprocess resizes with bilinear interpolation, center crops, CHW ImageNet norm.
func Preprocess(img image.Image, size int) Tensor {
	return preprocess(img, Options{
		Size:       size,
		Norm:       ImageNetNorm,
		Background: [3]float32{1, 1, 1},
	})
}

func (o Options) withDefaults() Options {
	if o.Size <= 0 {
		o.Size = 224
	}
	if o.Norm == (Normalization{}) {
		o.Norm = ImageNetNorm
	}
	if o.Background == ([3]float32{}) {
		o.Background = [3]float32{1, 1, 1}
	}
	return o
}

func preprocess(img image.Image, o Options) Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	size := o.Size
	if size < 0 {
		size = 0
	}
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
	resized := resizeBilinear(img, nw, nh, o.Background)
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
	mean, std := o.Norm.Mean, o.Norm.Std
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			sx, sy := x0+x, y0+y
			if sx >= nw {
				sx = nw - 1
			}
			if sy >= nh {
				sy = nh - 1
			}
			i := (sy*nw + sx) * 3
			rf := resized[i]
			gf := resized[i+1]
			bf := resized[i+2]
			idx := y*size + x
			t.Data[0*size*size+idx] = (rf - mean[0]) / std[0]
			t.Data[1*size*size+idx] = (gf - mean[1]) / std[1]
			t.Data[2*size*size+idx] = (bf - mean[2]) / std[2]
		}
	}
	return t
}

// resizeBilinear min-side (or arbitrary) bilinear resize to nw×nh, RGB float32 in [0,1], HWC packed.
// Samples at (x+0.5)*scale-0.5 with clamped neighbors.
func resizeBilinear(img image.Image, nw, nh int, bg [3]float32) []float32 {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw < 1 {
		sw = 1
	}
	if sh < 1 {
		sh = 1
	}
	out := make([]float32, nw*nh*3)
	scaleX := float64(sw) / float64(nw)
	scaleY := float64(sh) / float64(nh)
	for y := 0; y < nh; y++ {
		sy := (float64(y)+0.5)*scaleY - 0.5
		for x := 0; x < nw; x++ {
			sx := (float64(x)+0.5)*scaleX - 0.5
			rgb := bilinearSample(img, b, sx, sy, bg)
			i := (y*nw + x) * 3
			out[i], out[i+1], out[i+2] = rgb[0], rgb[1], rgb[2]
		}
	}
	return out
}

func bilinearSample(img image.Image, b image.Rectangle, sx, sy float64, bg [3]float32) [3]float32 {
	x0 := math.Floor(sx)
	y0 := math.Floor(sy)
	wx := sx - x0
	wy := sy - y0
	ix0 := clampIndex(int(x0), b.Dx()-1)
	ix1 := clampIndex(int(x0)+1, b.Dx()-1)
	iy0 := clampIndex(int(y0), b.Dy()-1)
	iy1 := clampIndex(int(y0)+1, b.Dy()-1)
	c00 := pixelRGB(img, b.Min.X+ix0, b.Min.Y+iy0, bg)
	c10 := pixelRGB(img, b.Min.X+ix1, b.Min.Y+iy0, bg)
	c01 := pixelRGB(img, b.Min.X+ix0, b.Min.Y+iy1, bg)
	c11 := pixelRGB(img, b.Min.X+ix1, b.Min.Y+iy1, bg)
	var out [3]float32
	for i := 0; i < 3; i++ {
		top := float64(c00[i])*(1-wx) + float64(c10[i])*wx
		bot := float64(c01[i])*(1-wx) + float64(c11[i])*wx
		out[i] = float32(top*(1-wy) + bot*wy)
	}
	return out
}

func pixelRGB(img image.Image, x, y int, bg [3]float32) [3]float32 {
	r, g, b, a := img.At(x, y).RGBA()
	const max16 = 65535.0
	rf := float64(r) / max16
	gf := float64(g) / max16
	bf := float64(b) / max16
	af := float64(a) / max16
	// RGBA() is premultiplied. Un-premultiply where a>0, then composite over background.
	if af > 0 {
		rf /= af
		gf /= af
		bf /= af
	}
	ia := 1 - af
	return [3]float32{
		float32(rf*af + float64(bg[0])*ia),
		float32(gf*af + float64(bg[1])*ia),
		float32(bf*af + float64(bg[2])*ia),
	}
}

func clampIndex(v, hi int) int {
	if hi < 0 {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > hi {
		return hi
	}
	return v
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
	defer func() { _ = f.Close() }()
	return encodePNG(f, img)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
