package vision

import (
	"image"
	"image/color"
	"os"
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

func TestGoldenSolidRedBothNorms(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "red2x2.png")
	if err := SolidPNG(path, color.RGBA{R: 255, A: 255}, 2, 2); err != nil {
		t.Fatal(err)
	}
	presets := []struct {
		name string
		norm Normalization
	}{
		{"imagenet", ImageNetNorm},
		{"clip", CLIPNorm},
	}
	for _, p := range presets {
		t.Run(p.name, func(t *testing.T) {
			ten, err := LoadAndPreprocessOpts(path, Options{Size: 224, Norm: p.norm})
			if err != nil {
				t.Fatal(err)
			}
			if ten.C != 3 || ten.H != 224 || ten.W != 224 {
				t.Fatalf("shape %d %d %d", ten.C, ten.H, ten.W)
			}
			want := [3]float32{
				(1 - p.norm.Mean[0]) / p.norm.Std[0],
				(0 - p.norm.Mean[1]) / p.norm.Std[1],
				(0 - p.norm.Mean[2]) / p.norm.Std[2],
			}
			hw := 224 * 224
			for c := 0; c < 3; c++ {
				for i := 0; i < hw; i++ {
					got := ten.Data[c*hw+i]
					if d := abs32(got - want[c]); d > 1e-5 {
						t.Fatalf("c=%d i=%d got=%g want=%g delta=%g", c, i, got, want[c], d)
					}
				}
			}
		})
	}
}

func TestCenterCropTallAndWide(t *testing.T) {
	cases := []struct {
		name string
		w, h int
	}{
		{"tall", 32, 64},
		{"wide", 64, 32},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tc.w, tc.h))
			// Dark border, bright center block so the crop must keep the middle.
			for y := 0; y < tc.h; y++ {
				for x := 0; x < tc.w; x++ {
					img.Set(x, y, color.RGBA{A: 255})
				}
			}
			cx0, cx1 := tc.w/2-4, tc.w/2+4
			cy0, cy1 := tc.h/2-4, tc.h/2+4
			for y := cy0; y < cy1; y++ {
				for x := cx0; x < cx1; x++ {
					img.Set(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
				}
			}
			dir := t.TempDir()
			path := filepath.Join(dir, "center.png")
			if err := writePNG(path, img); err != nil {
				t.Fatal(err)
			}
			ten, err := LoadAndPreprocessOpts(path, Options{Size: 224, Norm: ImageNetNorm})
			if err != nil {
				t.Fatal(err)
			}
			if ten.C != 3 || ten.H != 224 || ten.W != 224 || len(ten.Data) != 3*224*224 {
				t.Fatalf("shape %+v len=%d", ten, len(ten.Data))
			}
			// After ImageNet norm, white ≈ (1-m)/s, black ≈ (0-m)/s. Center should be closer to white.
			whiteR := (1 - ImageNetNorm.Mean[0]) / ImageNetNorm.Std[0]
			blackR := (0 - ImageNetNorm.Mean[0]) / ImageNetNorm.Std[0]
			mid := ten.Data[0*224*224+112*224+112]
			corner := ten.Data[0]
			if abs32(mid-whiteR) > 0.15 {
				t.Fatalf("center R=%g want ~%g (white)", mid, whiteR)
			}
			if abs32(corner-blackR) > 0.15 {
				t.Fatalf("corner R=%g want ~%g (black)", corner, blackR)
			}
		})
	}
}

func TestBilinearRamp(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{A: 255})
	img.Set(1, 0, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	row := resizeBilinear(img, 4, 1, [3]float32{1, 1, 1})
	vals := []float32{row[0], row[3], row[6], row[9]}
	var sawMid bool
	for i := 1; i < 4; i++ {
		if vals[i] < vals[i-1]-1e-6 {
			t.Fatalf("not monotonic: %v", vals)
		}
		if vals[i] > 1e-4 && vals[i] < 1-1e-4 {
			sawMid = true
		}
	}
	if !sawMid {
		t.Fatalf("expected intermediate values, got %v", vals)
	}
	if vals[0] > 0.4 || vals[3] < 0.6 {
		t.Fatalf("endpoints not stretched toward black/white: %v", vals)
	}
}

func TestAlphaCompositesOverBackground(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	// fully transparent
	dir := t.TempDir()
	path := filepath.Join(dir, "clear.png")
	if err := writePNG(path, img); err != nil {
		t.Fatal(err)
	}
	bg := [3]float32{1, 1, 1}
	ten, err := LoadAndPreprocessOpts(path, Options{Size: 8, Norm: ImageNetNorm, Background: bg})
	if err != nil {
		t.Fatal(err)
	}
	hw := ten.H * ten.W
	for c := 0; c < 3; c++ {
		want := (bg[c] - ImageNetNorm.Mean[c]) / ImageNetNorm.Std[c]
		for i := 0; i < hw; i++ {
			got := ten.Data[c*hw+i]
			if d := abs32(got - want); d > 1e-5 {
				t.Fatalf("c=%d i=%d got=%g want=%g (background after norm)", c, i, got, want)
			}
		}
	}
}

func TestParseNormalization(t *testing.T) {
	im, err := ParseNormalization("ImageNet")
	if err != nil || im != ImageNetNorm {
		t.Fatalf("imagenet: %+v %v", im, err)
	}
	cl, err := ParseNormalization("CLIP")
	if err != nil || cl != CLIPNorm {
		t.Fatalf("clip: %+v %v", cl, err)
	}
	if _, err := ParseNormalization("unknown"); err == nil {
		t.Fatal("expected error for unknown name")
	}
	if _, err := ParseNormalization(""); err == nil {
		t.Fatal("expected error for empty name")
	}
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return encodePNG(f, img)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func TestLoadAndPreprocessDefaultSize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "red.png")
	if err := SolidPNG(path, color.RGBA{R: 255, A: 255}, 2, 2); err != nil {
		t.Fatal(err)
	}
	ten, err := LoadAndPreprocess(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if ten.H != 224 || ten.W != 224 {
		t.Fatalf("default size: %d x %d", ten.H, ten.W)
	}
}
