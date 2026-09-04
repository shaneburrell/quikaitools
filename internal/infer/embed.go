// Package infer ONNX-adjacent helpers: text/vision/audio embeddings without requiring ORT in CI.
package infer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shaneburrell/quikaitools/internal/audio"
	"github.com/shaneburrell/quikaitools/internal/gpt2"
	"github.com/shaneburrell/quikaitools/internal/vision"
)

// EmbedTextOptions configures text embedding.
type EmbedTextOptions struct {
	ModelDir string
	Text     string
	// StrictStub, when true, returns an error instead of placeholder stub output
	// (ONNX hash / vision-gap / whisper-mel). Default false preserves today's
	// permissive CLI behavior; wire --allow-stub=false to StrictStub later.
	StrictStub bool
}

// EmbedText returns a bag-of-token-embedding mean from GPT-2 WTE (portable MVP).
// When a real ONNX session is wired later, this becomes the fallback.
func EmbedText(opt EmbedTextOptions) ([]float32, string, error) {
	if opt.Text == "" {
		return nil, "", fmt.Errorf("embed: empty text")
	}
	// Prefer GPT-2 style folder
	if _, err := os.Stat(filepath.Join(opt.ModelDir, "model.safetensors")); err == nil {
		m, err := gpt2.LoadDir(opt.ModelDir)
		if err != nil {
			return nil, "", err
		}
		tok, err := gpt2.LoadTokenizer(opt.ModelDir)
		if err != nil {
			return nil, "", err
		}
		ids := tok.Encode(opt.Text)
		if len(ids) == 0 {
			return nil, "", fmt.Errorf("embed: no tokens")
		}
		d := m.Cfg.NEmbd
		out := make([]float32, d)
		for _, id := range ids {
			if id < 0 || id >= m.Cfg.VocabSize {
				return nil, "", fmt.Errorf("embed: token id %d outside vocab %d", id, m.Cfg.VocabSize)
			}
			row := m.WTE[id*d : (id+1)*d]
			for j := 0; j < d; j++ {
				out[j] += row[j]
			}
		}
		inv := 1 / float32(len(ids))
		for j := range out {
			out[j] *= inv
		}
		l2normalize(out)
		return out, "bag-of-wte", nil
	}
	// ONNX folder: stub embedding from text hash until ORT is linked
	if hasONNX(opt.ModelDir) {
		if err := denyStub(opt.StrictStub, "onnx-stub-hash"); err != nil {
			return nil, "", err
		}
		out := hashEmbed(opt.Text, 384)
		return out, "onnx-stub-hash", nil
	}
	return nil, "", fmt.Errorf("embed: no model.safetensors or .onnx under %s", opt.ModelDir)
}

// EmbedVisionOptions configures image embedding.
type EmbedVisionOptions struct {
	ModelDir string
	Image    string
	Size     int
	// Normalize selects the RGB mean/std preset: "imagenet" (default) or "clip".
	Normalize string
	// StrictStub opt-in fail-closed for stub engines. See EmbedTextOptions.
	StrictStub bool
}

// EmbedVision preprocesses the image; returns channel means + spatial stats (stub) or notes onnx.
func EmbedVision(opt EmbedVisionOptions) ([]float32, string, error) {
	norm := vision.ImageNetNorm
	if opt.Normalize != "" {
		n, err := vision.ParseNormalization(opt.Normalize)
		if err != nil {
			return nil, "", err
		}
		norm = n
	}
	tp, err := vision.LoadAndPreprocessOpts(opt.Image, vision.Options{Size: opt.Size, Norm: norm})
	if err != nil {
		return nil, "", err
	}
	ten := *tp
	// Global average pool → 3 dims, then pad/project to 64 with spatial std
	out := make([]float32, 64)
	hw := ten.H * ten.W
	for c := 0; c < 3; c++ {
		var sum, sumsq float32
		row := ten.Data[c*hw : (c+1)*hw]
		for _, v := range row {
			sum += v
			sumsq += v * v
		}
		mean := sum / float32(hw)
		out[c] = mean
		v := sumsq/float32(hw) - mean*mean
		if v < 0 {
			v = 0
		}
		out[3+c] = float32(math.Sqrt(float64(v)))
	}
	out[6] = float32(ten.H)
	out[7] = float32(ten.W)
	engine := "vision-gap-stub"
	if hasONNX(opt.ModelDir) {
		engine = "onnx-stub+vision-preprocess"
	}
	if err := denyStub(opt.StrictStub, engine); err != nil {
		return nil, "", err
	}
	l2normalize(out)
	return out, engine, nil
}

// TranscribeOptions configures ASR.
type TranscribeOptions struct {
	ModelDir string
	Audio    string
	// StrictStub opt-in fail-closed for stub engines. See EmbedTextOptions.
	StrictStub bool
}

// Transcribe loads WAV, builds log-mel, returns a stub transcript (ORT Whisper later).
func Transcribe(opt TranscribeOptions) (string, string, error) {
	samples, rate, err := audio.LoadWAVMono16(opt.Audio)
	if err != nil {
		return "", "", err
	}
	if rate != 16000 {
		samples = audio.Resample(samples, rate, 16000)
		rate = 16000
	}
	mel := audio.LogMel(samples, rate, 400, 160, 80)
	var energy float32
	for _, v := range mel.Data {
		energy += v
	}
	energy /= float32(len(mel.Data) + 1)
	engine := "whisper-mel-stub"
	if hasONNX(opt.ModelDir) {
		engine = "onnx-stub+whisper-mel"
	}
	if err := denyStub(opt.StrictStub, engine); err != nil {
		return "", "", err
	}
	text := fmt.Sprintf("[stub transcript] frames=%d mels=%d energy=%.4f", mel.NFrames, mel.NMels, energy)
	return text, engine, nil
}

func denyStub(strict bool, engine string) error {
	if !strict {
		return nil
	}
	return fmt.Errorf("embed: %s is a stub (ONNX runtime not linked); pass --allow-stub to get placeholder output", engine)
}

func hasONNX(dir string) bool {
	found := false
	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		if strings.HasSuffix(strings.ToLower(info.Name()), ".onnx") {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

func hashEmbed(s string, dim int) []float32 {
	out := make([]float32, dim)
	var h uint64 = 14695981039346656037
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
		out[int(h%uint64(dim))] += 1
	}
	l2normalize(out)
	return out
}

func l2normalize(v []float32) {
	var s float32
	for _, x := range v {
		s += x * x
	}
	if s < 1e-12 {
		return
	}
	inv := float32(1 / math.Sqrt(float64(s)))
	for i := range v {
		v[i] *= inv
	}
}
