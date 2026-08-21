// Package audio is a torchaudio slice: WAV decode, mono, resample, log-mel.
package audio

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Mel is a log-mel spectrogram [nMels, nFrames].
type Mel struct {
	NMels, NFrames int
	SampleRate     int
	Data           []float32
}

// LoadWAVMono16 reads a PCM WAV (16-bit LE) and returns mono float32 samples + rate.
func LoadWAVMono16(path string) (samples []float32, rate int, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("audio: not a WAV file")
	}
	// find fmt and data chunks
	i := 12
	var channels, bits int
	for i+8 <= len(b) {
		id := string(b[i : i+4])
		sz := int(binary.LittleEndian.Uint32(b[i+4 : i+8]))
		i += 8
		if i+sz > len(b) {
			break
		}
		switch id {
		case "fmt ":
			if sz < 16 {
				return nil, 0, fmt.Errorf("audio: short fmt")
			}
			channels = int(binary.LittleEndian.Uint16(b[i+2 : i+4]))
			rate = int(binary.LittleEndian.Uint32(b[i+4 : i+8]))
			bits = int(binary.LittleEndian.Uint16(b[i+14 : i+16]))
		case "data":
			if bits != 16 {
				return nil, 0, fmt.Errorf("audio: want 16-bit PCM, got %d", bits)
			}
			n := sz / 2
			raw := make([]int16, n)
			for j := 0; j < n; j++ {
				raw[j] = int16(binary.LittleEndian.Uint16(b[i+j*2 : i+j*2+2]))
			}
			if channels == 1 {
				samples = make([]float32, n)
				for j, v := range raw {
					samples[j] = float32(v) / 32768
				}
			} else if channels == 2 {
				samples = make([]float32, n/2)
				for j := 0; j < len(samples); j++ {
					samples[j] = (float32(raw[j*2]) + float32(raw[j*2+1])) / 2 / 32768
				}
			} else {
				return nil, 0, fmt.Errorf("audio: unsupported channels %d", channels)
			}
			return samples, rate, nil
		}
		i += sz
		if sz%2 == 1 {
			i++
		}
	}
	return nil, 0, fmt.Errorf("audio: no data chunk")
}

// Resample linear to targetRate.
func Resample(x []float32, from, to int) []float32 {
	if from == to || from <= 0 || to <= 0 {
		return append([]float32(nil), x...)
	}
	n := int(float64(len(x)) * float64(to) / float64(from))
	if n < 1 {
		n = 1
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		src := float64(i) * float64(from) / float64(to)
		j := int(src)
		f := float32(src - float64(j))
		if j+1 < len(x) {
			out[i] = x[j]*(1-f) + x[j+1]*f
		} else if j < len(x) {
			out[i] = x[j]
		}
	}
	return out
}

// LogMel computes a simple log-mel (magnitude STFT → mel filterbank → log1p).
func LogMel(samples []float32, sampleRate, nFFT, hop, nMels int) Mel {
	if nFFT <= 0 {
		nFFT = 400
	}
	if hop <= 0 {
		hop = 160
	}
	if nMels <= 0 {
		nMels = 80
	}
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	nFrames := 1
	if len(samples) > nFFT {
		nFrames = 1 + (len(samples)-nFFT)/hop
	}
	fb := melFilterbank(nMels, nFFT, sampleRate)
	data := make([]float32, nMels*nFrames)
	window := hann(nFFT)
	for f := 0; f < nFrames; f++ {
		off := f * hop
		frame := make([]float32, nFFT)
		for i := 0; i < nFFT && off+i < len(samples); i++ {
			frame[i] = samples[off+i] * window[i]
		}
		mag := rfftMag(frame)
		for m := 0; m < nMels; m++ {
			var e float32
			for k, w := range fb[m] {
				if k < len(mag) {
					e += w * mag[k]
				}
			}
			data[m*nFrames+f] = float32(math.Log1p(float64(e)))
		}
	}
	return Mel{NMels: nMels, NFrames: nFrames, SampleRate: sampleRate, Data: data}
}

func hann(n int) []float32 {
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1))))
	}
	return w
}

func rfftMag(x []float32) []float32 {
	// naive DFT magnitude for real input (0..n/2)
	n := len(x)
	out := make([]float32, n/2+1)
	for k := 0; k <= n/2; k++ {
		var re, im float64
		for t := 0; t < n; t++ {
			ang := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			re += float64(x[t]) * math.Cos(ang)
			im += float64(x[t]) * math.Sin(ang)
		}
		out[k] = float32(math.Sqrt(re*re + im*im))
	}
	return out
}

func melFilterbank(nMels, nFFT, sr int) [][]float32 {
	nFreq := nFFT/2 + 1
	fb := make([][]float32, nMels)
	melMin := hzToMel(0)
	melMax := hzToMel(float64(sr) / 2)
	points := make([]float64, nMels+2)
	for i := range points {
		points[i] = melToHz(melMin + (melMax-melMin)*float64(i)/float64(nMels+1))
	}
	bins := make([]int, nMels+2)
	for i, hz := range points {
		bins[i] = int(math.Floor((float64(nFFT) + 1) * hz / float64(sr)))
		if bins[i] >= nFreq {
			bins[i] = nFreq - 1
		}
	}
	for m := 0; m < nMels; m++ {
		fb[m] = make([]float32, nFreq)
		left, center, right := bins[m], bins[m+1], bins[m+2]
		for k := left; k < center; k++ {
			if center != left && k >= 0 && k < nFreq {
				fb[m][k] = float32(k-left) / float32(center-left)
			}
		}
		for k := center; k < right; k++ {
			if right != center && k >= 0 && k < nFreq {
				fb[m][k] = float32(right-k) / float32(right-center)
			}
		}
	}
	return fb
}

func hzToMel(hz float64) float64 {
	return 2595 * math.Log10(1+hz/700)
}

func melToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2595) - 1)
}

// WriteToneWAV writes a mono 16-bit sine WAV for tests.
func WriteToneWAV(path string, rate, samples int, hz float64) error {
	data := make([]byte, 44+samples*2)
	copy(data[0:4], "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], uint32(36+samples*2))
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(data[22:24], 1) // mono
	binary.LittleEndian.PutUint32(data[24:28], uint32(rate))
	binary.LittleEndian.PutUint32(data[28:32], uint32(rate*2))
	binary.LittleEndian.PutUint16(data[32:34], 2)
	binary.LittleEndian.PutUint16(data[34:36], 16)
	copy(data[36:40], "data")
	binary.LittleEndian.PutUint32(data[40:44], uint32(samples*2))
	for i := 0; i < samples; i++ {
		v := int16(math.Sin(2*math.Pi*hz*float64(i)/float64(rate)) * 16000)
		binary.LittleEndian.PutUint16(data[44+i*2:], uint16(v))
	}
	return os.WriteFile(path, data, 0o644)
}
