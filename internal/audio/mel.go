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

const (
	wavPCM16     = 1
	wavIEEEFloat = 3
)

// LoadWAVMono16 reads a WAV (PCM16 format 1 or IEEE float32 format 3) and returns mono float32 samples + rate.
func LoadWAVMono16(path string) (samples []float32, rate int, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("audio: not a WAV file")
	}
	i := 12
	var channels, bits, format int
	for i+8 <= len(b) {
		id := string(b[i : i+4])
		sz, ok := u32ToInt(binary.LittleEndian.Uint32(b[i+4 : i+8]))
		if !ok {
			return nil, 0, fmt.Errorf("audio: chunk too large")
		}
		i += 8
		if sz < 0 || i+sz > len(b) {
			break
		}
		switch id {
		case "fmt ":
			if sz < 16 {
				return nil, 0, fmt.Errorf("audio: short fmt")
			}
			format = int(binary.LittleEndian.Uint16(b[i : i+2]))
			channels = int(binary.LittleEndian.Uint16(b[i+2 : i+4]))
			urate := binary.LittleEndian.Uint32(b[i+4 : i+8])
			rate64, ok := u32ToInt(urate)
			if !ok {
				return nil, 0, fmt.Errorf("audio: sample rate too large")
			}
			rate = rate64
			bits = int(binary.LittleEndian.Uint16(b[i+14 : i+16]))
		case "data":
			pcm, err := decodeWAVData(b[i:i+sz], format, bits)
			if err != nil {
				return nil, 0, err
			}
			samples, err = downmixMono(pcm, channels)
			if err != nil {
				return nil, 0, err
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

func decodeWAVData(raw []byte, format, bits int) ([]float32, error) {
	switch format {
	case wavPCM16:
		if bits != 16 {
			return nil, fmt.Errorf("audio: PCM16 wants 16-bit, got %d", bits)
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for j := 0; j < n; j++ {
			v := uint16ToInt16(binary.LittleEndian.Uint16(raw[j*2 : j*2+2]))
			out[j] = float32(v) / 32768
		}
		return out, nil
	case wavIEEEFloat:
		if bits != 32 {
			return nil, fmt.Errorf("audio: IEEE float wants 32-bit, got %d", bits)
		}
		n := len(raw) / 4
		out := make([]float32, n)
		for j := 0; j < n; j++ {
			out[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j*4 : j*4+4]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("audio: unsupported format %d (supported: PCM16 format %d, IEEE float32 format %d)", format, wavPCM16, wavIEEEFloat)
	}
}

func downmixMono(interleaved []float32, channels int) ([]float32, error) {
	switch {
	case channels == 1:
		return interleaved, nil
	case channels >= 2:
		n := len(interleaved) / channels
		out := make([]float32, n)
		ch := float32(channels)
		for i := 0; i < n; i++ {
			var s float32
			base := i * channels
			for c := 0; c < channels; c++ {
				s += interleaved[base+c]
			}
			out[i] = s / ch
		}
		return out, nil
	default:
		return nil, fmt.Errorf("audio: unsupported channels %d", channels)
	}
}

// uint16ToInt16 reinterprets a two's-complement bit pattern. Safe: every uint16 is a valid int16 encoding.
func uint16ToInt16(v uint16) int16 {
	return int16(v)
}

// int16ToUint16 reinterprets a two's-complement bit pattern for WAV PCM writers.
func int16ToUint16(v int16) uint16 {
	return uint16(v)
}

// u32ToInt converts a uint32 to int after a range check (gosec G115).
func u32ToInt(v uint32) (int, bool) {
	if uint64(v) > uint64(math.MaxInt) {
		return 0, false
	}
	return int(v), true
}

// intToUint32 converts n to uint32 after a range check (gosec G115).
func intToUint32(n int) (uint32, error) {
	if n < 0 || uint64(n) > uint64(math.MaxUint32) {
		return 0, fmt.Errorf("audio: value %d out of uint32 range", n)
	}
	return uint32(n), nil
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
// For Whisper-compatible features, use LogMelWhisper.
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

const (
	whisperNFFT = 400
	whisperHop  = 160
	whisperRate = 16000
)

// LogMelWhisper computes a Whisper-exact log-mel spectrogram.
// n_fft=400, hop=160, periodic Hann, reflect-pad n_fft/2 (center=True),
// power spectrum, drop last frame, Slaney mel (fmin=0, fmax=8000), then
// log10(max(x,1e-10)), clamp to (max-8), (x+4)/4.
// sampleRate must be 16000.
func LogMelWhisper(samples []float32, sampleRate int, nMels int) (Mel, error) {
	if sampleRate != whisperRate {
		return Mel{}, fmt.Errorf("audio: LogMelWhisper requires sampleRate %d, got %d", whisperRate, sampleRate)
	}
	if nMels <= 0 {
		nMels = 80
	}
	nFFT := whisperNFFT
	hop := whisperHop
	padded := reflectPad(samples, nFFT/2)
	// Whisper / torch.stft drops the last frame: frames = len(samples)/hop
	nFrames := len(samples) / hop
	fb := slaneyMelFilterbank(nMels, nFFT, sampleRate, 0, float64(sampleRate)/2)
	data := make([]float32, nMels*nFrames)
	window := hannPeriodic(nFFT)
	for f := 0; f < nFrames; f++ {
		off := f * hop
		frame := make([]float32, nFFT)
		for i := 0; i < nFFT && off+i < len(padded); i++ {
			frame[i] = padded[off+i] * window[i]
		}
		power := rfftPower(frame)
		for m := 0; m < nMels; m++ {
			var e float32
			for k, w := range fb[m] {
				if k < len(power) {
					e += w * power[k]
				}
			}
			data[m*nFrames+f] = e
		}
	}
	for i, v := range data {
		x := float64(v)
		if x < 1e-10 {
			x = 1e-10
		}
		data[i] = float32(math.Log10(x))
	}
	if len(data) > 0 {
		mx := data[0]
		for _, v := range data[1:] {
			if v > mx {
				mx = v
			}
		}
		floor := mx - 8
		for i, v := range data {
			if v < floor {
				v = floor
			}
			data[i] = (v + 4) / 4
		}
	}
	return Mel{NMels: nMels, NFrames: nFrames, SampleRate: sampleRate, Data: data}, nil
}

func hann(n int) []float32 {
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1))))
	}
	return w
}

// hannPeriodic matches torch.hann_window(n, periodic=True).
func hannPeriodic(n int) []float32 {
	w := make([]float32, n)
	if n <= 0 {
		return w
	}
	for i := 0; i < n; i++ {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n))))
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

func rfftPower(x []float32) []float32 {
	n := len(x)
	out := make([]float32, n/2+1)
	for k := 0; k <= n/2; k++ {
		var re, im float64
		for t := 0; t < n; t++ {
			ang := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			re += float64(x[t]) * math.Cos(ang)
			im += float64(x[t]) * math.Sin(ang)
		}
		out[k] = float32(re*re + im*im)
	}
	return out
}

func reflectPad(x []float32, pad int) []float32 {
	n := len(x)
	if pad <= 0 {
		return append([]float32(nil), x...)
	}
	out := make([]float32, n+2*pad)
	copy(out[pad:pad+n], x)
	for i := 0; i < pad; i++ {
		out[pad-1-i] = x[reflectIndex(-1-i, n)]
		out[pad+n+i] = x[reflectIndex(n+i, n)]
	}
	return out
}

// reflectIndex maps i outside [0,n) by reflecting without duplicating edges (numpy/torch reflect).
func reflectIndex(i, n int) int {
	if n <= 1 {
		return 0
	}
	period := 2 * (n - 1)
	i %= period
	if i < 0 {
		i += period
	}
	if i >= n {
		i = period - i
	}
	return i
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

// slaneyMelFilterbank matches librosa.filters.mel(..., htk=False, norm="slaney").
func slaneyMelFilterbank(nMels, nFFT, sr int, fmin, fmax float64) [][]float32 {
	nFreq := nFFT/2 + 1
	fb := make([][]float32, nMels)
	melMin := hzToMelSlaney(fmin)
	melMax := hzToMelSlaney(fmax)
	melF := make([]float64, nMels+2)
	for i := range melF {
		melF[i] = melToHzSlaney(melMin + (melMax-melMin)*float64(i)/float64(nMels+1))
	}
	fftfreqs := make([]float64, nFreq)
	for k := 0; k < nFreq; k++ {
		fftfreqs[k] = float64(k) * float64(sr) / float64(nFFT)
	}
	for m := 0; m < nMels; m++ {
		fb[m] = make([]float32, nFreq)
		left, center, right := melF[m], melF[m+1], melF[m+2]
		d1 := center - left
		d2 := right - center
		for k := 0; k < nFreq; k++ {
			f := fftfreqs[k]
			var w float64
			if f >= left && f <= center && d1 > 0 {
				w = (f - left) / d1
			} else if f > center && f <= right && d2 > 0 {
				w = (right - f) / d2
			}
			if w < 0 {
				w = 0
			}
			fb[m][k] = float32(w)
		}
		// Slaney area normalization: 2 / (hz[m+2] - hz[m])
		width := melF[m+2] - melF[m]
		if width > 0 {
			enorm := 2.0 / width
			for k := range fb[m] {
				fb[m][k] *= float32(enorm)
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

func hzToMelSlaney(hz float64) float64 {
	const (
		fMin      = 0.0
		fSp       = 200.0 / 3
		minLogHz  = 1000.0
		minLogMel = (minLogHz - fMin) / fSp
	)
	logstep := math.Log(6.4) / 27.0
	if hz >= minLogHz {
		return minLogMel + math.Log(hz/minLogHz)/logstep
	}
	return (hz - fMin) / fSp
}

func melToHzSlaney(mel float64) float64 {
	const (
		fMin      = 0.0
		fSp       = 200.0 / 3
		minLogHz  = 1000.0
		minLogMel = (minLogHz - fMin) / fSp
	)
	logstep := math.Log(6.4) / 27.0
	if mel >= minLogMel {
		return minLogHz * math.Exp(logstep*(mel-minLogMel))
	}
	return fMin + fSp*mel
}

// WriteToneWAV writes a mono 16-bit sine WAV for tests.
func WriteToneWAV(path string, rate, samples int, hz float64) error {
	if rate <= 0 || samples <= 0 {
		return fmt.Errorf("audio: invalid tone rate=%d samples=%d", rate, samples)
	}
	if samples > (math.MaxInt-44)/2 {
		return fmt.Errorf("audio: too many samples")
	}
	riffSize, err := intToUint32(36 + samples*2)
	if err != nil {
		return err
	}
	urate, err := intToUint32(rate)
	if err != nil {
		return err
	}
	byteRate, err := intToUint32(rate * 2)
	if err != nil {
		return err
	}
	dataBytes, err := intToUint32(samples * 2)
	if err != nil {
		return err
	}
	data := make([]byte, 44+samples*2)
	copy(data[0:4], "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], riffSize)
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(data[22:24], 1) // mono
	binary.LittleEndian.PutUint32(data[24:28], urate)
	binary.LittleEndian.PutUint32(data[28:32], byteRate)
	binary.LittleEndian.PutUint16(data[32:34], 2)
	binary.LittleEndian.PutUint16(data[34:36], 16)
	copy(data[36:40], "data")
	binary.LittleEndian.PutUint32(data[40:44], dataBytes)
	for i := 0; i < samples; i++ {
		v := int16(math.Sin(2*math.Pi*hz*float64(i)/float64(rate)) * 16000)
		binary.LittleEndian.PutUint16(data[44+i*2:], int16ToUint16(v))
	}
	return os.WriteFile(path, data, 0o644)
}
