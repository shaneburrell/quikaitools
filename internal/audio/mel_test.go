package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWAVToneMel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tone.wav")
	if err := WriteToneWAV(path, 16000, 16000, 440); err != nil {
		t.Fatal(err)
	}
	samples, rate, err := LoadWAVMono16(path)
	if err != nil || rate != 16000 || len(samples) != 16000 {
		t.Fatalf("rate=%d len=%d err=%v", rate, len(samples), err)
	}
	rs := Resample(samples, 16000, 8000)
	if len(rs) < 7000 || len(rs) > 9000 {
		t.Fatalf("resample len=%d", len(rs))
	}
	mel := LogMel(samples[:3200], 16000, 400, 160, 40)
	if mel.NMels != 40 || mel.NFrames < 1 || len(mel.Data) != mel.NMels*mel.NFrames {
		t.Fatalf("%+v", mel)
	}
}

func TestLogMelWhisperFrameCount(t *testing.T) {
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}
	mel, err := LogMelWhisper(samples, 16000, 80)
	if err != nil {
		t.Fatal(err)
	}
	if mel.NFrames != 100 {
		t.Fatalf("NFrames=%d want 100 (Whisper: 3000 frames / 30 s)", mel.NFrames)
	}
	if mel.NMels != 80 || len(mel.Data) != 80*100 {
		t.Fatalf("shape mels=%d data=%d", mel.NMels, len(mel.Data))
	}
}

func TestLogMelWhisperNormalizedRange(t *testing.T) {
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}
	mel, err := LogMelWhisper(samples, 16000, 80)
	if err != nil {
		t.Fatal(err)
	}
	mn, mx := mel.Data[0], mel.Data[0]
	for _, v := range mel.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite %v", v)
		}
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if mx-mn > 2.0+1e-5 {
		t.Fatalf("max-min=%g want <= 2.0", mx-mn)
	}
}

func TestLogMelWhisperWrongRate(t *testing.T) {
	if _, err := LogMelWhisper([]float32{0}, 8000, 80); err == nil {
		t.Fatal("expected error for sampleRate != 16000")
	}
}

func TestLogMelWhisperEmpty(t *testing.T) {
	if _, err := LogMelWhisper(nil, 16000, 80); err == nil {
		t.Fatal("expected error for empty samples")
	}
	if _, err := LogMelWhisper([]float32{}, 16000, 80); err == nil {
		t.Fatal("expected error for empty samples")
	}
}

func TestLoadWAVZeroRate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "zero.wav")
	if err := writeWAVWithRate(path, 0, 16); err != nil {
		t.Fatal(err)
	}
	_, _, err := LoadWAVMono16(path)
	if err == nil || !strings.Contains(err.Error(), "sample rate") {
		t.Fatalf("want sample rate error, got %v", err)
	}
}

func TestSlaneyFilterbankTriangles(t *testing.T) {
	fb := slaneyMelFilterbank(80, 400, 16000, 0, 8000)
	if len(fb) != 80 {
		t.Fatalf("rows=%d", len(fb))
	}
	for m, row := range fb {
		var sum float32
		peak := 0
		for k, w := range row {
			sum += w
			if w > row[peak] {
				peak = k
			}
		}
		if sum <= 0 || row[peak] <= 0 {
			t.Fatalf("filter %d sum=%g peak=%g want a positive triangle", m, sum, row[peak])
		}
		for k := 1; k <= peak; k++ {
			if row[k]+1e-7 < row[k-1] {
				t.Fatalf("filter %d not rising to peak at %d", m, peak)
			}
		}
		for k := peak + 1; k < len(row); k++ {
			if row[k] > row[k-1]+1e-7 {
				t.Fatalf("filter %d not falling after peak at %d", m, peak)
			}
		}
	}
}

func TestTone440MelBin(t *testing.T) {
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}
	nMels := 80
	mel, err := LogMelWhisper(samples, 16000, nMels)
	if err != nil {
		t.Fatal(err)
	}
	// FFT bin nearest 440 Hz: k = 440 * 400 / 16000 = 11
	fb := slaneyMelFilterbank(nMels, 400, 16000, 0, 8000)
	lo, hi := nMels, -1
	for m, row := range fb {
		if 11 < len(row) && row[11] > 0 {
			if m < lo {
				lo = m
			}
			if m > hi {
				hi = m
			}
		}
	}
	if hi < lo {
		t.Fatal("no Slaney filter covers 440 Hz")
	}
	// Per-frame argmax should land in the filters that cover ~440 Hz.
	hits := 0
	for f := 0; f < mel.NFrames; f++ {
		best, bestV := 0, mel.Data[f]
		for m := 1; m < nMels; m++ {
			v := mel.Data[m*mel.NFrames+f]
			if v > bestV {
				best, bestV = m, v
			}
		}
		if best >= lo && best <= hi {
			hits++
		}
	}
	if hits < mel.NFrames/2 {
		t.Fatalf("argmax in [%d,%d] for only %d/%d frames", lo, hi, hits, mel.NFrames)
	}
}

func TestFloat32WAVRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f32.wav")
	want := []float32{0, 0.25, -0.5, 1, -1, 0.125}
	if err := writeFloat32WAV(path, 16000, want); err != nil {
		t.Fatal(err)
	}
	got, rate, err := LoadWAVMono16(path)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Fatalf("rate=%d", rate)
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if abs32(got[i]-want[i]) > 1e-6 {
			t.Fatalf("sample[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestLoadWAVUnsupportedFormat(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.wav")
	// 8-bit PCM (format 1, bits 8) should fail listing supported formats.
	data := make([]byte, 44+4)
	copy(data[0:4], "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], 36+4)
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 2) // ADPCM — unsupported
	binary.LittleEndian.PutUint16(data[22:24], 1)
	binary.LittleEndian.PutUint32(data[24:28], 8000)
	binary.LittleEndian.PutUint32(data[28:32], 8000)
	binary.LittleEndian.PutUint16(data[32:34], 1)
	binary.LittleEndian.PutUint16(data[34:36], 8)
	copy(data[36:40], "data")
	binary.LittleEndian.PutUint32(data[40:44], 4)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := LoadWAVMono16(path)
	if err == nil {
		t.Fatal("expected unsupported format error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "PCM16") || !strings.Contains(msg, "float32") {
		t.Fatalf("error should list supported formats: %v", err)
	}
}

func TestLoadFixtureToneWAV(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "fixtures", "tone.wav")
	samples, rate, err := LoadWAVMono16(path)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Fatalf("rate=%d want 16000", rate)
	}
	if len(samples) != 16000 {
		t.Fatalf("len=%d want 16000", len(samples))
	}
}

func TestDownmixStereoAndMore(t *testing.T) {
	// 3-channel PCM16: samples (1, 3, 5) should average to 3/32768
	dir := t.TempDir()
	path := filepath.Join(dir, "3ch.wav")
	const nFrames = 2
	const ch = 3
	payload := nFrames * ch * 2
	data := make([]byte, 44+payload)
	copy(data[0:4], "RIFF")
	riff, err := intToUint32(36 + payload)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(data[4:8], riff)
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1)
	binary.LittleEndian.PutUint16(data[22:24], uint16(ch))
	binary.LittleEndian.PutUint32(data[24:28], 16000)
	binary.LittleEndian.PutUint32(data[28:32], 16000*6)
	binary.LittleEndian.PutUint16(data[32:34], 6)
	binary.LittleEndian.PutUint16(data[34:36], 16)
	copy(data[36:40], "data")
	udata, err := intToUint32(payload)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(data[40:44], udata)
	vals := []int16{1000, 2000, 3000, -1000, -2000, -3000}
	for i, v := range vals {
		binary.LittleEndian.PutUint16(data[44+i*2:], int16ToUint16(v))
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	samples, _, err := LoadWAVMono16(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("len=%d", len(samples))
	}
	if abs32(samples[0]-2000.0/32768) > 1e-6 {
		t.Fatalf("avg0=%g", samples[0])
	}
}

func writeFloat32WAV(path string, rate int, samples []float32) error {
	n := len(samples)
	dataBytes := n * 4
	riffSize, err := intToUint32(36 + dataBytes)
	if err != nil {
		return err
	}
	urate, err := intToUint32(rate)
	if err != nil {
		return err
	}
	byteRate, err := intToUint32(rate * 4)
	if err != nil {
		return err
	}
	udata, err := intToUint32(dataBytes)
	if err != nil {
		return err
	}
	buf := make([]byte, 44+dataBytes)
	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], riffSize)
	copy(buf[8:12], "WAVE")
	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], wavIEEEFloat)
	binary.LittleEndian.PutUint16(buf[22:24], 1)
	binary.LittleEndian.PutUint32(buf[24:28], urate)
	binary.LittleEndian.PutUint32(buf[28:32], byteRate)
	binary.LittleEndian.PutUint16(buf[32:34], 4)
	binary.LittleEndian.PutUint16(buf[34:36], 32)
	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], udata)
	for i, v := range samples {
		binary.LittleEndian.PutUint32(buf[44+i*4:], math.Float32bits(v))
	}
	return os.WriteFile(path, buf, 0o644)
}

func writeWAVWithRate(path string, rate uint32, samples int) error {
	if samples <= 0 {
		samples = 1
	}
	dataBytes := samples * 2
	riff, err := intToUint32(36 + dataBytes)
	if err != nil {
		return err
	}
	udata, err := intToUint32(dataBytes)
	if err != nil {
		return err
	}
	buf := make([]byte, 44+dataBytes)
	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], riff)
	copy(buf[8:12], "WAVE")
	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1)
	binary.LittleEndian.PutUint16(buf[22:24], 1)
	binary.LittleEndian.PutUint32(buf[24:28], rate)
	binary.LittleEndian.PutUint32(buf[28:32], rate*2)
	binary.LittleEndian.PutUint16(buf[32:34], 2)
	binary.LittleEndian.PutUint16(buf[34:36], 16)
	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], udata)
	return os.WriteFile(path, buf, 0o644)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
