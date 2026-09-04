package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseRoundTrip(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	hdrMap := map[string]any{
		"w": map[string]any{
			"dtype":        "F32",
			"shape":        []int{2, 3},
			"data_offsets": []int{0, len(raw)},
		},
	}
	hdr, err := json.Marshal(hdrMap)
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8+len(hdr)+len(raw))
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(hdr)))
	copy(blob[8:], hdr)
	copy(blob[8+len(hdr):], raw)
	got, err := Parse(blob)
	if err != nil {
		t.Fatal(err)
	}
	w := got["w"]
	if len(w.Data) != 6 || w.Shape[0] != 2 || w.Data[5] != 6 {
		t.Fatalf("%+v", w)
	}
}

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, math.Float32bits(9))
	hdr, err := json.Marshal(map[string]any{
		"x": map[string]any{"dtype": "F32", "shape": []int{1}, "data_offsets": []int{0, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8+len(hdr)+4)
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(hdr)))
	copy(blob[8:], hdr)
	copy(blob[8+len(hdr):], raw)
	path := filepath.Join(dir, "m.safetensors")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if m["x"].Data[0] != 9 {
		t.Fatalf("%v", m["x"])
	}
}

func TestParseMalformed(t *testing.T) {
	f32 := make([]byte, 4)
	binary.LittleEndian.PutUint32(f32, math.Float32bits(1))
	validHdr, err := json.Marshal(map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int{1}, "data_offsets": []int{0, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		blob []byte
	}{
		{"truncated header length", []byte{1, 2, 3}},
		{"header longer than file", func() []byte {
			b := make([]byte, 12)
			binary.LittleEndian.PutUint64(b[:8], 1000)
			return b
		}()},
		{"bad JSON", func() []byte {
			h := []byte("{not json")
			b := make([]byte, 8+len(h))
			binary.LittleEndian.PutUint64(b[:8], uint64(len(h)))
			copy(b[8:], h)
			return b
		}()},
		{"offsets out of range", func() []byte {
			h, err := json.Marshal(map[string]any{
				"w": map[string]any{"dtype": "F32", "shape": []int{1}, "data_offsets": []int{0, 99}},
			})
			if err != nil {
				t.Fatal(err)
			}
			b := make([]byte, 8+len(h)+4)
			binary.LittleEndian.PutUint64(b[:8], uint64(len(h)))
			copy(b[8:], h)
			copy(b[8+len(h):], f32)
			return b
		}()},
		{"shape mismatch", func() []byte {
			h, err := json.Marshal(map[string]any{
				"w": map[string]any{"dtype": "F32", "shape": []int{2, 2}, "data_offsets": []int{0, 4}},
			})
			if err != nil {
				t.Fatal(err)
			}
			b := make([]byte, 8+len(h)+4)
			binary.LittleEndian.PutUint64(b[:8], uint64(len(h)))
			copy(b[8:], h)
			copy(b[8+len(h):], f32)
			return b
		}()},
		{"unknown dtype", func() []byte {
			h, err := json.Marshal(map[string]any{
				"w": map[string]any{"dtype": "I64", "shape": []int{1}, "data_offsets": []int{0, 4}},
			})
			if err != nil {
				t.Fatal(err)
			}
			b := make([]byte, 8+len(h)+4)
			binary.LittleEndian.PutUint64(b[:8], uint64(len(h)))
			copy(b[8:], h)
			copy(b[8+len(h):], f32)
			return b
		}()},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse(tc.blob)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
	// Sanity: a well-formed blob still parses.
	ok := make([]byte, 8+len(validHdr)+4)
	binary.LittleEndian.PutUint64(ok[:8], uint64(len(validHdr)))
	copy(ok[8:], validHdr)
	copy(ok[8+len(validHdr):], f32)
	if _, err := Parse(ok); err != nil {
		t.Fatalf("valid blob: %v", err)
	}
}

func TestParseLowercaseDtype(t *testing.T) {
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, math.Float32bits(3.5))
	hdr, err := json.Marshal(map[string]any{
		"w": map[string]any{"dtype": "f32", "shape": []int{1}, "data_offsets": []int{0, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8+len(hdr)+4)
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(hdr)))
	copy(blob[8:], hdr)
	copy(blob[8+len(hdr):], raw)
	got, err := Parse(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got["w"].Dtype != "F32" || got["w"].Data[0] != 3.5 {
		t.Fatalf("%+v", got["w"])
	}
}

func TestFloat16Golden(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
		nan  bool
		inf  int
	}{
		{"1.0", 0x3c00, 1, false, 0},
		{"-2.5", 0xc100, -2.5, false, 0},
		{"65504", 0x7bff, 65504, false, 0},
		{"smallest subnormal", 0x0001, float32(math.Ldexp(1, -24)), false, 0},
		{"+inf", 0x7c00, 0, false, 1},
		{"nan", 0x7e00, 0, true, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var raw [2]byte
			binary.LittleEndian.PutUint16(raw[:], tc.bits)
			got, err := Float32("F16", raw[:])
			if err != nil {
				t.Fatal(err)
			}
			checkConverted(t, got[0], tc.want, tc.nan, tc.inf)
			got2, err := Float32("f16", raw[:])
			if err != nil {
				t.Fatal(err)
			}
			checkConverted(t, got2[0], tc.want, tc.nan, tc.inf)
		})
	}
}

func TestBFloat16Golden(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
		nan  bool
		inf  int
	}{
		{"1.0", 0x3f80, 1, false, 0},
		{"-2.5", 0xc020, -2.5, false, 0},
		{"65504", 0x477f, math.Float32frombits(0x477f0000), false, 0},
		{"smallest subnormal", 0x0001, math.Float32frombits(0x00010000), false, 0},
		{"+inf", 0x7f80, 0, false, 1},
		{"nan", 0x7fc0, 0, true, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var raw [2]byte
			binary.LittleEndian.PutUint16(raw[:], tc.bits)
			got, err := Float32("BF16", raw[:])
			if err != nil {
				t.Fatal(err)
			}
			checkConverted(t, got[0], tc.want, tc.nan, tc.inf)
			got2, err := Float32("bf16", raw[:])
			if err != nil {
				t.Fatal(err)
			}
			checkConverted(t, got2[0], tc.want, tc.nan, tc.inf)
		})
	}
}

func TestParseF16BF16(t *testing.T) {
	var raw [2]byte
	binary.LittleEndian.PutUint16(raw[:], 0x3c00) // F16 1.0
	hdr, err := json.Marshal(map[string]any{
		"w": map[string]any{"dtype": "F16", "shape": []int{1}, "data_offsets": []int{0, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8+len(hdr)+2)
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(hdr)))
	copy(blob[8:], hdr)
	copy(blob[8+len(hdr):], raw[:])
	got, err := Parse(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got["w"].Data[0] != 1 || got["w"].Dtype != "F16" {
		t.Fatalf("%+v", got["w"])
	}

	binary.LittleEndian.PutUint16(raw[:], 0x3f80) // BF16 1.0
	hdr, err = json.Marshal(map[string]any{
		"w": map[string]any{"dtype": "bf16", "shape": []int{1}, "data_offsets": []int{0, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	blob = make([]byte, 8+len(hdr)+2)
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(hdr)))
	copy(blob[8:], hdr)
	copy(blob[8+len(hdr):], raw[:])
	got, err = Parse(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got["w"].Data[0] != 1 || got["w"].Dtype != "BF16" {
		t.Fatalf("%+v", got["w"])
	}
}

func TestWriteReadRoundTripMetaAlign(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.safetensors")
	in := map[string]Tensor{
		"a": {Name: "a", Dtype: "F32", Shape: []int{2, 2}, Data: []float32{1, -2, 3.5, 0}},
	}
	meta := map[string]string{"format": "pt"}
	if err := WriteFileMeta(path, in, meta); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if (8+n)%8 != 0 {
		t.Fatalf("data start %d not 8-byte aligned", 8+n)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	var md map[string]string
	if err := json.Unmarshal(hdr["__metadata__"], &md); err != nil {
		t.Fatal(err)
	}
	if md["format"] != "pt" {
		t.Fatalf("metadata=%v", md)
	}
	got, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got["a"].Data) != 4 || got["a"].Data[2] != 3.5 {
		t.Fatalf("%+v", got["a"])
	}
}

func TestWriteFilePadsHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "p.safetensors")
	if err := WriteFile(path, map[string]Tensor{
		"x": {Name: "x", Shape: []int{1}, Data: []float32{1}},
	}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if n%8 != 0 {
		t.Fatalf("header len %d not multiple of 8", n)
	}
	if !strings.Contains(string(raw[8:8+n]), "x") {
		t.Fatal("header missing tensor")
	}
}

func TestFloat32Unknown(t *testing.T) {
	if _, err := Float32("I32", make([]byte, 4)); err == nil {
		t.Fatal("expected error")
	}
}

func checkConverted(t *testing.T, got, want float32, nan bool, inf int) {
	t.Helper()
	if nan {
		if !math.IsNaN(float64(got)) {
			t.Fatalf("got %v want NaN", got)
		}
		return
	}
	if inf != 0 {
		if !math.IsInf(float64(got), inf) {
			t.Fatalf("got %v want inf(%d)", got, inf)
		}
		return
	}
	if got != want {
		t.Fatalf("got %v (%x) want %v (%x)", got, math.Float32bits(got), want, math.Float32bits(want))
	}
}
