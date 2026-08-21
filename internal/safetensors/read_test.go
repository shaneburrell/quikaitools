package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
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
	hdr, _ := json.Marshal(map[string]any{
		"x": map[string]any{"dtype": "F32", "shape": []int{1}, "data_offsets": []int{0, 4}},
	})
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
