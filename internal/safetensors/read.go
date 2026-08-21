// Package safetensors reads Hugging Face .safetensors (F32 little-endian).
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// Tensor is a named dense float32 array.
type Tensor struct {
	Name  string
	Dtype string
	Shape []int
	Data  []float32
}

type headerEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// LoadFile reads all F32 tensors from path.
func LoadFile(path string) (map[string]Tensor, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Parse decodes a safetensors blob.
func Parse(b []byte) (map[string]Tensor, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: short file")
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if uint64(len(b)) < 8+n {
		return nil, fmt.Errorf("safetensors: truncated header")
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &hdr); err != nil {
		return nil, fmt.Errorf("safetensors header: %w", err)
	}
	payload := b[8+n:]
	out := make(map[string]Tensor, len(hdr))
	for name, raw := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if e.Dtype != "F32" && e.Dtype != "f32" {
			return nil, fmt.Errorf("%s: unsupported dtype %s (want F32)", name, e.Dtype)
		}
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start < 0 || end < start || end > len(payload) {
			return nil, fmt.Errorf("%s: bad offsets [%d,%d]", name, start, end)
		}
		rawT := payload[start:end]
		if len(rawT)%4 != 0 {
			return nil, fmt.Errorf("%s: not float32 aligned", name)
		}
		data := make([]float32, len(rawT)/4)
		for i := range data {
			data[i] = float32frombits(binary.LittleEndian.Uint32(rawT[i*4:]))
		}
		if numel(e.Shape) != len(data) {
			return nil, fmt.Errorf("%s: shape %v wants %d got %d", name, e.Shape, numel(e.Shape), len(data))
		}
		out[name] = Tensor{Name: name, Dtype: "F32", Shape: e.Shape, Data: data}
	}
	return out, nil
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(shape) == 0 {
		return 0
	}
	return n
}

func float32frombits(u uint32) float32 {
	return float32frombitsImpl(u)
}
