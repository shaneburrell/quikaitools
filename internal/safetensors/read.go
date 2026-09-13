// Package safetensors reads Hugging Face .safetensors (F32/F16/BF16 little-endian).
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// Tensor is a named dense float32 array (F16/BF16 payloads are converted on read).
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

// LoadFile reads all F32/F16/BF16 tensors from path (converted to float32).
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
	if n > uint64(len(b)-8) {
		return nil, fmt.Errorf("safetensors: header longer than file")
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
		dt, ok := canonicalDtype(e.Dtype)
		if !ok {
			return nil, fmt.Errorf("%s: unsupported dtype %s (want F32, F16, or BF16)", name, e.Dtype)
		}
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start < 0 || end < start || end > len(payload) {
			return nil, fmt.Errorf("%s: bad offsets [%d,%d]", name, start, end)
		}
		rawT := payload[start:end]
		nval, err := numel(e.Shape)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		sz := dtypeBytes(dt)
		if nval > 0 && sz > math.MaxInt/nval {
			return nil, fmt.Errorf("%s: shape %v overflow", name, e.Shape)
		}
		want := nval * sz
		if want != len(rawT) {
			return nil, fmt.Errorf("%s: shape %v dtype %s wants %d bytes got %d", name, e.Shape, dt, want, len(rawT))
		}
		data, err := Float32(dt, rawT)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		out[name] = Tensor{Name: name, Dtype: dt, Shape: e.Shape, Data: data}
	}
	return out, nil
}

// Float32 decodes little-endian tensor bytes as float32.
// dtype may be F32, F16, or BF16 (any case).
func Float32(dtype string, raw []byte) ([]float32, error) {
	dt, ok := canonicalDtype(dtype)
	if !ok {
		return nil, fmt.Errorf("safetensors: unsupported dtype %s", dtype)
	}
	sz := dtypeBytes(dt)
	if sz == 0 || len(raw)%sz != 0 {
		return nil, fmt.Errorf("safetensors: %s payload length %d not aligned", dt, len(raw))
	}
	n := len(raw) / sz
	out := make([]float32, n)
	switch dt {
	case "F32":
		for i := range out {
			out[i] = float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		for i := range out {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		for i := range out {
			out[i] = bfloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	}
	return out, nil
}

func canonicalDtype(dt string) (string, bool) {
	switch strings.ToUpper(dt) {
	case "F32", "F16", "BF16":
		return strings.ToUpper(dt), true
	default:
		return "", false
	}
}

func dtypeBytes(dt string) int {
	switch dt {
	case "F32":
		return 4
	case "F16", "BF16":
		return 2
	default:
		return 0
	}
}

func numel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 1, nil
	}
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("safetensors: negative dimension %d", d)
		}
		if d > 0 && n > math.MaxInt/d {
			return 0, fmt.Errorf("safetensors: shape overflow")
		}
		n *= d
	}
	return n, nil
}

func float32frombits(u uint32) float32 {
	return float32frombitsImpl(u)
}
