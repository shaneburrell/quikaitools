package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// WriteFile writes F32 tensors in safetensors format.
// The JSON header is padded with spaces to an 8-byte boundary.
func WriteFile(path string, tensors map[string]Tensor) error {
	return WriteFileMeta(path, tensors, nil)
}

// WriteFileMeta is WriteFile plus optional __metadata__ (e.g. {"format":"pt"}).
// The JSON header is padded with spaces so the tensor payload starts at a multiple of 8.
func WriteFileMeta(path string, tensors map[string]Tensor, meta map[string]string) error {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := make(map[string]any, len(names)+1)
	if len(meta) > 0 {
		hdr["__metadata__"] = meta
	}
	var payload []byte
	off := 0
	for _, name := range names {
		t := tensors[name]
		n := len(t.Data) * 4
		hdr[name] = headerEntry{Dtype: "F32", Shape: t.Shape, DataOffsets: [2]int{off, off + n}}
		for _, v := range t.Data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			payload = append(payload, buf[:]...)
		}
		off += n
	}
	hb, err := json.Marshal(hdr)
	if err != nil {
		return err
	}
	if pad := (8 - len(hb)%8) % 8; pad > 0 {
		hb = append(hb, bytes.Repeat([]byte{' '}, pad)...)
	}
	out := make([]byte, 8+len(hb)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hb)))
	copy(out[8:], hb)
	copy(out[8+len(hb):], payload)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return fmt.Errorf("safetensors write: %w", err)
	}
	return nil
}
