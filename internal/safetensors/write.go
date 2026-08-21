package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// WriteFile writes F32 tensors in safetensors format.
func WriteFile(path string, tensors map[string]Tensor) error {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := map[string]headerEntry{}
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
	out := make([]byte, 8+len(hb)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hb)))
	copy(out[8:], hb)
	copy(out[8+len(hb):], payload)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return fmt.Errorf("safetensors write: %w", err)
	}
	return nil
}
