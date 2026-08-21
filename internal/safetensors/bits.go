package safetensors

import "math"

func float32frombitsImpl(u uint32) float32 {
	return math.Float32frombits(u)
}
