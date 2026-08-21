// Package backend is the Layer A device API: one Go profile that binds the
// fast engine on each lab machine (V100, Strix Halo, Mac).
package backend

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

// Kind is a detected or requested lab profile.
type Kind string

const (
	KindAuto Kind = "auto"
	KindV100 Kind = "v100"
	KindCUDA Kind = "cuda"
	KindHalo Kind = "halo"
	KindMac  Kind = "mac"
	KindCPU  Kind = "cpu"
)

// AllKinds is the user-facing profile list (excluding auto).
var AllKinds = []Kind{KindV100, KindCUDA, KindHalo, KindMac, KindCPU}

// Profile is what trainers, pipelines, and doctor use.
type Profile struct {
	Kind           Kind   `json:"kind"`
	DisplayName    string `json:"display_name"`
	TrainEngine    string `json:"train_engine"`
	InferONNX      string `json:"infer_onnx"`
	InferGGUF      string `json:"infer_gguf"`
	MixedPrecision string `json:"mixed_precision"`
	Notes          string `json:"notes"`
	DetectedFrom   string `json:"detected_from,omitempty"`
	HardwareHint   string `json:"hardware_hint,omitempty"`
}

// HostInfo is the probe result. Tests inject this instead of talking to the OS.
type HostInfo struct {
	GOOS       string
	NvidiaName string
	HasKFD     bool
	ROCmGFX    string
}

// ProbeFunc gathers HostInfo. Replace in tests.
type ProbeFunc func() HostInfo

// DefaultProbe inspects this process and common device nodes/tools.
func DefaultProbe() HostInfo {
	info := HostInfo{GOOS: runtime.GOOS}
	info.NvidiaName = firstLine(runLookup("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"))
	info.HasKFD = fileExists("/dev/kfd")
	info.ROCmGFX = detectGFX(runLookup("rocminfo"))
	return info
}

// Detect resolves a profile. kind may be auto or an explicit Kind.
func Detect(kind Kind, probe ProbeFunc) (Profile, error) {
	if probe == nil {
		probe = DefaultProbe
	}
	if kind == "" {
		kind = KindAuto
	}
	if env := strings.TrimSpace(os.Getenv("QUIKAITOOLS_PROFILE")); kind == KindAuto && env != "" {
		kind = Kind(strings.ToLower(env))
	}

	host := probe()
	resolved := kind
	from := "flag"
	if kind == KindAuto {
		resolved = inferKind(host)
		from = "probe"
	}

	p, err := profileFor(resolved)
	if err != nil {
		return Profile{}, err
	}
	p.DetectedFrom = from
	p.HardwareHint = hardwareHint(host)
	return p, nil
}

func inferKind(h HostInfo) Kind {
	if h.GOOS == "darwin" {
		return KindMac
	}
	name := strings.ToUpper(h.NvidiaName)
	if strings.Contains(name, "V100") {
		return KindV100
	}
	if strings.TrimSpace(h.NvidiaName) != "" {
		return KindCUDA
	}
	gfx := strings.ToLower(h.ROCmGFX)
	if h.HasKFD || strings.Contains(gfx, "gfx1151") || strings.Contains(gfx, "strix") {
		return KindHalo
	}
	return KindCPU
}

func profileFor(kind Kind) (Profile, error) {
	switch kind {
	case KindV100:
		return Profile{
			Kind:           KindV100,
			DisplayName:    "NVIDIA V100",
			TrainEngine:    "gomlx-xla-cuda",
			InferONNX:      "hugot-ort-cuda",
			InferGGUF:      "llamacpp-cuda",
			MixedPrecision: "fp16",
			Notes:          "Volta has FP16 tensor cores, not BF16. Prefer LoRA/QLoRA on 16–32 GB HBM2.",
		}, nil
	case KindCUDA:
		return Profile{
			Kind:           KindCUDA,
			DisplayName:    "NVIDIA CUDA (non-V100)",
			TrainEngine:    "gomlx-xla-cuda",
			InferONNX:      "hugot-ort-cuda",
			InferGGUF:      "llamacpp-cuda",
			MixedPrecision: "fp16",
			Notes:          "Same Go engines as V100. Ampere+ may use BF16 later; this API still defaults to FP16.",
		}, nil
	case KindHalo:
		return Profile{
			Kind:           KindHalo,
			DisplayName:    "AMD Strix Halo (gfx1151)",
			TrainEngine:    "gomlx-go",
			InferONNX:      "hugot-go",
			InferGGUF:      "llamacpp-vulkan",
			MixedPrecision: "none",
			Notes:          "Fast path is GGUF via llama.cpp Vulkan (stable) or HIP (tuned prefill). Go train on the iGPU is not supported yet.",
		}, nil
	case KindMac:
		return Profile{
			Kind:           KindMac,
			DisplayName:    "Apple Silicon",
			TrainEngine:    "gomlx-go",
			InferONNX:      "hugot-coreml",
			InferGGUF:      "llamacpp-metal",
			MixedPrecision: "none",
			Notes:          "XLA on macOS is CPU-only. GPU train is Relux or go-darwinml (alpha).",
		}, nil
	case KindCPU:
		return Profile{
			Kind:           KindCPU,
			DisplayName:    "CPU (portable Go)",
			TrainEngine:    "gomlx-go",
			InferONNX:      "hugot-go",
			InferGGUF:      "llamacpp-cpu",
			MixedPrecision: "none",
			Notes:          "Portable fallback for CI and machines with no GPU SDK.",
		}, nil
	default:
		return Profile{}, fmt.Errorf("unknown profile %q (want auto, v100, cuda, halo, mac, cpu)", kind)
	}
}

func hardwareHint(h HostInfo) string {
	var parts []string
	if h.NvidiaName != "" {
		parts = append(parts, "nvidia="+h.NvidiaName)
	}
	if h.HasKFD {
		parts = append(parts, "kfd=true")
	}
	if h.ROCmGFX != "" {
		parts = append(parts, "gfx="+h.ROCmGFX)
	}
	if h.GOOS != "" {
		parts = append(parts, "goos="+h.GOOS)
	}
	return strings.Join(parts, " ")
}

func (p Profile) String() string {
	return fmt.Sprintf("%s (%s)", p.DisplayName, p.Kind)
}

// CatalogMachine maps a profile to the YAML catalog column.
func (p Profile) CatalogMachine() string {
	switch p.Kind {
	case KindCUDA:
		return string(KindV100)
	default:
		return string(p.Kind)
	}
}
