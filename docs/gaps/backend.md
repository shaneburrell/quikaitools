# Gap: a single backend fast on Mac + CUDA + Halo

This is the hardest item. We treat it as **two layers** so the README is not a lie.

## Layer A — one Go API (v0.1)

`internal/backend` is what applications call:

- `Detect()` → `v100` | `cuda` | `halo` | `mac` | `cpu`
- Fields: `TrainEngine`, `InferONNX`, `InferGGUF`, `MixedPrecision`
- `quikaitools doctor` prints the bind and warns on CPU fallback

Under the hood this is still three engines. That is how we get **fast on all three in 2026**.

## Layer B — one compiler (long pole)

GoMLX `compute.Backend` is the right long-term **train** API.

| Machine | Today | Next experiment |
|---------|-------|-----------------|
| V100 | XLA CUDA (done) | Stay here |
| Mac | XLA CPU / `go`; Metal via llama.cpp | Harden [go-darwinml](https://github.com/gomlx/go-darwinml) (CoreML + MPSGraph) |
| Halo | `go` / XLA CPU; infer via llama.cpp | Load AMD JAX ROCm PJRT: `GOMLX_BACKEND=xla:/path/to/plugin.so` on ROCm 7.2.4 + kernel 6.18.4+ |

JAX 0.8.2 + ROCm 7.2.4 has been verified on Framework Desktop gfx1151 ([framework-rocm](https://github.com/geoff-davis/framework-rocm)). **GoMLX + that PJRT is unpublished.** Try it, then update this page with commands and whether graphs actually run.

llama.cpp remains the portable **fast infer** path (CUDA / Metal / Vulkan / HIP). We will not beat it with a new Go GPU compiler.

## What “fast” means

- V100 train: GoMLX XLA CUDA. Infer: ORT CUDA or llama.cpp CUDA.
- Halo infer: Vulkan (stable/tg) or HIP + `ROCBLAS_USE_HIPBLASLT=1` + `HIP_NO_VMM` (prefill). Train: not fast on the iGPU in Go until Layer B works.
- Mac infer: Metal / CoreML. Train: Relux or go-darwinml, not XLA GPU.

## WebGPU / Born

[Born](https://github.com/born-ml/born) (WebGPU, ~56 ONNX ops) is a watch item for a portable GPU story. It is not Layer A in v0.1.
