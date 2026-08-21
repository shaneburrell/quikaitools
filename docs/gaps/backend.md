# Gap: a single backend fast on Mac + CUDA + Halo

**Status:** **Layer A MVP shipped.** Applications call one profile; engines still differ per box.

## Layer A — one Go API (MVP)

`internal/backend` + `internal/infer`:

- `Detect()` → `v100` | `cuda` | `halo` | `mac` | `cpu`
- `quikaitools doctor [--strict]` — strict checks for llama.cpp binary and GPU probes
- `generate --gguf` → llama.cpp exec wrapper (Metal / CUDA / Vulkan / CPU by profile)
- `embed` / `transcribe` → preprocess + ONNX stub until Hugot/ORT is linked in-process

Under the hood this is still three engines. That is how we get **fast on all three in 2026**.

## Layer B — one compiler (post-MVP)

GoMLX `compute.Backend` is the long-term **train** API (XLA CUDA on V100; go-darwinml / ROCm PJRT experiments). Not an MVP blocker.

## What “fast” means

- V100 train: GoMLX XLA CUDA (Layer B). Infer: ORT CUDA or llama.cpp CUDA.
- Halo infer: Vulkan (stable) or HIP. Train: not fast on the iGPU in Go until Layer B.
- Mac infer: Metal / CoreML. Train: portable Go LoRA/QLoRA in MVP.

See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).
