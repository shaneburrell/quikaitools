# Compose: llama.cpp from Go

llama.cpp is the **portable fast infer** engine. QuikAITools does not replace it. We bind it and ship the right **per-arch** library.

## Bindings that still track upstream

| Repo | Notes |
|------|-------|
| [tcpipuk/llama-go](https://github.com/tcpipuk/go-llama.cpp) | CGO fork of dead go-skynet; CUDA/ROCm/Metal/Vulkan/SYCL; CI includes CUDA |
| [fuguohong1024/gollama.cpp](https://github.com/fuguohong1024/gollama.cpp) | No CGO (`purego`); ships platform libs |

Do not use [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp) (idle since October 2023).

## Backend per lab machine

| Machine | llama.cpp backend | How to think about it |
|---------|-------------------|------------------------|
| V100 | CUDA | `-ngl 999`. Boring, good. |
| Mac | Metal | Default on Apple Silicon. |
| Halo | Vulkan RADV **or** HIP gfx1151 | See [../lab/halo.md](../lab/halo.md). You must **build** a Halo-tuned `.so`. A CUDA wheel will not work. |

QuikAITools Layer A reports `llamacpp-cuda`, `llamacpp-metal`, `llamacpp-vulkan` (Halo default). HIP is an operator choice on Halo, not a second profile.

## Go integration rule

The Go module talks to `libllama` / `libggml`. The **build of that library** is a lab artifact:

- V100 image: `GGML_CUDA=ON`
- Mac: Metal enabled
- Halo: either `GGML_VULKAN=ON` or the HIP flags in the Halo lab doc

`quikaitools doctor` cannot compile kernels. It can only tell you which name to bind.

## Serving

For a full OpenAI-compatible server, [LocalAI](https://localai.io/) is a Go host that pulls backends as OCI images. Fine if the product is “serve models.” This repo stays a compose + gap-fill toolset, not a second LocalAI.
