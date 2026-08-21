# Compose: Hugot

[Hugot](https://github.com/knights-analytics/hugot) is Hugging Face **pipelines** for Go: download ONNX, tokenize, infer. Knights Analytics runs it in production. Jan Pfeifer (GoMLX) is an advisor — treat Hugot and GoMLX as one platform, not rivals.

## Sessions

| Constructor | Build tag | Use on |
|-------------|-----------|--------|
| `NewGoSession` | default | Halo ONNX, CI, no CGO |
| `NewORTSession` | `-tags ORT` | V100 CUDA, Mac CoreML, fastest CPU ONNX |
| `NewXLASession` | `-tags XLA` | Fine-tune (ONNX → XLA → GoMLX → ONNX). Not generation. |

**Tested accelerators:** CPU, TPU, NVIDIA CUDA.

**Advertised, less tested:** TensorRT, DirectML, **CoreML**, OpenVINO.

ORT **removed the ROCm execution provider in 1.23**. Do not plan Hugot-on-Halo-GPU.

## V100 CUDA sketch

Need the GPU build of ONNX Runtime, CUDA 12.x, cuDNN 9.x:

```go
session, err := hugot.NewORTSession(ctx, options.WithCuda(map[string]string{
    "device_id": "0",
}))
```

## What we use it for

- Catalog tasks `embed`, `classify`, `token-classify`, `rerank`
- First smoke test: DistilBERT SST-2 (see `catalog/models/distilbert-sst2.yaml`)
- Fine-tune of small ONNX transformers once LoRA exists — Hugot already converts through GoMLX XLA

Generation pipelines: ORT only. Prefer llama.cpp GGUF for lab LLMs ([llamacpp.md](llamacpp.md)).
