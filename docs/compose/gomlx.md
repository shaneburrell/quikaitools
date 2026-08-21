# Compose: GoMLX

[GoMLX](https://github.com/gomlx/gomlx) is the closest thing to JAX/PyTorch for Go. Graph-mode (not eager). Same XLA engine as JAX when you use the `xla` backend.

Docs: [overview](https://gomlx.github.io/docs/overview/) · [backends](https://gomlx.github.io/docs/backends/) (updated through August 2026).

## When QuikAITools picks it

| Profile | Backend | Why |
|---------|---------|-----|
| V100 / CUDA | `xla:cuda` | Only first-class Go GPU **train** path |
| Halo | `go` (or `xla:cpu`) | No supported ROCm PJRT in the installer |
| Mac | `go` (or `xla:cpu`) | Darwin PJRT is CPU-only |
| CPU / CI | `go` | Pure Go, WASM-capable |

```bash
export GOMLX_BACKEND=xla:cuda          # V100
export GOMLX_BACKEND=go                # Halo, Mac portable, CI
export GOMLX_NO_AUTO_INSTALL=1         # air-gapped / Docker
```

Plugins cache under `~/.local/lib/go-xla` (Linux) or `~/Library/Application Support/go-xla` (Mac).

## Companions

- [go-huggingface](https://github.com/gomlx/go-huggingface) — Hub download, tokenizers, safetensors/GGUF weights
- [onnx-gomlx](https://github.com/gomlx/onnx-gomlx) — ONNX ↔ GoMLX (fine-tune, then write ONNX back)
- [go-xla](https://github.com/gomlx/go-xla) / [gopjrt](https://github.com/gomlx/gopjrt) — PJRT loader
- [go-darwinml](https://github.com/gomlx/go-darwinml) — alpha CoreML + MPSGraph

## Shape and speed notes

- XLA is **static shapes**. Pad batches to powers of two.
- Autodiff is gradients only (no Jacobian).
- Multi-GPU via XLA Shardy is experimental. See [../gaps/distributed.md](../gaps/distributed.md).
- Their line, which we keep: it is still only a slice of a major ML framework.

## Halo experiment (not supported)

JAX + ROCm 7.2.4 has been shown on Framework Desktop gfx1151. GoMLX can load an arbitrary plugin:

```bash
export GOMLX_BACKEND=xla:/path/to/rocm_pjrt.so
```

Nobody has published this combination. If you try it, write the result into [../gaps/backend.md](../gaps/backend.md).
