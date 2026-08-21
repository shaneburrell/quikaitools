# Lab profile: Apple Silicon

Unified memory, Metal, CoreML / ANE. **XLA on Darwin is CPU-only.** There is no Metal PJRT in the GoMLX installer. GPU work has to come from llama.cpp, ORT CoreML, Relux, or go-darwinml.

## What is fast

| Job | Engine | Notes |
|-----|--------|-------|
| GGUF generate | llama.cpp **Metal** | Default local LLM path. |
| ONNX transformers | Hugot **CoreML** | Available; less battle-tested than CUDA. |
| Portable / CI | Hugot Go session, GoMLX `go` | No CGO, slower. |
| Train on GPU | Relux (AMX/MPS) or **go-darwinml** (alpha) | Not XLA. |

## Install

1. Xcode CLT (Metal). Full Xcode if you compile CoreML models (`coremlcompiler`).
2. Go 1.27+.
3. llama.cpp with Metal (default on Apple Silicon when the toolchain is present).
4. Confirm:

```bash
quikaitools doctor --profile auto
# expect: kind mac, infer_gguf llamacpp-metal, infer_onnx hugot-coreml
```

GoMLX `GOMLX_BACKEND=xla:cpu` works and leaves the GPU idle. That is expected.

## Training honesty

- Small graphs: GoMLX `go` or XLA CPU.
- Decoder-only experiments: [Relux](https://github.com/xDarkicex/relux) talks to AMX/MPS without CGO.
- Long-term GoMLX GPU: [go-darwinml](https://github.com/gomlx/go-darwinml) (CoreML + MPSGraph). Alpha; last published module March 2026. Track it in [../gaps/backend.md](../gaps/backend.md).

## Doctor contract

`GOOS=darwin` always resolves `auto` to `mac`, even if you also have a remote CUDA box. Use `--profile v100` when editing V100 docs on a laptop.

```bash
quikaitools doctor --strict
# expects llama-cli (or QUIKAITOOLS_LLAMA) when exercising GGUF generate
```

## Mac e2e (MVP)

Proven on this machine for the portable Go paths (Hub pull + LoRA/QLoRA + generate adapter + vision/audio preprocess). GGUF Metal needs a local `llama-cli`.

```bash
make build
./bin/quikaitools pull tiny-random-gpt2
./bin/quikaitools train lora \
  --model ~/.cache/quikaitools/models/hf-internal-testing/tiny-random-gpt2 \
  --data testdata/fixtures/stories.txt --steps 25 --rank 4 --accum 2 \
  --out testdata/artifacts/adapter-lora --profile mac
./bin/quikaitools generate --model ~/.cache/quikaitools/models/hf-internal-testing/tiny-random-gpt2 \
  --adapter testdata/artifacts/adapter-lora --prompt "Once upon" --tokens 8
./bin/quikaitools train qlora --model … --data … --steps 10 --out testdata/artifacts/adapter-qlora
./bin/quikaitools embed --vision --model . --image testdata/fixtures/red.png
./bin/quikaitools transcribe --model . --audio testdata/fixtures/tone.wav
```

Full checklist: [mvp-smoke.md](mvp-smoke.md).

## See also

- [../compose/hugot.md](../compose/hugot.md)
- [../compose/llamacpp.md](../compose/llamacpp.md)
- [../gaps/backend.md](../gaps/backend.md)
