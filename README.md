# QuikAITools

**A Go-native ML lab guide and the missing PyTorch-class tooling** — compose what already works, fill what does not, on V100, Strix Halo, and Mac.

[![CI](https://github.com/shaneburrell/quikaitools/actions/workflows/ci.yml/badge.svg)](https://github.com/shaneburrell/quikaitools/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Reference](https://pkg.go.dev/badge/github.com/shaneburrell/quikaitools.svg)](https://pkg.go.dev/github.com/shaneburrell/quikaitools)

QuikAITools is not another tensor core and not a Python wrapper. It is a **public map** of maintained Go ML projects, a **device profile** that picks the fast engine on each lab machine, and the place we implement the gaps PyTorch still owns: LoRA/TRL-lite, a curated model zoo, vision/audio slices, Accelerate-lite, and a single backend API.

```bash
quikaitools doctor [--strict]
quikaitools catalog --machine halo --task generate
quikaitools pull tiny-random-gpt2
quikaitools train lora --model ~/.cache/quikaitools/models/hf-internal-testing/tiny-random-gpt2 \
  --data testdata/fixtures/stories.txt --steps 30 --rank 4 --accum 2 --profile mac \
  --seq 32 --seed 1 --out testdata/artifacts/adapter-lora
quikaitools generate --model … --adapter testdata/artifacts/adapter-lora --prompt "Once" \
  --temperature 0.8 --top-k 40 --top-p 0.9 --seed 1 --stop-eos
quikaitools generate --gguf FILE --prompt "Hi" --tokens 16 --timeout 30s
quikaitools train qlora --model … --data … --steps 20 --mask-prompt
quikaitools embed --model … --text "hello"              # --allow-stub=false fails closed on ONNX stubs
quikaitools embed --vision --model … --image testdata/fixtures/red.png
quikaitools transcribe --model … --audio testdata/fixtures/tone.wav
quikaitools export merge --model DIR --adapter DIR --out DIR
quikaitools export gguf --model DIR --out FILE.gguf
quikaitools export modelfile --gguf FILE --out Modelfile   # --force to overwrite
quikaitools validate-sft --path FILE.jsonl
quikaitools version          # also --version
```

`catalog --task` takes catalog task ids (`generate`, `embed`, `asr`, …). `transcribe` is accepted as an alias for `asr`.

Smoke checklist: [docs/lab/mvp-smoke.md](docs/lab/mvp-smoke.md) (Mac filled; Halo/V100 when those boxes are on).

## Why QuikAITools?

| PyTorch-shaped problem | What this repo does |
|------------------------|---------------------|
| “Which Go project is actually maintained?” | A living compose guide: GoMLX, Hugot, llama.cpp — not Gorgonia or libtorch |
| Three machines, three stacks | One `doctor` profile: V100, Strix Halo (gfx1151), Apple Silicon |
| No PEFT / LoRA / TRL in Go | `train lora` / `train qlora` + `generate --adapter` on tiny GPT-2 (Mac e2e) |
| Hugging Face zoo vs a handful of Go ports | `pull <catalog-id>` + [catalog](catalog/models) with per-box status |
| torchvision / torchaudio | `embed --vision` + `transcribe` via `internal/vision` / `internal/audio` |
| Accelerate / DeepSpeed / FSDP | `internal/dist`: accum, checkpoint, `--resume`, `--profile` (FSDP later) |
| One backend fast on Mac + CUDA + Halo | Layer A: `doctor`/`generate`/`embed` bind engines. Layer B: compilers per box |

This is **honest software**. Halo GPU training in Go is an experiment. Unsloth will not be ported. DeepSpeed will not be rewritten.

## Lab machines

| Profile | Hardware | Fast infer | Fast train (today) |
|---------|----------|------------|--------------------|
| `v100` | NVIDIA V100 (16/32 GB HBM2) | Hugot ORT CUDA, llama.cpp CUDA | GoMLX XLA CUDA, **FP16 not BF16** |
| `halo` | AMD Strix Halo / gfx1151 | llama.cpp Vulkan or tuned HIP | GoMLX CPU/`go` until a ROCm PJRT is proven |
| `mac` | Apple Silicon | llama.cpp Metal, Hugot CoreML | Relux or go-darwinml (alpha); XLA is CPU-only |
| `cpu` | Anything | Hugot Go session, llama.cpp CPU | GoMLX `go` backend |

Read the machine notes:

- [docs/lab/v100.md](docs/lab/v100.md)
- [docs/lab/halo.md](docs/lab/halo.md)
- [docs/lab/mac.md](docs/lab/mac.md)
- [docs/lab/mvp-smoke.md](docs/lab/mvp-smoke.md) — three-machine MVP checklist

## Install

Requires [Go 1.27+](https://go.dev/dl/).

```bash
go install github.com/shaneburrell/quikaitools/cmd/quikaitools@latest
```

Or clone and build:

```bash
git clone https://github.com/shaneburrell/quikaitools.git
cd quikaitools
make build
./bin/quikaitools version
```

## Quick start

```bash
# What engine should this box use?
quikaitools doctor
quikaitools doctor --profile v100

# Models that belong on this machine (or a named one)
quikaitools catalog
quikaitools catalog --machine halo --task generate
quikaitools catalog --task embed --catalog ./catalog

# Override detection
export QUIKAITOOLS_PROFILE=halo
```

### Environment

| Variable | Read by | Purpose |
|----------|---------|---------|
| `QUIKAITOOLS_PROFILE` | `doctor` / `backend.Detect` | Force `v100` / `halo` / `mac` / `cpu` when `--profile` is auto |
| `QUIKAITOOLS_CACHE` | `pull` / `hub` | Weight cache (default `~/.cache/quikaitools`) |
| `QUIKAITOOLS_LLAMA` | `generate --gguf` | Path to `llama-cli` (or other llama.cpp binary) |

`doctor` must fail loudly in your head if a V100 box prints `train: gomlx-go` — that means you landed on CPU. Install CUDA 12 and the GoMLX XLA plugin, then re-run.

## How it works

```text
quikaitools doctor ──► backend.Detect ──► Profile (v100 | halo | mac | cpu)
                                              │
catalog/*.yaml ──► zoo.LoadDir / zoo.LoadFS ──┤ per-machine status + engine labels
                                              ▼
              Train  → portable Go LoRA/QLoRA (GoMLX XLA = Layer B)
              ONNX   → preprocess + stub until Hugot/ORT linked
              GGUF   → llama.cpp exec (CUDA / Vulkan|HIP / Metal) when binary present
```

**Layer A** (this repo, now): one Go API and profile strings. Applications do not name engines in app code — but many engines are still unbound (see `doctor` notes).

**Layer B** (long pole): make GoMLX’s `compute.Backend` actually fast on Mac GPU and Halo iGPU. Documented in [docs/gaps/backend.md](docs/gaps/backend.md).

## Compose, don’t rewrite

Use these maintained projects. QuikAITools ties them together.

| Piece | Project | Role |
|-------|---------|------|
| Train / graphs | [GoMLX](https://github.com/gomlx/gomlx) | JAX/XLA-class framework for Go |
| HF pipelines | [Hugot](https://github.com/knights-analytics/hugot) | ONNX transformers, some fine-tune |
| Hub + tokenizers | [go-huggingface](https://github.com/gomlx/go-huggingface) | Download + tokenize without Python |
| Local LLMs | [llama.cpp](https://github.com/ggml-org/llama.cpp) via [llama-go](https://github.com/tcpipuk/go-llama.cpp) or [gollama.cpp](https://github.com/fuguohong1024/gollama.cpp) | GGUF on all three GPUs |
| Halo ops | [kyuz0 toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) | Known-good Vulkan/ROCm containers |

Do not start from Gorgonia, unmaintained `go-skynet/go-llama.cpp`, or gotch/libtorch.

Guides: [docs/compose/gomlx.md](docs/compose/gomlx.md) · [docs/compose/hugot.md](docs/compose/hugot.md) · [docs/compose/llamacpp.md](docs/compose/llamacpp.md)

## Missing vs PyTorch (all in scope)

We are filling **the job**, not cloning the Python trees.

1. **PEFT / LoRA / QLoRA / TRL** — LoRA + SFT on V100, then QLoRA, then DPO. Unsloth kernels are out. → [docs/gaps/peft.md](docs/gaps/peft.md)
2. **Model zoo breadth** — YAML catalog + conversion notes, not 200 architectures. → [docs/gaps/zoo.md](docs/gaps/zoo.md)
3. **torchvision / torchaudio** — transforms + a small ONNX set. → [docs/gaps/vision-audio.md](docs/gaps/vision-audio.md)
4. **Accelerate / DeepSpeed / FSDP** — device profile, FP16, accum, checkpoint; Shardy later. → [docs/gaps/distributed.md](docs/gaps/distributed.md)
5. **Single fast backend** — Layer A now, Layer B as we learn. → [docs/gaps/backend.md](docs/gaps/backend.md)

## Project layout

```text
cmd/quikaitools/        CLI entrypoint
internal/cli/           doctor, catalog, pull, train
internal/backend/       Layer A device profiles
internal/hub/           Hugging Face download
internal/gpt2/          Tiny GPT-2 forward/backward
internal/peft/          LoRA adapters + SFT step
internal/train/         train lora job
internal/zoo/           Catalog loader
catalog/models/         Per-model YAML (status on v100 / halo / mac)
testdata/fixtures/      Checked-in train text (not artifacts)
docs/lab/               Machine install + “what is fast”
docs/compose/           How to use GoMLX, Hugot, llama.cpp
docs/gaps/              Designs for the five PyTorch gaps
```

## Development

```bash
git clone https://github.com/shaneburrell/quikaitools.git
cd quikaitools
make check
make build
```

| Make target | What it does |
|-------------|--------------|
| `make fmt` | `gofmt -l` check (fails if any file needs rewrite) |
| `make fmt-fix` | `gofmt` rewrite |
| `make lint` | `golangci-lint run ./...` |
| `make vet` | `go vet ./...` |
| `make test` | Unit tests |
| `make test-race` | Race detector |
| `make cover` | Coverage HTML + **70%** gate → `testdata/artifacts/` |
| `make bench` | Benchmarks → `testdata/artifacts/bench.txt` |
| `make tidy` | `go mod tidy` |
| `make tidy-check` | `go mod tidy -diff` (fails if go.mod/go.sum would change) |
| `make check` | tidy-check → fmt → vet → lint → race → cover |
| `make build` | `bin/quikaitools` |
| `make clean` | Remove `bin/`, `dist/`, `testdata/artifacts/` |

All generated coverage, benches, and CLI dumps go under **`testdata/artifacts/`** (gitignored). See [testdata/README.md](testdata/README.md).

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status & roadmap

**v0.1** — public guide, Layer A `doctor` / `catalog`, MIT, CI.

**v0.2** — `pull` + `train lora` on `hf-internal-testing/tiny-random-gpt2` (pure Go GPT-2 + LoRA).

**v0.3 (MVP)** — `train qlora`, `generate` (+ `--adapter` / `--gguf`), `embed` / `embed --vision` / `transcribe`, Accelerate-lite accum/checkpoint/resume, catalog-id pull, Go 1.27 CI. ONNX paths are labeled stubs until Hugot/ORT is linked; GGUF needs a local `llama-cli`. Trainer is portable Go (profile strings are Layer A intent, not a bound GoMLX session yet).

**v0.4** — `export merge|gguf|modelfile`, messages JSONL train + `validate-sft`, ChatML `--template` / `--messages` on `generate`. ONNX remains stub until Hugot/ORT is linked.

**v0.5** — real GPT-2 byte-level BPE; optimizer state on `--resume`; `--mask-prompt` loss masking for messages JSONL; `--seed`; sampling flags (`--temperature` / `--top-k` / `--top-p` / `--stop-eos`); `embed --vision --normalize clip`; Whisper-exact log-mel (`LogMelWhisper`); `HF_TOKEN` / `HF_ENDPOINT` / `HF_HOME`; `--allow-stub=false` fails closed on stub engines; HF PEFT `adapter_config.json` + `adapter_model.safetensors`; safetensors F16/BF16 read; CI runs gofmt, tidy, and golangci-lint with SHA-pinned actions.

**Next**

- Real ONNX via Hugot/ORT; DPO
- Instruct-model LoRA beyond GPT-2 (Layer B / GoMLX)
- Wire FP16 into a GPU train path (Layer B / GoMLX XLA CUDA)
- go-darwinml + Halo JAX ROCm PJRT experiments

## License

[MIT](LICENSE) © 2026 Shane Burrell

## Acknowledgments

GoMLX (Jan Pfeifer), Hugot (Knights Analytics), llama.cpp, and the Strix Halo community (kyuz0, bkpaine1, framework-rocm) did the hard compute work. This repo exists so a three-machine Go lab can use them as one toolset — and so the remaining PyTorch-shaped holes have a place to get filled.
