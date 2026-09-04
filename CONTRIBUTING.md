# Contributing to QuikAITools

Thanks for helping. This project is a **guide first**, then missing tooling. Correctness and honesty beat a fake “PyTorch in Go” claim.

## Development setup

1. Install [Go](https://go.dev/dl/) 1.27+.
2. Clone and test:

```bash
git clone https://github.com/shaneburrell/quikaitools.git
cd quikaitools
make check
make build
```

| Command | Purpose |
|---------|---------|
| `make fmt` | `gofmt -l` check (fails if any file needs rewrite) |
| `make fmt-fix` | `gofmt` rewrite |
| `make lint` | `golangci-lint run ./...` (errcheck, govet, staticcheck, unused, …) |
| `make vet` | `go vet ./...` |
| `make test` | Unit tests (also writes doctor/catalog dumps under `testdata/artifacts/`) |
| `make test-race` | Race detector |
| `make cover` | Coverage HTML + **70%** gate on `./internal/...` → `testdata/artifacts/` |
| `make bench` | Benchmarks → `testdata/artifacts/bench.txt` |
| `make tidy` | `go mod tidy` |
| `make tidy-check` | `go mod tidy -diff` |
| `make check` | tidy-check → fmt → vet → lint → race → cover |
| `make build` | `bin/quikaitools` |
| `make clean` | Remove `bin/`, `dist/`, `testdata/artifacts/` |

CI (`.github/workflows/ci.yml`) enforces `gofmt -l`, `go mod tidy -diff`, and golangci-lint on Ubuntu/macOS.

Generated coverage, benches, and CLI dumps land in `testdata/artifacts/` (gitignored). See [testdata/README.md](testdata/README.md). Never commit those files, `*.gguf`, or `adapters/`.

## Before you commit

- [ ] `make check` passes
- [ ] New catalog models have `id`, `task`, and machine status
- [ ] Lab/compose/gap docs updated if you change what `doctor` binds
- [ ] No secrets, `.env`, local model weights, `testdata/artifacts/`, or `*.gguf` committed

## Design priorities

1. **Compose maintained projects** — GoMLX, Hugot, llama.cpp. Do not start another tensor core.
2. **One profile per machine** — V100, Halo, Mac. Fast path must be named in `doctor`.
3. **Fill the PyTorch job, not the Python tree** — LoRA not Unsloth; catalog not Hugging Face; Accelerate-lite not DeepSpeed.
4. **Write down failures** — especially Halo PJRT and go-darwinml experiments in `docs/gaps/backend.md`.

## Catalog entries

Add `catalog/models/<id>.yaml`. Status values: `works`, `cpu_only`, `untested`, `wont`. If you only ran CPU, do not mark `works` on `v100`.

## Commit messages

Short, imperative, explain *why* when it is not obvious:

```text
catalog: mark whisper-small cpu_only on halo

ORT has no ROCm EP; Go session ran the ONNX encode path.
```

## License

By contributing you agree the work is MIT, copyright Shane Burrell and contributors.
