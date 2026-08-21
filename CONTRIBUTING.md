# Contributing to QuikAITools

Thanks for helping. This project is a **guide first**, then missing tooling. Correctness and honesty beat a fake “PyTorch in Go” claim.

## Development setup

1. Install [Go](https://go.dev/dl/) 1.22+.
2. Clone and test:

```bash
git clone https://github.com/shaneburrell/quikaitools.git
cd quikaitools
make check
make build
```

| Command | Purpose |
|---------|---------|
| `make fmt` | `gofmt` |
| `make vet` | `go vet ./...` |
| `make test` | Unit tests |
| `make test-race` | Race detector |
| `make check` | tidy → fmt → vet → race |
| `make build` | `bin/quikaitools` |

## Before you open a PR

- [ ] `make check` passes
- [ ] New catalog models have `id`, `task`, and machine status
- [ ] Lab/compose/gap docs updated if you change what `doctor` binds
- [ ] No secrets, `.env`, or local model weights committed

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
