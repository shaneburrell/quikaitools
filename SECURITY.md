# Security Policy

## Supported versions

Fixes land on `main` and the newest `v*` tag.

## What this tool talks to

The CLI is local (`doctor`, `catalog`, `pull`, `train`, `generate`, `embed`, `transcribe`, `export`, `validate-sft`). It does not start servers or pin TLS.

It does:

- **`pull`** — download weights from `huggingface.co` (or `HF_ENDPOINT`) into `QUIKAITOOLS_CACHE` / `HF_HOME`, honoring `HF_TOKEN` when set
- **`generate --gguf`** — shell out to a local llama.cpp binary (`llama-cli`, or `QUIKAITOOLS_LLAMA`)
- **`export gguf`** — shell out to a local Python convert script (`convert_hf_to_gguf.py`) and `llama-quantize` when those tools are on PATH
- Read models and tokenizers from disk
- Load shared libraries later (ONNX Runtime, llama.cpp, XLA PJRT)

Treat those binaries and libraries as part of your trust boundary. Prefer checksummed Hub downloads and pinned plugin paths (`GOMLX_NO_AUTO_INSTALL=1` in production images).

## Reporting

Open a private GitHub security advisory on [shaneburrell/quikaitools](https://github.com/shaneburrell/quikaitools) or email the maintainer via the GitHub profile. Do not file public issues for exploitable bugs in model loaders or path handling.

## Secrets

Never commit `.env`, Hub tokens, or `HF_TOKEN`. Catalog YAML is public metadata only.
