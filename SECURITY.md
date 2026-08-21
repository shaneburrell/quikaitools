# Security Policy

## Supported versions

Fixes land on `main` and the newest `v*` tag.

## What this tool talks to

v0.1 is a **local CLI** (`doctor`, `catalog`). It does not start servers, pin TLS, or download weights by default.

Later train/infer commands will:

- Read models and tokenizers from disk or the Hugging Face Hub
- Load shared libraries (ONNX Runtime, llama.cpp, XLA PJRT)

Treat those libraries as part of your trust boundary. Prefer checksummed Hub downloads and pinned plugin paths (`GOMLX_NO_AUTO_INSTALL=1` in production images).

## Reporting

Open a private GitHub security advisory on [shaneburrell/quikaitools](https://github.com/shaneburrell/quikaitools) or email the maintainer via the GitHub profile. Do not file public issues for exploitable bugs in model loaders or path handling.

## Secrets

Never commit `.env`, Hub tokens, or `HF_TOKEN`. Catalog YAML is public metadata only.
