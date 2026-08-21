# Gap: model zoo breadth

**Job in PyTorch:** `from_pretrained` across hundreds of architectures.

**Status:** v0.1 ships the catalog format and a starter set. Breadth is **registry + conversion**, not a transformers fork.

## Format

One YAML file per model in [`catalog/models/`](../../catalog/models). Required fields: `id`, `task`. Machines use `works` | `cpu_only` | `untested` | `wont`.

```yaml
id: distilbert-sst2
task: classify
formats: [onnx]
machines:
  v100: works
  halo: cpu_only
  mac: works
engine:
  v100: hugot-ort-cuda
  halo: hugot-go
  mac: hugot-coreml
```

Non-V100 NVIDIA cards reuse the `v100` column (`backend.Profile.CatalogMachine`).

## How breadth grows

1. Every model that already runs in Hugot, a GoMLX demo, or llama.cpp gets an entry.
2. HF → ONNX (Optimum) and HF → GGUF are **documented conversion boundaries**. Runtime stays Go.
3. New architectures land upstream (GoMLX/Hugot) or as a thin GoMLX port — then we catalog them.

## Conversion boundary

We will not pretend Python disappeared for export. The guide will keep a short, pinned recipe (Optimum / llama.cpp convert) so the lab does that **once** per model. Serving and train loops stay in this repo.

## CLI

```bash
quikaitools catalog
quikaitools catalog --task generate --machine halo
```
