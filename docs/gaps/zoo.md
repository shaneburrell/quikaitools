# Gap: model zoo breadth

**Job in PyTorch:** `from_pretrained` across hundreds of architectures.

**Status:** **MVP shipped** for catalog + `pull <catalog-id>`. Breadth stays registry + conversion, not a transformers fork.

## Schema (`zoo.Model`)

Each `catalog/models/*.yaml` file maps to one `zoo.Model`:

| Field | YAML | Type | Notes |
|-------|------|------|--------|
| `ID` | `id` | string | required, unique |
| `Name` | `name` | string | display name |
| `Task` | `task` | string | required catalog task id (`generate`, `embed`, `asr`, `classify`, `token-classify`, `rerank`, `vision-embed`, …) |
| `Formats` | `formats` | `[]string` | e.g. `safetensors`, `onnx`, `gguf` |
| `License` | `license` | string | SPDX-ish label |
| `Source` | `source` | string | Hub URL or `org/name` |
| `Notes` | `notes` | string | free text |
| `Machines` | `machines` | `map[string]Status` | keys: `v100`, `halo`, `mac` (and optionally `cpu`) |
| `Engine` | `engine` | `map[string]string` | recommended engine label per machine |

Allowed `machines` status values (`zoo.Status`):

| Status | Meaning |
|--------|---------|
| `works` | ran on that box |
| `cpu_only` | ran, but CPU only |
| `untested` | not tried |
| `wont` | will not work / out of scope |

`zoo.LoadDir(dir)` reads `*.yaml` / `*.yml` in `dir`. If that directory has no YAML files, it falls back to `<dir>/models/*.yaml`. `zoo.LoadFS` walks an `fs.FS` (the embedded `catalog.Models` set).

## Shipped

```bash
quikaitools catalog [--task TASK] [--machine v100|halo|mac|cpu] [--catalog DIR]
quikaitools pull tiny-random-gpt2          # resolves YAML source
quikaitools pull hf-internal-testing/…   # raw Hub id still works
```

`--task` matches catalog task ids. `transcribe` is accepted as an alias for `asr`.

Starter zoo (Mac-runnable paths):

| ID | Task |
|----|------|
| `tiny-random-gpt2` | `generate` (train / generate LoRA) |
| `distilbert-sst2` / `nomic-embed-text` | `classify` / `embed` (ONNX targets) |
| `gemma3-270m-it` | `generate` (small GGUF/ONNX) |
| `whisper-small` | `asr` (CLI `transcribe` is an alias) |
| `clip-vit-base-patch32` | `vision-embed` |

Verified status is whatever the YAML says, not the smoke checklist:

| ID | v100 | halo | mac |
|----|------|------|-----|
| `tiny-random-gpt2` | `works` | `works` | `works` |
| `distilbert-sst2` | `untested` | `untested` | `untested` |
| `nomic-embed-text` | `untested` | `untested` | `untested` |
| `gemma3-270m-it` | `untested` | `untested` | `untested` |
| `whisper-small` | `untested` | `untested` | `untested` |
| `clip-vit-base-patch32` | `untested` | `untested` | `untested` |
| `bert-base-ner` | `untested` | `untested` | `untested` |
| `mxbai-rerank-base-v1` | `untested` | `untested` | `untested` |

`tiny-random-gpt2` is the only entry marked `works` on V100/Halo/Mac (portable Go LoRA trainer). Halo/V100 GPU infer/train boxes are still unchecked in [mvp-smoke.md](../lab/mvp-smoke.md).

ONNX catalog targets (`nomic-embed-text`, CLIP, Whisper, classify/rerank) stay `untested` with **stub** runtime until Hugot/ORT is linked — do not mark `works` for ONNX embed/ASR/classify.

## Conversion boundary

HF → ONNX / GGUF stays a documented export step. Runtime stays Go.

## Success

`pull <id>` resolves `source` from YAML; catalog status columns match the smoke matrix.
