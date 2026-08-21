# Gap: model zoo breadth

**Job in PyTorch:** `from_pretrained` across hundreds of architectures.

**Status:** **MVP shipped** for catalog + `pull <catalog-id>`. Breadth stays registry + conversion, not a transformers fork.

## Shipped

```bash
quikaitools catalog [--task] [--machine]
quikaitools pull tiny-random-gpt2          # resolves YAML source
quikaitools pull hf-internal-testing/…   # raw Hub id still works
```

Starter zoo (Mac-runnable paths):

| ID | Role |
|----|------|
| `tiny-random-gpt2` | train / generate LoRA |
| `distilbert-sst2` / `nomic-embed-text` | ONNX embed targets |
| `gemma3-270m-it` | small GGUF/ONNX generate |
| `whisper-small` | transcribe |
| `clip-vit-base-patch32` | vision-embed |

After Mac e2e, `mac: works` on the preprocess/CLI paths. Halo/V100 stay `untested` until [mvp-smoke.md](../lab/mvp-smoke.md) is filled.

## Conversion boundary

HF → ONNX / GGUF stays a documented export step. Runtime stays Go.

## Success

`pull <id>` resolves `source` from YAML; catalog status columns match the smoke matrix.
