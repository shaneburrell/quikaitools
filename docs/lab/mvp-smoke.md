# MVP smoke checklist (v0.3)

Same CLI on Mac, Strix Halo, and V100. **Mac is the e2e box for this tag.** Halo and V100 share the commands; fill their checkboxes when those machines are on.

## Mac (filled)

Profile: Apple Silicon → `doctor` reports `mac`, `infer_gguf=llamacpp-metal`, `infer_onnx=hugot-coreml`.

| Check | Command | Result |
|-------|---------|--------|
| doctor | `./bin/quikaitools doctor --profile auto` | expect `mac` |
| doctor strict | `./bin/quikaitools doctor --strict` | ok if `llama-cli` (or `QUIKAITOOLS_LLAMA`) on PATH; else prints install hint and exits 1 |
| pull catalog id | `./bin/quikaitools pull tiny-random-gpt2` | resolves to `hf-internal-testing/tiny-random-gpt2` |
| train lora | `./bin/quikaitools train lora --model ~/.cache/quikaitools/models/hf-internal-testing/tiny-random-gpt2 --data testdata/fixtures/stories.txt --steps 20 --rank 4 --accum 2 --seq 32 --seed 1 --out testdata/artifacts/adapter-lora --profile mac` | loss finite; adapter + `meta.json` |
| train resume | `./bin/quikaitools train lora --model … --data … --steps 5 --resume testdata/artifacts/adapter-lora --out testdata/artifacts/adapter-lora-r2` | resumes from adapter |
| generate + adapter | `./bin/quikaitools generate --model … --adapter testdata/artifacts/adapter-lora --prompt "Once upon" --tokens 8 --temperature 0.8 --seed 1` | prints tokens (train→use) |
| train qlora | `./bin/quikaitools train qlora --model … --data … --steps 10 --out testdata/artifacts/adapter-qlora` | 4-bit frozen base + LoRA; loss finite |
| embed text | `./bin/quikaitools embed --model … --text "hello world"` | `engine=bag-of-wte` on GPT-2 folders; ONNX catalogs stay stub until ORT |
| embed vision | `./bin/quikaitools embed --vision --model . --image testdata/fixtures/red.png` | real preprocess + labeled stub vector |
| transcribe | `./bin/quikaitools transcribe --model . --audio testdata/fixtures/tone.wav` | real mel + labeled stub transcript |
| generate GGUF | `./bin/quikaitools generate --gguf PATH/to/model.gguf --prompt "Hi" --tokens 16 --timeout 30s` | requires llama.cpp Metal binary |
| export merge | `./bin/quikaitools export merge --model … --adapter testdata/artifacts/adapter-lora --out testdata/artifacts/merged` | untested |
| export gguf | `./bin/quikaitools export gguf --model … --out testdata/artifacts/model.gguf` | untested (SKIP when converter missing) |
| export modelfile | `./bin/quikaitools export modelfile --gguf FILE --out testdata/artifacts/Modelfile` | untested (`--force` overwrites) |
| validate-sft | `./bin/quikaitools validate-sft --path FILE.jsonl` | untested |
| generate ChatML | `./bin/quikaitools generate --model … --template chatml --messages '[{"role":"user","content":"Hi"}]'` | untested |

v0.4 rows (`export`, `validate-sft`, ChatML `generate`) are **untested** on Mac — README does not claim Mac e2e for them.

CI stays CPU-only (`go test ./...`); Hub/Metal runs are local. Artifacts under `testdata/artifacts/` are gitignored.

## Halo (blank — run when box is on)

Expect: `doctor` → `halo`, GGUF = Vulkan (HIP optional). If `auto` prints `cpu`, smoke **failed**.

- [ ] `quikaitools doctor --strict` matches Halo profile
- [ ] `quikaitools generate --gguf …` on a small GGUF (Vulkan)
- [ ] `quikaitools embed --model … --text …` (ONNX Go session / stub)
- [ ] Optional: `train lora` on tiny GPT-2 (CPU) to prove trainer portability

## V100 (blank — run when box is on)

Expect: `doctor` → `v100`, `mixed_precision=fp16`, GGUF = CUDA, ONNX = ORT CUDA. If `auto` prints `cpu`, smoke **failed**.

- [ ] `quikaitools doctor --strict` matches V100 profile
- [ ] `quikaitools generate --gguf …` (CUDA)
- [ ] `quikaitools embed …` on CUDA / ORT path
- [ ] Optional: `train lora` / `train qlora` on tiny GPT-2 (Go trainer; GoMLX XLA CUDA is post-MVP Layer B)

## Not in this MVP

Unsloth kernels, DeepSpeed ZeRO-3, GoMLX-on-Halo-GPU, DPO, full HF zoo, full torchvision/torchaudio clones.
