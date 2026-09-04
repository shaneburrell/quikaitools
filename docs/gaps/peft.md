# Gap: PEFT / LoRA / QLoRA / Unsloth / TRL

**Job in PyTorch:** cheap fine-tune, SFT/DPO recipes, VRAM tricks.

**Status:** **MVP + export + messages JSONL.** LoRA + QLoRA + SFT on tiny GPT-2; `generate --adapter` closes train→use; `export merge` folds adapters; ChatML formatting on generate. DPO and instruct-model LoRA remain open. Unsloth kernels stay out.

## Shipped CLI

```bash
quikaitools pull tiny-random-gpt2
quikaitools train lora --model DIR --data FILE --accum 2 --resume DIR --profile mac --seq 32 --seed 1
quikaitools train lora --model DIR --data train.jsonl --smoke --mask-prompt   # JSONL by extension; --smoke = steps=20, seq=32, accum=4
quikaitools train qlora --model DIR --data FILE
quikaitools validate-sft --path train.jsonl
quikaitools generate --model … --adapter … --prompt "…" --tokens 16 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 1 --stop-eos
quikaitools generate --model … --template chatml --messages '[{"role":"user","content":"Hi"}]'
quikaitools export merge --model DIR --adapter DIR --out DIR
quikaitools export gguf --model DIR --out FILE.gguf   # skip if convert tools absent
quikaitools export modelfile --gguf FILE --out Modelfile   # --force to overwrite
```

- `internal/peft`: LoRA on `c_attn` / `c_fc`, save/load `adapter.json`, greedy generate, **MergeIntoBase**
- `internal/peft` QLoRA: pack Conv1D to uint4, dequant in forward
- `internal/train`: messages JSONL flatten (`.jsonl` extension, not `--smoke`); `validate-sft`
- `internal/chatfmt`: ChatML / raw templates for `generate`
- `internal/dist`: accum, checkpoint `meta.json`, `--profile` (FP16 flag for V100)
- `--mask-prompt` trains loss on assistant completion tokens only; `--seq` / `--seed` on `train`
- `generate` sampling: `--temperature` / `--top-k` / `--top-p` / `--seed` / `--stop-eos` (in-process; rejected with `--gguf`)

## What we will not port

**Unsloth** is fused NVIDIA kernels. On V100 we meet the *need* (lower VRAM) with QLoRA, not Unsloth kernels.

## Still open

- Instruct-model LoRA (Qwen-class architectures) — Layer B / GoMLX
- True chat-template training (loss on assistant tokens only)
- DPO

## Success

Finite loss, adapter saves, `generate --adapter` runs, `export merge` writes densified weights. See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).

## Adapter format

`Save` still writes legacy `adapter.json` (A `[in, r]`, B `[r, out]`, matching GPT-2 Conv1D `y = x @ W`) plus `optimizer.json`. It also writes Hugging Face PEFT files:

- `adapter_config.json` — `peft_type=LORA`, `task_type=CAUSAL_LM`, `r`, `lora_alpha`, `lora_dropout=0`, `bias=none`, `fan_in_fan_out=true`, `target_modules=["c_attn","c_fc"]`, `inference_mode=false`
- `adapter_model.safetensors` — F32 tensors with `__metadata__ {"format":"pt"}`

PEFT stores `lora_A` as nn.Linear `[r, in]` and `lora_B` as `[out, r]`. Our trainer keeps A `[in, r]` and B `[r, out]`, so both matrices are **transposed on write and on read**. Keys:

`base_model.model.transformer.h.{N}.attn.c_attn.lora_A.weight` `[r, in]`  
`base_model.model.transformer.h.{N}.attn.c_attn.lora_B.weight` `[out, r]`  
and the same for `mlp.c_fc`.

`Load` prefers the PEFT pair when both files exist (missing tensors stay zero-B); otherwise it falls back to `adapter.json`. Scale is `lora_alpha / r`.
