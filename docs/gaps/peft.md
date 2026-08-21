# Gap: PEFT / LoRA / QLoRA / Unsloth / TRL

**Job in PyTorch:** cheap fine-tune, SFT/DPO recipes, VRAM tricks.

**Status:** **MVP + export + messages JSONL.** LoRA + QLoRA + SFT on tiny GPT-2; `generate --adapter` closes train→use; `export merge` folds adapters; ChatML formatting on generate. DPO and instruct-model LoRA remain open. Unsloth kernels stay out.

## Shipped CLI

```bash
quikaitools pull tiny-random-gpt2
quikaitools train lora --model DIR --data FILE --accum 2 --resume DIR --profile mac
quikaitools train lora --model DIR --data train.jsonl --smoke   # messages JSONL → flatten
quikaitools train qlora --model DIR --data FILE
quikaitools validate-sft --path train.jsonl
quikaitools generate --model … --adapter … --prompt "…" --tokens 16
quikaitools generate --model … --template chatml --messages '[{"role":"user","content":"Hi"}]'
quikaitools export merge --model DIR --adapter DIR --out DIR
quikaitools export gguf --model DIR --out FILE.gguf   # skip if convert tools absent
quikaitools export modelfile --gguf FILE --out Modelfile
```

- `internal/peft`: LoRA on `c_attn` / `c_fc`, save/load `adapter.json`, greedy generate, **MergeIntoBase**
- `internal/peft` QLoRA: pack Conv1D to uint4, dequant in forward
- `internal/train`: messages JSONL flatten for GPT-2 smoke; `validate-sft`
- `internal/chatfmt`: ChatML / raw templates for `generate`
- `internal/dist`: accum, checkpoint `meta.json`, `--profile` (FP16 flag for V100)

## What we will not port

**Unsloth** is fused NVIDIA kernels. On V100 we meet the *need* (lower VRAM) with QLoRA, not Unsloth kernels.

## Still open

- Instruct-model LoRA (Qwen-class architectures) — Layer B / GoMLX
- True chat-template training (loss on assistant tokens only)
- DPO

## Success

Finite loss, adapter saves, `generate --adapter` runs, `export merge` writes densified weights. See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).
