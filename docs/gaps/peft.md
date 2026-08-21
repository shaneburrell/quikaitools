# Gap: PEFT / LoRA / QLoRA / Unsloth / TRL

**Job in PyTorch:** cheap fine-tune, SFT/DPO recipes, VRAM tricks.

**Status:** **MVP shipped.** LoRA + QLoRA + SFT on tiny GPT-2; `generate --adapter` closes train→use. DPO is next tag. Unsloth kernels stay out.

## Shipped CLI

```bash
quikaitools pull tiny-random-gpt2
quikaitools train lora --model DIR --data FILE --accum 2 --resume DIR --profile mac
quikaitools train qlora --model DIR --data FILE   # 4-bit frozen base + FP32 LoRA
quikaitools generate --model DIR --adapter DIR --prompt "…" --tokens 16
```

- `internal/peft`: LoRA on `c_attn` / `c_fc`, save/load `adapter.json`, greedy generate.
- `internal/peft` QLoRA: pack Conv1D to uint4, dequant in forward.
- `internal/dist`: accum, checkpoint `meta.json`, `--profile` (FP16 flag for V100).
- `--eval-every N` smoke-evaluates a forward pass.

## What we will not port

**Unsloth** is fused NVIDIA kernels. On V100 we meet the *need* (lower VRAM) with QLoRA, not Unsloth kernels.

## Per machine

| Machine | MVP |
|---------|-----|
| Mac | Full train→generate path (e2e). |
| V100 | Same Go trainer; FP16 job flag ready; Layer B GoMLX later. |
| Halo | Same CPU LoRA path; apply adapters at infer (GGUF) preferred. |

## Success

Finite loss, adapter saves, `generate --adapter` runs. See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).
