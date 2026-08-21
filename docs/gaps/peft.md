# Gap: PEFT / LoRA / QLoRA / Unsloth / TRL

**Job in PyTorch:** cheap fine-tune, SFT/DPO recipes, VRAM tricks.

**Status:** design in this doc; package `internal/peft` and `internal/train` not shipped in v0.1.

## What we will write

1. **LoRA** on GoMLX linear/attention (`internal/peft`): rank, alpha, dropout, freeze base, adapter-only save/load.
2. **SFT trainer** (`internal/train`): chat template, loss, eval. TRL’s most-used loop.
3. **QLoRA** next, because V100 is 16/32 GB. 4-bit base + FP16 LoRA. V100 first.
4. **DPO** after SFT works. Later GRPO if anyone needs it.

## What we will not port

**Unsloth** is fused NVIDIA kernels and graph rewrites. It will never be the portable backend. On V100 we meet the *need* (faster, lower-VRAM fine-tune) with LoRA/QLoRA + GoMLX XLA fusion.

Full TRL (PPO, every reward trainer) is a later catalog, not the first two trainers.

## Per machine

| Machine | Plan |
|---------|------|
| V100 | Live here. FP16. QLoRA on 16 GB. |
| Mac | LoRA only if Relux/go-darwinml can train the base; otherwise CPU demo. |
| Halo | Apply or merge adapters at **infer** (GGUF) before Halo GPU train exists. |

## Success

`quikaitools train sft --profile v100 --adapter lora` against a catalog model, checkpoint that `doctor` still reports `gomlx-xla-cuda`.
