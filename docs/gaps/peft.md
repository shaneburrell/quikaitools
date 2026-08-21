# Gap: PEFT / LoRA / QLoRA / Unsloth / TRL

**Job in PyTorch:** cheap fine-tune, SFT/DPO recipes, VRAM tricks.

**Status:** v0.2 ships LoRA + SFT on a tiny GPT-2 (`quikaitools pull` + `train lora`). QLoRA and DPO are still later.

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

## Lab command (tiny model)

```bash
quikaitools pull hf-internal-testing/tiny-random-gpt2
quikaitools train lora \
  --model ~/.cache/quikaitools/models/hf-internal-testing/tiny-random-gpt2 \
  --data testdata/fixtures/stories.txt \
  --steps 30 --rank 4 --out testdata/artifacts/adapter-lora
```

The Hub model is ~450KB (n_embd=32, 5 layers). Weights are random; this proves the Go LoRA loop, not story quality. Adapter JSON is written to `--out`.

## Success

`quikaitools train lora` against the pulled tiny GPT-2 writes `adapter.json` and a finite loss curve.
