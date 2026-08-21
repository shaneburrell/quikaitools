# Gap: Accelerate, DeepSpeed, FSDP

**Job in PyTorch:** one trainer API, mixed precision, multi-GPU shard, ZeRO.

**Status:** **MVP shipped** for Accelerate-lite. FSDP/DeepSpeed remain post-MVP.

## Shipped

`internal/dist.Job`:

- `Profile` from `doctor` / `--profile`
- `AccumSteps` (`train … --accum N`) — real gradient accumulation: microbatches share grads; Adam runs once per accum window
- `CheckpointDir` + `meta.json` every N steps
- `--resume DIR` loads `adapter.json` + step
- `FP16` flag is set on V100/CUDA profiles but **compute stays FP32** in the portable Go trainer until Layer B; reserved for GoMLX/XLA later
- `ShardModeNone` only

```bash
quikaitools train lora --model DIR --data FILE --accum 2 --resume DIR --profile mac
```

## Not in MVP

DeepSpeed ZeRO-3, TorchElastic, multi-V100 Shardy/FSDP.

## Success

Trainer constructed with `dist.NewJob(profile)` respects accum/checkpoint/resume. See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).
