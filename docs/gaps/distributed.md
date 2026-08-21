# Gap: Accelerate, DeepSpeed, FSDP

**Job in PyTorch:** one trainer API, mixed precision, multi-GPU shard, ZeRO.

**Status:** Layer A `backend.Profile` is the start of Accelerate-lite. `internal/dist` is not shipped in v0.1.

## Order

1. **Accelerate-lite** — `DeviceProfile`, engine bind, **FP16 on V100** (not BF16), gradient accumulation, checkpoint/resume. Every trainer uses this.
2. **Cheap DeepSpeed-shaped extras** — activation checkpointing (GoMLX already has an API). Optimizer-state offload only if 16 GB forces it.
3. **FSDP / ZeRO-3** — wrap [GoMLX Shardy](https://gomlx.github.io/docs/overview/) when the lab has **two or more V100s**. Halo and Mac stay single-device.

## What we will not write

DeepSpeed’s kernel tree, TorchElastic, multi-node fault recovery.

## Per machine

| Machine | Dist story |
|---------|------------|
| One V100 | Accum + checkpoint + FP16. No shard. |
| Multi-V100 | Shardy / FSDP-lite later. |
| Halo | One fat unified device. Sharding does not buy much. |
| Mac | One device. |

## Success

A trainer constructed with `dist.New(profile)` respects `MixedPrecision` from `doctor` and refuses BF16 on V100.
