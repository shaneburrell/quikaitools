# Lab profile: AMD Strix Halo (gfx1151)

Ryzen AI Max+ 395 / Radeon 8060S, large **unified** LPDDR5X. This is the **large-GGUF inference box**. It is not Instinct MI300. Consumer Halo ROCm works in 2026 if kernel and ROCm versions match.

## What is fast

| Job | Engine | Notes |
|-----|--------|-------|
| GGUF generate (default) | llama.cpp **Vulkan RADV** | Most stable. Often best token-gen (bandwidth-bound). |
| GGUF prefill / long context | llama.cpp **HIP / ROCm 7.2.4** | Tuned gfx1151 kernels. See flags below. |
| ONNX embed / classify | Hugot `NewGoSession` | ORT dropped the ROCm EP in 1.23. No Halo GPU ONNX path we trust yet. |
| Train in Go | GoMLX `go` or XLA **CPU** | iGPU train via GoMLX is unpublished. JAX ROCm PJRT is an experiment ([../gaps/backend.md](../gaps/backend.md)). |

## Known-good HIP stack (mid-2026)

- ROCm **7.2.4** (or 7.2.1–7.2.3 on the matching kernel)
- Linux **6.18.4+**, or Ubuntu 24.04 HWE ≥ `6.17.0-19` / OEM ≥ `6.14.0-1018`
- **Do not** use `linux-firmware-20251125` (breaks ROCm on Halo)
- BIOS: minimize dedicated VRAM; let GTT use unified memory
- Build llama.cpp:

```bash
cmake -S llama.cpp -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
export ROCBLAS_USE_HIPBLASLT=1
```

`-DGGML_HIP_NO_VMM=ON` is not optional: HIP virtual memory has corrupted allocations on gfx1151.

`ROCBLAS_USE_HIPBLASLT=1` is the large **prompt-processing** win. ROCm 7.2 ships native gfx1151 hipBLASLt kernels — you do **not** need `HSA_OVERRIDE_GFX_VERSION`.

Launch flags that keep people out of trouble: `--no-mmap` and flash attention on.

## Vulkan default

If HIP is not worth the pain today, use Mesa **RADV**:

```bash
# community containers: docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv
llama-server -m model.gguf -ngl 999 -fa 1 --no-mmap
```

Vulkan often wins short-context decode. Tuned HIP usually wins prefill, long context, and bigger MoE.

## Doctor contract

`auto` on Linux with `/dev/kfd` or `rocminfo` showing `gfx1151` becomes `halo`. Train engine will say `gomlx-go` on purpose. That is not a failed install.

## See also

- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [bkpaine1/halo-llamacpp](https://github.com/bkpaine1/halo-llamacpp)
- [AMD RDNA 3.5 kernel/ROCm matrix](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
- [../compose/llamacpp.md](../compose/llamacpp.md)
