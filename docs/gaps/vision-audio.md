# Gap: torchvision / torchaudio

**Job in PyTorch:** datasets, transforms, pretrained vision/audio models.

**Status:** first catalog targets exist (`clip-vit-base-patch32`, `whisper-small`). Packages `internal/vision` and `internal/audio` land after LoRA/SFT.

## Slice we will write

- **vision:** decode (`image`), resize, center/random crop, ImageNet normalize, to CHW float32. GoCV only if stdlib is not enough.
- **audio:** resample, mono, log-mel enough for Whisper-class ONNX. Not a torchaudio clone.
- Catalog + Hugot/GoMLX for **one** classify, **one** vision embed, **one** ASR.

## What we will not write

The torchvision model zoo, detection/segmentation training, or torchcodec.

## Success

`quikaitools embed --task vision-embed` and `quikaitools transcribe` against catalog IDs, using Layer A to pick ORT CUDA vs Go session vs CoreML.
