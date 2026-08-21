# Gap: torchvision / torchaudio

**Job in PyTorch:** datasets, transforms, pretrained vision/audio models.

**Status:** **MVP slice shipped.** Not a torchvision/torchaudio clone.

## Shipped

- `internal/vision`: decode PNG/JPEG, resize, center-crop, ImageNet normalize, CHW float32
- `internal/audio`: WAV PCM16, mono, resample, log-mel
- CLI:
  - `quikaitools embed --vision --model DIR --image FILE`
  - `quikaitools transcribe --model DIR --audio FILE`

Full CLIP/Whisper ONNX sessions land when Hugot/ORT is linked. Until then, `embed --vision` / `transcribe` run **real preprocess** (`internal/vision`, `internal/audio`) and return clearly labeled stub vectors/transcripts (`vision-gap-stub`, `whisper-mel-stub`, `onnx-stub-*`) so the CLI path is exerciseable on Mac without CGO.

## Fixtures

- `testdata/fixtures/red.png`
- `testdata/fixtures/tone.wav`

## Success

Vision/audio unit tests pass in CI; Mac smoke checklist covers embed/transcribe. See [../lab/mvp-smoke.md](../lab/mvp-smoke.md).
