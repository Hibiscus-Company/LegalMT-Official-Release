# LegalMT

English-Vietnamese legal document translation (VLSP 2026, Document-Level Legal MT).
Base model: [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B).

## Setup

```bash
uv sync                                # CPU
scripts/LegalMT/setup_gpu_env.sh       # GPU
source scripts/LegalMT/env.sh
uv run --group gpu hf download tencent/Hy-MT2-1.8B --revision 9a341cd1b679d3efd23b46e847b01745a71ed792
```

## Translate

```bash
scripts/LegalMT/gpu_run.sh scripts/LegalMT/translate.py --config hymt2_final_graphs \
    --input-file in.md --output-file out.md
```

For fine-tuned weights add `--set model.local_path=<merged model dir>`.

## Docker

```bash
docker/build_image.sh --serve-config configs/LegalMT/serve.final.yaml [--model-dir <merged model dir> --no-adapter]
docker run -d --gpus all --shm-size 2g -p 8000:8000 <image>
docker/translate.sh localhost:8000 in.md out.md
```

## Tests

```bash
uv run pytest
```
