# LegalMT

English-Vietnamese document-level legal translation for VLSP 2026 (Document-Level Legal MT, models under 3B parameters).
Base model: [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B), revision `9a341cd1b679d3efd23b46e847b01745a71ed792`, fine-tuned with LoRA. Order: Setup, Data, Train (produces the model), then Translate or Docker with that model.

## Requirements

| | |
|---|---|
| Software | Python 3.12+, [uv](https://docs.astral.sh/uv/); Docker with the NVIDIA Container Toolkit for the image |
| Train | NVIDIA GPU with 12 GB or more; about 32 h per model on an RTX 3060 |
| Translate | NVIDIA GPU, driver 580+ (CUDA 13), about 8 GB of GPU memory |
| Data | the organizers' CSVs (see Data) |
| Disk | about 15 GB for the GPU environment and the base model, more for training outputs |

## Layout

```
.
├── configs/LegalMT/
│   ├── infer/                inference configs (hymt2_final_graphs.yaml is the final one)
│   ├── sft/                  SFT data builds (sft_v1_final.yaml: organizers' data only)
│   ├── train/                LoRA training configs: final_d3, final_d3b, final_d3q, shared base, pilots
│   ├── serve.yaml            Docker server config (default)
│   ├── serve.final.yaml      Docker server config of the final image
│   ├── models.yaml           pinned model revisions
│   ├── paths.yaml            data and output locations
│   ├── align.yaml            paragraph alignment settings (align_v1.yaml is used by the SFT builds)
│   ├── splits.yaml           dev-split protocol
│   ├── dev_split_v0.json     the frozen dev split (document ids)
│   ├── data_quality_v0.yaml  data-quality lists (flagged documents)
│   ├── dataset.yaml          expected dataset facts (checked by smoke_data.py)
│   └── glossary_seed_v0.tsv  EN-VI legal glossary for the terminology metric
├── docker/
│   ├── Dockerfile           image definition (.dockerignore keeps the data out)
│   ├── build_image.sh       stage the build context and build the image
│   ├── package.sh           archive the image (tar.gz and MD5) with the client script
│   ├── server.py            HTTP translation server inside the image
│   └── translate.sh         client: translate.sh HOST:PORT INPUT OUTPUT
├── scripts/LegalMT/
│   ├── translate.py         translate a file or a split; --evaluate scores it
│   ├── build_aligned.py     align EN/VI paragraphs
│   ├── build_sft.py         build SFT examples from the alignment
│   ├── train_lora.py        LoRA fine-tuning
│   ├── merge_lora.py        merge an adapter into the base model
│   ├── private_round.py     AIHub round: translate the input, write results.csv and the zip
│   ├── private_mbr.py       unit-level MBR over several decoding passes or models
│   ├── evaluate.py, evaluate_segments.py, comet_score.py, make_results_table.py    scoring and result tables
│   ├── make_dev_split.py, check_glossary.py, inspect_alignment.py, smoke_data.py   data checks
│   ├── dev_views.py, ensemble_check.py, postprocess_run.py, latency_probe.py       analysis of finished runs, latency
│   ├── soup_merge.py, pref_sample.py, pref_train.py    weight averaging and DPO experiments (not used by the final models)
│   └── env.sh, gpu_run.sh, setup_gpu_env.sh, setup_comet_env.sh, serve_local_test.sh   environment scripts, local serving test
├── src/LegalMT/
│   ├── layout.py            layout parser: documents to translation units and back
│   ├── chunking.py          pack units into model-sized requests
│   ├── prompts.py           prompt construction, shared by SFT and inference
│   ├── translate/           Translator (core.py) and the vLLM, Transformers and echo backends (backends.py)
│   ├── align.py, sft_data.py                       alignment and SFT example building
│   ├── data.py, paths.py, runs.py, splits.py, quality.py   data loading, paths, run folders, dev split, data checks
│   ├── metrics.py, segments.py, terminology.py, profiling.py   document and segment scoring, terminology, time and memory
│   └── langid.py, textnorm.py, postprocess.py      direction detection, text normalisation, output clean-up
├── tests/                   pytest suite (CPU; an echo backend stands in for the model)
├── tools/comet/             separate environment for COMET scoring
└── pyproject.toml, uv.lock, .python-version        dependencies and Python version
```

Outside the repository: `data/raw/Dataset/` holds the organizers' CSVs (see Data); `../legalmt-artifacts/` receives the outputs (`aligned/`, `sft/`, `checkpoints/`, `merged/`, `runs/`).

## Setup

Run all commands from the repository root, in one shell.

```bash
uv sync                                  # CPU environment: tests, data scripts
scripts/LegalMT/setup_gpu_env.sh         # GPU environment (.venv-gpu)
source scripts/LegalMT/env.sh            # selects it; outputs go to ../legalmt-artifacts (LEGALMT_ARTIFACTS_ROOT overrides)
uv run --group gpu hf download tencent/Hy-MT2-1.8B --revision 9a341cd1b679d3efd23b46e847b01745a71ed792   # base model for training
```

## Data

```
data/raw/Dataset/train/en-vi-train.csv
data/raw/Dataset/public test/en-vi-public_test.csv        # the folder name contains a space
```

## Train

```bash
uv run python scripts/LegalMT/build_aligned.py --config align_v1
CUDA_VISIBLE_DEVICES="" uv run --group gpu python scripts/LegalMT/build_sft.py --config configs/LegalMT/sft/sft_v1_final.yaml
scripts/LegalMT/gpu_run.sh scripts/LegalMT/train_lora.py --config final_d3 --set data.sft=sft_v1_final     # --plan-only previews steps and tokens
CUDA_VISIBLE_DEVICES="" uv run --group gpu python scripts/LegalMT/merge_lora.py --adapter <run name>        # merges the LoRA adapter into the base model
export MODEL=../legalmt-artifacts/merged/<run name>                                                         # the trained model used below
```

`<run name>` is the folder created under `../legalmt-artifacts/checkpoints/`. `final_d3b` is the same recipe without LoRA+.
The `final_*` configs default to the builds `sft_v1_mtetA_final` and `sft_v1_mtetQ_final`, which also read public MTet legal pairs from `../legalmt-artifacts/mtet_2026-09-26/mtet_legal_clean*.jsonl` (files not in this repository); the commands above train on the organizers' data only.

## Translate

```bash
scripts/LegalMT/gpu_run.sh scripts/LegalMT/translate.py --config hymt2_final_graphs --set model.local_path=$MODEL \
    --set backend.kv_cache_memory_bytes=3221225472 --set backend.gpu_memory_utilization=0.35 \
    --input-file in.md --output-file out.md
```

The direction (en to vi or vi to en) is detected; `--src-lang en|vi` forces it. Load time, translation time and words per second are printed. The two memory options are the Docker image's settings (3 GiB KV cache, peak about 7.4 GB). Scoring on a split: replace the file options with `--split public_test --evaluate`.

## Docker

```bash
docker/build_image.sh --serve-config configs/LegalMT/serve.final.yaml --model-dir $MODEL --no-adapter
docker run -d --gpus all --shm-size 2g -p 8000:8000 <image>       # <image> is printed by the build; GET /health reports readiness
docker/translate.sh localhost:8000 in.md out.md                     # prints run time and words per second
```

The image runs offline.

## Tests

```bash
uv run pytest        # CPU; tests that need the data or the GPU libraries skip
```
