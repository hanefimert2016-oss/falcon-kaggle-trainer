# Falcon FLM v0.4 — Three-Model Architecture

Falcon FLM now has three **separate, from-scratch checkpoints**:

1. **Main model** — byte-level causal language model trained on real educational web text.
2. **Computer Use model** — screenshot vision encoder + task encoder + action decoder for GUI operation prediction.
3. **Coder model** — separate byte-level causal language model trained on real code + natural-language programming data.

No pretrained Qwen/Llama/VLM weights are required by this architecture.

## Real Hugging Face data

The CPU data stage samples:

- `codelion/fineweb-edu-100M` → main model
- `Nan-Do/code-search-net-python` → coder model
- `markov-ai/computer-use` → computer-use vision/action model

The exact source IDs, licenses, sample counts and prepared sizes are saved in `sources.json`.

## CPU → accelerator split

```text
GitHub Actions CPU
  ├─ Hugging Face download
  ├─ text/code filtering
  ├─ screenshot JPEG preprocessing
  └─ private Kaggle dataset: flm-hf-v04
                    |
                    v
Kaggle GPU or TPU
  ├─ main model train + heldout eval
  ├─ computer_use vision model train + heldout eval
  └─ coder model train + heldout eval
                    |
                    v
       three separate checkpoint.pt files
```

CPU preparation explicitly runs with `CUDA_VISIBLE_DEVICES=""`. The Kaggle bundle explicitly sets `FLM_ACCELERATOR=gpu` for Nvidia accelerators and `FLM_ACCELERATOR=tpu` for TPU accelerators.

## Computer Use vision model

Input:

- screenshot
- natural-language task

Output:

- operation class: `CLICK | TYPE | KEY | SCROLL | MOVE | OTHER`
- byte-level executable action text

The vision tower is a from-scratch patch encoder using convolutional patchification + Transformer layers. It is fused with a byte-level task encoder and a recurrent action decoder.

## Workflows

- **FLM HF Data CPU Prep** — downloads/prepares real HF data on CPU and versions the private Kaggle dataset.
- **FLM Three Model Accelerator Train** — trains/evaluates all three checkpoints on T4 GPU or TPU.
- **Falcon CI** — syntax, YAML, GPU/TPU bundle generation, language regression, and vision computer-use regression.

## Accelerator options

The launcher supports:

- `NvidiaTeslaT4`
- `TpuV5E8`
- `TpuV6E8`

TPU execution uses the `torch_xla` path in `flm/runtime.py`; GPU execution uses CUDA mixed precision and uses multiple visible GPUs through `DataParallel` where applicable.

## Current scope

The default `FLM_SMOKE=1` profile is an architecture/data/accelerator validation run, not a fully converged production model. Increase model dimensions, data shard size and step counts only after the verified smoke pipeline is clean.
