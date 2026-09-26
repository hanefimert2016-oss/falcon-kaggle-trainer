# Falcon Kaggle Trainer v0.2

GitHub Actions is the control plane; Kaggle is the GPU training backend.

## What this repo does

- Uses GitHub Actions to validate, start, inspect, and download Kaggle training runs.
- Uses the official Kaggle CLI and non-interactive `KAGGLE_API_TOKEN` authentication.
- Runs a from-scratch byte-level causal language model on Kaggle GPU compute.
- Does not depend on Llama, Qwen, or another pretrained model/tokenizer.
- Saves `checkpoint.pt` and `metrics.json` as Kaggle outputs.

## Required GitHub configuration

Repository secret:

- `KAGGLE_API_TOKEN` — create/copy it from Kaggle Settings → API.

Repository variables:

- `KAGGLE_OWNER` — your Kaggle username/owner slug.
- `KAGGLE_KERNEL_SLUG` — optional, defaults to `falcon-flm-train`.

Do not commit your Kaggle token to this repository.

## Run from GitHub

Open **Actions → Falcon Kaggle Train → Run workflow**.

Actions:

- `push`: upload the current trainer to Kaggle and start a GPU run.
- `status`: show current kernel status.
- `logs`: print the latest Kaggle logs.
- `output`: download Kaggle outputs and publish them as a GitHub artifact.
- `quota`: show Kaggle accelerator quota.

Default accelerator: `NvidiaTeslaT4` (Kaggle T4 x2 where available).

## Local smoke test

```bash
./setup.sh
export KAGGLE_API_TOKEN='...'
export KAGGLE_OWNER='your-kaggle-name'
python falconctl.py render
python falconctl.py validate
python falconctl.py quota
```

## Architecture

```text
ChatGPT / GitHub UI
        |
        v
GitHub Actions
        |
        v
falconctl.py + Kaggle CLI
        |
        v
Kaggle GPU kernel
        |
        +--> checkpoint.pt
        +--> metrics.json
```
