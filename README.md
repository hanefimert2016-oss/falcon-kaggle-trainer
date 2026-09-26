# Falcon FLM Trainer v0.3

GitHub Actions is the control plane. The same from-scratch FLM training engine can now run on **Kaggle GPU** or **Google Colab GPU**.

## Backends

### Kaggle — automatic backend

- GitHub Actions authenticates with `KAGGLE_API_TOKEN`.
- GitHub can push the trainer, query status/logs/quota, and download outputs.
- Tested successfully on **2× Tesla T4**.
- Outputs include `checkpoint.pt`, `metrics.json`, and `runtime.json`.

Required repository configuration:

- Secret: `KAGGLE_API_TOKEN`
- Variable: `KAGGLE_OWNER`
- Optional variable: `KAGGLE_KERNEL_SLUG`

### Google Colab — interactive GPU backend

Open:

https://colab.research.google.com/github/hanefimert2016-oss/falcon-kaggle-trainer/blob/main/colab/Falcon_FLM_Trainer.ipynb

The notebook:

1. mounts Google Drive,
2. clones/updates this GitHub repo,
3. detects CUDA and the assigned GPU,
4. uses the same `kaggle/train_flm.py` engine,
5. reads `.txt` training data from `MyDrive/FalconFLM/data/`,
6. saves checkpoints to `MyDrive/FalconFLM/runs/colab-v0.3/`.

No pretrained Llama/Qwen model or pretrained tokenizer is required.

## Portable trainer paths

The shared trainer supports:

- `FLM_DATA_DIR` — input directory containing `.txt` files.
- `FLM_OUTPUT_DIR` — directory for checkpoint and metrics.

Defaults are selected automatically for Kaggle, Colab, or a local machine.

## GitHub Actions

The repository also contains:

- Kaggle training control workflows,
- Kaggle auth/status/output verification,
- single-CPU Actions test,
- CI that validates Python, YAML, the Colab notebook, Colab setup, model forward/backward, and a full 1-step CPU training run.

## Local smoke test

```bash
./setup.sh
export FLM_DATA_DIR="$PWD/data"
export FLM_OUTPUT_DIR="$PWD/outputs/local"
python kaggle/train_flm.py
```
