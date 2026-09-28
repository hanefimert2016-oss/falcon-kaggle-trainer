#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), check=True)


def valid_revision(path: Path, pipeline: str) -> bool:
    meta = path / "sources.json"
    if not meta.is_file():
        return False
    try:
        obj = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return obj.get("pipeline_version") == pipeline and int(obj.get("data_revision", 0)) >= 2


def ensure_dataset(ref: str, dest: Path, pipeline: str) -> None:
    if valid_revision(dest, pipeline):
        print("dataset_ready", ref, dest, flush=True)
        return
    if not (os.environ.get("KAGGLE_API_TOKEN") or Path.home().joinpath(".kaggle/kaggle.json").is_file()):
        raise SystemExit(
            "Kaggle kimlik bilgisi gerekli: Colab secret/env KAGGLE_API_TOKEN "
            "veya ~/.kaggle/kaggle.json tanimla. Bu sadece hazir r2 datasetini indirmek icindir."
        )
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "-m", "kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip")
    if not valid_revision(dest, pipeline):
        raise RuntimeError(f"downloaded dataset is not FLM v0.7 r2: {ref}")


def gpu_profile() -> tuple[int, int, int, int]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Colab GPU acik degil. Runtime > Change runtime type > GPU sec.")
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    min_gib = min(int(p.total_memory // 1024**3) for p in props)
    print("GPUs:", [(p.name, round(p.total_memory / 1024**3, 2)) for p in props], flush=True)

    if min_gib >= 40:
        return 4096, 8, 4, 8
    if min_gib >= 22:
        return 4096, 4, 8, 4
    if min_gib >= 14:
        return 4096, 1, 32, 2
    return 2048, 1, 32, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "main", "coder", "computer_use"], default="all")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/FalconFLM")
    ap.add_argument("--data-root", default="/content/flm-v07-r2-data")
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", "mertsigma"))
    ap.add_argument("--skip-drive-mount", action="store_true")
    args = ap.parse_args()

    if not args.skip_drive_mount:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            if not Path("/content/drive/MyDrive").exists():
                raise SystemExit(f"Google Drive mount failed: {exc}")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))

    run(sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "kaggle>=2.2.1", "tokenizers>=0.20", "numpy>=1.26", "pillow>=10")

    data_root = Path(args.data_root)
    text_dir = data_root / "text"
    computer_dir = data_root / "computer"
    ensure_dataset(f"{args.owner}/flm-v07-dev-text", text_dir, "v0.7-dev-text")
    ensure_dataset(f"{args.owner}/flm-v07-dev-computer", computer_dir, "v0.7-dev-computer")

    train_seq, batch, accum, cu_batch = gpu_profile()
    drive_root = Path(args.drive_root)
    out = drive_root / "runs" / "flm-v0.7-r2"
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root),
        "FLM_ACCELERATOR": "gpu",
        "FLM_V07_DATA_ROOT": str(data_root),
        "FLM_V07_TEXT_VERSION": "v0.7-dev-text",
        "FLM_V07_COMPUTER_VERSION": "v0.7-dev-computer",
        "FLM_V07_OUTPUT_ROOT": str(out),
        "FLM_V07_RESUME": "1",
        "FLM_V07_MAIN_SEQ": "4096",
        "FLM_V07_CODER_SEQ": "4096",
        "FLM_V07_MAIN_TRAIN_SEQ": str(train_seq),
        "FLM_V07_CODER_TRAIN_SEQ": str(train_seq),
        "FLM_V07_MAIN_LAYERS": "14",
        "FLM_V07_CODER_LAYERS": "14",
        "FLM_V07_MAIN_HEADS": "12",
        "FLM_V07_CODER_HEADS": "12",
        "FLM_V07_MAIN_EMBD": "768",
        "FLM_V07_CODER_EMBD": "768",
        "FLM_V07_POSITION_ENCODING": "rope",
        "FLM_V07_CODER_INIT_FROM_MAIN": "1",
        "FLM_V07_CU_INIT_FROM_MAIN": "1",
        "FLM_V07_TEXT_BATCH": str(batch),
        "FLM_V07_TEXT_ACCUM": str(accum),
        "FLM_V07_GRADIENT_CHECKPOINTING": "1" if train_seq < 4096 or batch <= 1 else "0",
        "FLM_V07_MAIN_STEPS": "0",
        "FLM_V07_MAIN_SFT_STEPS": "0",
        "FLM_V07_CODER_STEPS": "0",
        "FLM_V07_CODER_SFT_STEPS": "0",
        "FLM_V07_MAIN_PRETRAIN_EPOCHS": "1.0",
        "FLM_V07_MAIN_SFT_EPOCHS": "1.0",
        "FLM_V07_CODER_PRETRAIN_EPOCHS": "1.0",
        "FLM_V07_CODER_SFT_EPOCHS": "1.25",
        "FLM_V07_TEXT_CHECKPOINT_INTERVAL": "250",
        "FLM_V07_TEXT_EVAL_INTERVAL": "500",
        "FLM_V07_SFT_EVAL_INTERVAL": "250",
        "FLM_V07_EVAL_BATCHES": "12",
        "FLM_V07_CU_TASK_LEN": "512",
        "FLM_V07_CU_BATCH": str(cu_batch),
        "FLM_V07_CU_STEPS": "30000",
        "FLM_V07_CU_EVAL_EXAMPLES": "512",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    print(json.dumps({
        "mode": args.mode,
        "data_root": str(data_root),
        "checkpoint_root": str(out),
        "model_context": 4096,
        "training_seq": train_seq,
        "text_batch": batch,
        "text_accum": accum,
        "computer_batch": cu_batch,
        "resume": True,
    }, indent=2), flush=True)

    subprocess.run(
        [sys.executable, "-u", "-m", "flm.train_v07", "--only", args.mode],
        check=True,
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
