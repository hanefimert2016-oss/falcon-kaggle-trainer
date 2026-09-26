#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
if (HERE / "flm").is_dir():
    sys.path.insert(0, str(HERE))
elif (HERE.parent / "flm").is_dir():
    sys.path.insert(0, str(HERE.parent))

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.computer_use_v2 import ComputerUseV2, ComputerUseV2Config

OPS = ["CLICK","TYPE","KEY","SCROLL","MOVE","DRAG","GAME_ACTION","OTHER"]


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA istendi ama CUDA destekli GPU bulunamadı.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text(root: Path, kind: str, device: torch.device):
    ckpt = torch.load(root / kind / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = TextConfig(**ckpt["config"])
    model = ByteCausalLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tok.get_vocab_size() != cfg.vocab_size:
        raise RuntimeError(f"tokenizer/model vocab mismatch: {tok.get_vocab_size()} != {cfg.vocab_size}")
    return model, cfg, tok


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, generator):
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / max(1e-5, temperature)
    if top_k > 0:
        k = min(top_k, logits.numel())
        vals, inds = torch.topk(logits, k)
        probs = torch.softmax(vals, dim=-1)
        return int(inds[torch.multinomial(probs, 1, generator=generator)])
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_main(root: Path, prompt: str, device, max_new: int, temperature: float, top_k: int):
    model, cfg, tok = load_text(root, "main", device)
    bos = tok.token_to_id("<bos>")
    user = tok.token_to_id("<|user|>")
    assistant = tok.token_to_id("<|assistant|>")
    end = tok.token_to_id("<|end|>")
    eos = tok.token_to_id("<eos>")
    if None in (bos, user, assistant, end, eos):
        raise RuntimeError("v0.6 tokenizer special tokenları eksik")
    prefix = [bos, user] + tok.encode(prompt, add_special_tokens=False).ids + [assistant]
    ids = list(prefix)
    generator = torch.Generator(device=device)
    generator.manual_seed(606)
    for _ in range(max_new):
        x = torch.tensor([ids[-cfg.seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = sample_token(logits[0, -1], temperature, top_k, generator)
        if nxt in (end, eos):
            break
        ids.append(nxt)
    return tok.decode(ids[len(prefix):], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_coder(root: Path, prompt: str, device, max_new: int, temperature: float, top_k: int):
    model, cfg, tok = load_text(root, "coder", device)
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    prefix = ([bos] if bos is not None else []) + tok.encode(prompt, add_special_tokens=False).ids
    ids = list(prefix)
    generator = torch.Generator(device=device)
    generator.manual_seed(607)
    for _ in range(max_new):
        x = torch.tensor([ids[-cfg.seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = sample_token(logits[0, -1], temperature, top_k, generator)
        if eos is not None and nxt == eos:
            break
        ids.append(nxt)
    return tok.decode(ids[len(prefix):], skip_special_tokens=True)


def task_tensor(text: str, cfg, device):
    raw = list(text.encode("utf-8", "ignore")[:cfg.task_len])
    raw += [cfg.pad_token] * (cfg.task_len - len(raw))
    return torch.tensor([raw], dtype=torch.long, device=device)


def image_tensor(path: Path, cfg, device):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((cfg.image_size, cfg.image_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).contiguous().to(device)


@torch.no_grad()
def run_computer_use(root: Path, image_path: Path, task: str, device):
    ckpt = torch.load(root / "computer_use" / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = ComputerUseV2Config(**ckpt["config"])
    model = ComputerUseV2(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    domains = ckpt.get("domains", {})
    id_to_domain = {v:k for k,v in domains.items()}
    image = image_tensor(image_path, cfg, device)
    task_ids = task_tensor(task, cfg, device)
    action = [cfg.bos_token]
    last = None
    for pos in range(cfg.action_len):
        seq = action[:cfg.action_len] + [cfg.pad_token] * max(0, cfg.action_len-len(action))
        action_in = torch.tensor([seq], dtype=torch.long, device=device)
        out, _, _ = model(image, task_ids, action_in)
        last = out
        token = int(out["action_logits"][0, pos].argmax())
        if token == cfg.eos_token:
            break
        if token >= 256:
            break
        action.append(token)
    if last is None:
        raise RuntimeError("ComputerUse action decoder produced no output")
    op_id = int(last["op_logits"][0].argmax())
    domain_id = int(last["domain_logits"][0].argmax())
    result = {
        "operation": OPS[op_id] if op_id < len(OPS) else str(op_id),
        "domain": id_to_domain.get(domain_id, f"id:{domain_id}"),
        "coord": [round(float(x), 5) for x in last["coord"][0].cpu()],
        "bbox": [round(float(x), 5) for x in last["bbox"][0].cpu()],
        "action": bytes([x for x in action[1:] if x < 256]).decode("utf-8", "replace"),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Falcon FLM v0.6 inference")
    ap.add_argument("--root", default=".", help="FLM v0.6 package root")
    ap.add_argument("--model", choices=("main","coder","computer_use"), required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--image")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    device = choose_device(args.device)
    print(f"device: {device}")

    if args.model == "main":
        print(generate_main(root, args.prompt, device, args.max_new, args.temperature, args.top_k))
    elif args.model == "coder":
        print(generate_coder(root, args.prompt, device, args.max_new, args.temperature, args.top_k))
    else:
        if not args.image:
            raise SystemExit("computer_use için --image gerekli")
        result = run_computer_use(root, Path(args.image), args.prompt, device)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
