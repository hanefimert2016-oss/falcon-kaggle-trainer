#!/usr/bin/env python3
"""Turn a low-res LingBot clip into a 1280x720 / 16 FPS display stream."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--fps", type=float, default=16.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4))
    cap.release()
    if not frames:
        raise SystemExit("No frames read")

    duration = (len(frames) - 1) / src_fps if len(frames) > 1 else 0
    out_count = max(1, int(round(duration * args.fps)) + 1)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    for oi in range(out_count):
        t = oi / args.fps
        pos = t * src_fps
        i0 = min(int(pos), len(frames) - 1)
        i1 = min(i0 + 1, len(frames) - 1)
        alpha = max(0.0, min(1.0, pos - i0))
        out = cv2.addWeighted(frames[i0], 1.0 - alpha, frames[i1], alpha, 0.0)
        writer.write(out)
    writer.release()
    print(f"wrote {args.output} ({out_count} frames @ {args.fps} FPS)")

if __name__ == "__main__":
    main()
