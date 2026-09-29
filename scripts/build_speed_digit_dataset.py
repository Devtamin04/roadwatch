"""Build the speed-digit classification dataset from a YOLO-format sign dataset.

Input: one or more Roboflow "YOLOv11" exports of VR-TSD (a folder with data.yaml
and train/valid/test/{images,labels}). Output: 64x64 PNG crops in
<out>/{train,val,test}/<label>/ plus <out>/index.csv, where <label> is the
speed value ("20", ..., "100") or "other".

- Speed-limit crops come from classes named like "127-toc-do-toi-da-50"; the
  crop geometry is roadwatch.perception.speed_digits.crop_sign (same as runtime).
- "other" = crops of every other sign class + random background boxes, so the
  classifier learns to reject non-speed signs the detector mislabels.
- Splits are re-made by *source image*: Roboflow names augmented copies
  "<source>_jpg.rf.<hash>.jpg"; all copies of a source land in the same split.
- --synthetic N adds N rendered signs per value to the train split only.

    .venv/bin/python scripts/build_speed_digit_dataset.py --yolo-dataset ~/Downloads/vr-tsd \\
        --out data/speed_digits --synthetic 150
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.perception.speed_digits import OTHER, crop_sign  # noqa: E402

SPEED_RE = re.compile(r"toc-do-toi-da-(\d{2,3})$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def source_key(image_path: Path) -> str:
    """Group key: Roboflow augmented copies share the part before '.rf.'."""
    stem = image_path.stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def assign_split(key: str) -> str:
    """Deterministic 80/10/10 split from a hash of the source key."""
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < SPLITS["train"] else "val" if u < SPLITS["train"] + SPLITS["val"] else "test"


def read_names(dataset: Path) -> list[str]:
    with open(dataset / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    return list(names) if isinstance(names, list) else [names[k] for k in sorted(names)]


def iter_samples(dataset: Path):
    """Yield (image_path, [(class_id, cx, cy, w, h) normalised])."""
    for img_dir in sorted(dataset.glob("*/images")):
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = img_dir.parent / "labels" / f"{img.stem}.txt"
            rows = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:  # polygon rows (segmentation exports) are skipped
                        rows.append((int(parts[0]), *map(float, parts[1:5])))
            yield img, rows


def render_synthetic(value: int, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """A rendered Vietnamese-style speed-limit sign (red ring, black digits) on noise."""
    big = 192
    img = np.full((big, big, 3), rng.integers(40, 220, 3), np.uint8)
    img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    c, r = big // 2, int(big * rng.uniform(0.36, 0.44))
    cv2.circle(img, (c, c), r, (250, 250, 250), -1, cv2.LINE_AA)
    red = (int(rng.integers(0, 40)), int(rng.integers(0, 40)), int(rng.integers(170, 240)))
    cv2.circle(img, (c, c), r, red, int(r * 0.2), cv2.LINE_AA)
    text = str(value)
    font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX][int(rng.integers(0, 2))]
    scale = (1.9 if len(text) == 2 else 1.45) * r / 70
    thick = max(2, int(scale * 3))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (c - tw // 2, c + th // 2), font, scale, (15, 15, 15), thick, cv2.LINE_AA)
    # Perspective / rotation / blur / brightness jitter.
    src = np.float32([[0, 0], [big, 0], [big, big], [0, big]])
    dst = src + rng.uniform(-big * 0.06, big * 0.06, src.shape).astype(np.float32)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (big, big),
                              borderMode=cv2.BORDER_REFLECT)
    rot = cv2.getRotationMatrix2D((c, c), rng.uniform(-10, 10), 1.0)
    img = cv2.warpAffine(img, rot, (big, big), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.6:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.6, 1.3), beta=rng.uniform(-30, 30))
    return crop_sign(img, (c - r, c - r, c + r, c + r), 0.10, size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yolo-dataset", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--expand", type=float, default=0.10)
    ap.add_argument("--other-per-image", type=int, default=1,
                    help="max non-speed sign crops taken per image")
    ap.add_argument("--background-per-image", type=float, default=0.3,
                    help="expected random background crops per image")
    ap.add_argument("--synthetic", type=int, default=0, help="rendered signs per value (train)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows: list[tuple[str, str, str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    values_seen: set[int] = set()

    def save(crop: np.ndarray, split: str, label: str, key: str) -> None:
        d = args.out / split / label
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{label}_{len(rows):06d}.png"
        cv2.imwrite(str(path), crop)
        rows.append((str(path.relative_to(args.out)), label, split, key))
        counts[(split, label)] += 1

    for dataset in args.yolo_dataset:
        names = read_names(dataset)
        speed_of = {i: int(m.group(1)) for i, n in enumerate(names) if (m := SPEED_RE.search(n))}
        if not speed_of:
            print(f"{dataset}: no speed-limit classes found in data.yaml", file=sys.stderr)
            return 1
        values_seen.update(speed_of.values())
        for img_path, labels in iter_samples(dataset):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            key = f"{dataset.name}/{source_key(img_path)}"
            split = assign_split(key)
            others = []
            for cid, cx, cy, bw, bh in labels:
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                if min(box[2] - box[0], box[3] - box[1]) < 8:
                    continue  # too small to read
                if cid in speed_of:
                    save(crop_sign(img, box, args.expand, args.size), split, str(speed_of[cid]), key)
                else:
                    others.append(box)
            for i in rng.permutation(len(others))[: args.other_per_image]:
                save(crop_sign(img, others[i], args.expand, args.size), split, OTHER, key)
            if rng.random() < args.background_per_image:
                s = rng.uniform(0.04, 0.2) * min(h, w)
                x, y = rng.uniform(0, w - s), rng.uniform(0, h - s)
                save(crop_sign(img, (x, y, x + s, y + s), 0.0, args.size), split, OTHER, key)

    for value in sorted(values_seen):
        for _ in range(args.synthetic):
            save(render_synthetic(value, rng, args.size), "train", str(value), "synthetic")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("path", "label", "split", "group"), *rows])

    groups: dict[str, set[str]] = defaultdict(set)
    for _, _, split, key in rows:
        groups[key].add(split)
    leaks = [k for k, s in groups.items() if k != "synthetic" and len(s) > 1]
    if leaks:
        raise SystemExit(f"split leak for {leaks[:3]}")

    labels = sorted({lbl for _, lbl in counts},
                    key=lambda s: (s == OTHER, int(s) if s.isdigit() else 0))
    print(f"{'label':>6s} " + " ".join(f"{s:>6s}" for s in SPLITS))
    for lbl in labels:
        real_train = counts[("train", lbl)] - (args.synthetic if lbl != OTHER else 0)
        flag = "  <-- fewer than 50 real train crops" if lbl != OTHER and real_train < 50 else ""
        print(f"{lbl:>6s} " + " ".join(f"{counts[(s, lbl)]:6d}" for s in SPLITS) + flag)
    print(f"\n{len(rows)} crops written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
