"""Train the speed-digit classifier on a dataset from build_speed_digit_dataset.py.

Export-venv script (needs torch; CPU is enough, the model is ~0.2M params):

    .venv-export/bin/python scripts/train_speed_digits.py --data data/speed_digits
    # -> models/speed_digits_64.onnx + models/speed_digits_64_classes.json (+ manifest entry)

Normalisation and crop layout come from roadwatch.perception.speed_digits, so the
exported model sees exactly what the runtime feeds it. Prints per-class accuracy
and the confusion matrix on the test split (watch 50/60, 30/80, 100).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.manifest import sha256_file, upsert_manifest_entry  # noqa: E402
from roadwatch.perception.speed_digits import OTHER, to_tensor  # noqa: E402

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def load_split(data: Path, split: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with open(data / "index.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    imgs = [cv2.imread(str(data / r["path"])) for r in rows]
    labels = np.array([classes.index(r["label"]) for r in rows], np.int64)
    return np.stack(imgs) if imgs else np.zeros((0, 64, 64, 3), np.uint8), labels


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric + small geometric jitter (no flips: digits are not symmetric)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-0.06, 0.06, 2) * (w, h)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.2))
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.65, 1.35), beta=rng.uniform(-35, 35))
    if rng.random() < 0.3:  # low-resolution / distant sign
        s = int(rng.integers(16, 40))
        img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), (w, h))
    return img


def make_model(n_classes: int) -> nn.Module:
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                         nn.Linear(128, n_classes))


def evaluate(model, x: np.ndarray, y: np.ndarray, n: int) -> tuple[float, np.ndarray]:
    model.eval()
    conf = np.zeros((n, n), np.int64)
    with torch.no_grad():
        for i in range(0, len(x), 256):
            pred = model(torch.from_numpy(to_tensor(list(x[i:i + 256]), MEAN, STD))).argmax(1).numpy()
            np.add.at(conf, (y[i:i + 256], pred), 1)
    return float(np.trace(conf) / max(1, conf.sum())), conf


def print_confusion(conf: np.ndarray, classes: list[str]) -> None:
    print("       " + " ".join(f"{c:>5s}" for c in classes) + "   acc")
    for i, c in enumerate(classes):
        acc = conf[i, i] / max(1, conf[i].sum())
        print(f"{c:>6s} " + " ".join(f"{v:5d}" for v in conf[i]) + f"  {acc:5.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--name", default="speed_digits_64")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with open(args.data / "index.csv", encoding="utf-8") as f:
        labels = {r["label"] for r in csv.DictReader(f)}
    classes = sorted((c for c in labels if c != OTHER), key=int) + ([OTHER] if OTHER in labels else [])
    x_tr, y_tr = load_split(args.data, "train", classes)
    x_va, y_va = load_split(args.data, "val", classes)
    x_te, y_te = load_split(args.data, "test", classes)
    print(f"classes {classes}; train {len(y_tr)}, val {len(y_va)}, test {len(y_te)}")
    if len(y_tr) == 0 or len(y_va) == 0:
        raise SystemExit("empty train or val split")

    model = make_model(len(classes))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k")
    counts = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights / weights.mean(), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(y_tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(y_tr))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = torch.from_numpy(to_tensor([augment(x_tr[j], rng) for j in idx], MEAN, STD))
            yb = torch.from_numpy(y_tr[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        val_acc, _ = evaluate(model, x_va, y_va, len(classes))
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {total / len(order):.4f}  val acc {val_acc:6.2%}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    split_name, x_ev, y_ev = ("test", x_te, y_te) if len(y_te) else ("val", x_va, y_va)
    acc, conf = evaluate(model, x_ev, y_ev, len(classes))
    print(f"\nbest val acc {best_acc:.2%}; {split_name} acc {acc:.2%}")
    print_confusion(conf, classes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / f"{args.name}.onnx"
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, 64, 64), str(onnx_path), opset_version=12,
                      input_names=["images"], output_names=["logits"], dynamo=False)
    meta = {"classes": classes, "input_size": 64, "mean": list(MEAN), "std": list(STD),
            "expand": 0.10, f"{split_name}_acc": round(acc, 4), "val_acc": round(best_acc, 4)}
    (args.out_dir / f"{args.name}_classes.json").write_text(json.dumps(meta, indent=2))
    upsert_manifest_entry(args.out_dir / "manifest.json", {
        "name": args.name, "file": onnx_path.name, "sha256": sha256_file(onnx_path),
        "input_shape": [1, 3, 64, 64], "output_shape": [1, len(classes)],
        "source": f"scripts/train_speed_digits.py on {args.data.name} "
                  f"({len(y_tr)} train crops, seed {args.seed})",
        "classes": classes,
        "notes": f"{split_name} acc {acc:.4f}; RGB/255, mean/std 0.5; crop = box +10% per side",
    })
    print(f"exported {onnx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
