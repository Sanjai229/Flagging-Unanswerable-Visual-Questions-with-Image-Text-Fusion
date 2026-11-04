"""Train every model with one shared loop, so differences come only from the inputs and the fusion method.

Usage (from the repo root, e.g. in Colab):
    python -m vizwiz.train --config configs/default.yaml --models all
    python -m vizwiz.train --models gated cross_attention --seeds 0

Each run writes <runs_dir>/<model>/seed<k>/:
    predictions.csv   pair_id, split, model, seed, y, p_unanswerable for dev and test
                      (+ gate_q for gated fusion; + p_no_question, p_no_image for fusion models)
    history.csv       epoch, train_loss, dev_loss, dev_auprc, seconds (for overfitting curves)
    summary.json      settings, parameters, epochs, best epoch, time per epoch, peak GPU memory, head latency
    model.pt          best weights (selected on dev AUPRC)
The test split is never used for training, early stopping, thresholds or weights.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vizwiz.data import load_meta, load_store, modality
from vizwiz.models import build_model, model_inputs

FLOORS = ["majority", "photo_quality"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run_dir(runs_dir, name, seed):
    d = Path(runs_dir) / name / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_finished(runs_dir, name, seed):
    d = Path(runs_dir) / name / f"seed{seed}"
    return (d / "predictions.csv").exists() and (d / "summary.json").exists()


@torch.no_grad()
def predict(model, store, idx, device, blank=None, bs=1024):
    """P(unanswerable) for the rows in idx (and the mean gate value, for gated fusion)."""
    model.eval()
    probs, gates = [], []
    for i in range(0, len(idx), bs):
        logits = model(store.batch(idx[i:i + bs], model.needs, device, blank=blank))
        probs.append(torch.sigmoid(logits).float().cpu())
        if getattr(model, "last_gate_q", None) is not None:
            gates.append(model.last_gate_q.float().cpu())
    return torch.cat(probs).numpy(), (torch.cat(gates).numpy() if gates else None)


@torch.no_grad()
def head_latency_ms(model, store, idx, device, runs):
    """Median time of one forward pass for a single pair (features already extracted)."""
    model.eval()
    times = []
    for k in range(runs + 5):
        b = store.batch(idx[k % len(idx):k % len(idx) + 1], model.needs, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(b)
        if device == "cuda":
            torch.cuda.synchronize()
        if k >= 5:
            times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 3)


def train_one(store, spec, name, seed, tcfg, runs_dir, device):
    """Train one model with one seed, early-stop on dev AUPRC, and save everything needed downstream."""
    set_seed(seed)
    model = build_model(spec, store.dims()).to(device)
    tr, dev, test = store.index("train"), store.index("dev"), store.index("test")
    y = store.meta["y"].values
    target = store.target(spec.get("label", "hard"))

    pos = float(y[tr].mean())  # class imbalance: weight the minority (unanswerable) class
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    rng = np.random.default_rng(seed)
    probe = rng.choice(tr, size=min(len(tr), tcfg["train_probe_size"]), replace=False)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_ap, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // tcfg["batch_size"])):
            loss = loss_fn(model(store.batch(b, model.needs, device)), target[torch.as_tensor(b)].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        p_probe, _ = predict(model, store, probe, device)
        p_dev, _ = predict(model, store, dev, device)
        ap = float(average_precision_score(y[dev], p_dev))  # selection metric (Saito & Rehmsmeier 2015)
        history.append({"epoch": epoch, "train_loss": log_loss(p_probe, y[probe]),
                        "dev_loss": log_loss(p_dev, y[dev]), "dev_auprc": ap, "seconds": round(secs, 2)})
        if ap > best_ap:
            best_ap, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None

    out = run_dir(runs_dir, name, seed)
    torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    modalities = {modality(k) for k in model.needs} - {None}
    frames = []
    for split, idx in (("dev", dev), ("test", test)):
        p, gate = predict(model, store, idx, device)
        f = pd.DataFrame({"pair_id": store.meta["pair_id"].values[idx], "split": split, "model": name,
                          "seed": seed, "y": y[idx], "p_unanswerable": p})
        if gate is not None:
            f["gate_q"] = gate
        if len(modalities) == 2:  # fusion model: also score it with one input missing
            f["p_no_question"] = predict(model, store, idx, device, blank="question")[0]
            f["p_no_image"] = predict(model, store, idx, device, blank="image")[0]
        frames.append(f)
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)

    best = history[best_epoch - 1]
    summary = {"model": name, "seed": seed, "type": spec["type"], "group": spec.get("group"),
               "label": spec.get("label", "hard"), "inputs": model_inputs(spec),
               "hyperparameters": {k: v for k, v in spec.items() if k not in ("type", "group", "label")},
               "training": {k: tcfg[k] for k in ("epochs", "batch_size", "lr", "weight_decay", "patience")},
               "params": int(sum(p.numel() for p in model.parameters())),
               "epochs_run": len(history), "best_epoch": best_epoch, "best_dev_auprc": round(best_ap, 4),
               "train_loss_at_best": round(best["train_loss"], 4), "dev_loss_at_best": round(best["dev_loss"], 4),
               "sec_per_epoch": round(float(np.mean([h["seconds"] for h in history])), 2),
               "peak_vram_gb": peak,
               "head_latency_ms": head_latency_ms(model, store, dev, device, tcfg.get("latency_runs", 50))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name} seed {seed}] {len(history)} epochs (best {best_epoch}), dev AUPRC {best_ap:.4f}, "
          f"{summary['sec_per_epoch']} s/epoch, {summary['params']:,} params")
    return out


def late_fusion(runs_dir, name, spec, seed):
    """w * P(member 1) + (1 - w) * P(member 2), with w chosen on dev AUPRC from a grid of 21 values."""
    a, b = spec["members"]
    pa = pd.read_csv(Path(runs_dir) / a / f"seed{seed}" / "predictions.csv")
    pb = pd.read_csv(Path(runs_dir) / b / f"seed{seed}" / "predictions.csv")
    m = pa.merge(pb[["pair_id", "split", "p_unanswerable"]], on=["pair_id", "split"], suffixes=("_a", "_b"))
    dev = m[m.split == "dev"]
    grid = np.linspace(0, 1, 21)
    scores = [average_precision_score(dev.y, w * dev.p_unanswerable_a + (1 - w) * dev.p_unanswerable_b) for w in grid]
    w = float(grid[int(np.argmax(scores))])
    out = run_dir(runs_dir, name, seed)
    pd.DataFrame({"pair_id": m.pair_id, "split": m.split, "model": name, "seed": seed, "y": m.y,
                  "p_unanswerable": w * m.p_unanswerable_a + (1 - w) * m.p_unanswerable_b}
                 ).to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"model": name, "seed": seed, "type": "late", "group": spec.get("group"), "members": [a, b],
         "weight_on_first_member": w, "best_dev_auprc": round(float(max(scores)), 4), "params": 1,
         "epochs_run": 0, "sec_per_epoch": 0.0, "peak_vram_gb": None}, indent=2))
    print(f"[{name} seed {seed}] weight on {a}: {w:.2f}, dev AUPRC {max(scores):.4f}")


def fit_floors(meta, runs_dir):
    """Two models without a neural network, as floors: the majority class, and photo quality only
    (logistic regression on log sharpness, brightness, log megapixels, log file size). Both are deterministic."""
    idx = {s: np.where(meta["split"].values == s)[0] for s in ("train", "dev", "test")}
    y = meta["y"].values

    def save(name, p, info):
        out = run_dir(runs_dir, name, 0)
        frames = [pd.DataFrame({"pair_id": meta["pair_id"].values[idx[s]], "split": s, "model": name, "seed": 0,
                                "y": y[idx[s]], "p_unanswerable": p[idx[s]]}) for s in ("dev", "test")]
        pd.concat(frames).to_csv(out / "predictions.csv", index=False)
        (out / "summary.json").write_text(json.dumps({"model": name, "seed": 0, "type": "floor", "group": "floor",
                                                      "epochs_run": 0, "sec_per_epoch": 0.0, **info}, indent=2))

    prior = float(y[idx["train"]].mean())
    save("majority", np.full(len(meta), prior), {"params": 0, "train_unanswerable_rate": round(prior, 4)})
    q = meta[["sharpness", "brightness", "megapixels", "file_kb"]].copy()
    q = q.fillna(q.iloc[idx["train"]].median())
    X = np.column_stack([np.log1p(q.sharpness), q.brightness, np.log(q.megapixels), np.log1p(q.file_kb)])
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    lr.fit(X[idx["train"]], y[idx["train"]].astype(int))
    save("photo_quality", lr.predict_proba(X)[:, 1],
         {"params": 5, "inputs": ["log sharpness", "brightness", "log megapixels", "log file size"]})
    print("[floors] majority and photo_quality saved")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the VizWiz answerability models")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="*", default=["all"], help="model names from the config, 'floors', or 'all'")
    ap.add_argument("--seeds", type=int, nargs="*", help="override the seeds in the config")
    ap.add_argument("--features-dir", help="override paths.features_dir")
    ap.add_argument("--runs-dir", help="override paths.runs_dir")
    ap.add_argument("--local-dir", help="override paths.local_features_dir ('none' = read features in place)")
    ap.add_argument("--epochs", type=int, help="override training.epochs (quick tests only)")
    ap.add_argument("--force", action="store_true", help="retrain runs that already have predictions")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths, tcfg = cfg["paths"], cfg["training"]
    features_dir = args.features_dir or paths["features_dir"]
    runs_dir = args.runs_dir or paths["runs_dir"]
    local = args.local_dir or paths.get("local_features_dir")
    local = None if local in (None, "none") else local
    if args.epochs:
        tcfg["epochs"] = args.epochs
    seeds = args.seeds if args.seeds else tcfg["seeds"]
    specs = cfg["models"]
    names = list(specs) + FLOORS if args.models == ["all"] else args.models
    if "floors" in names:
        names = [n for n in names if n != "floors"] + FLOORS
    unknown = [n for n in names if n not in specs and n not in FLOORS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(specs) + FLOORS}")

    # Trainable models first, then late fusion (needs its members), then the floors
    trainable = [n for n in names if n in specs and specs[n]["type"] != "late"]
    late = [n for n in names if n in specs and specs[n]["type"] == "late"]
    for n in late:  # members must exist
        trainable += [m for m in specs[n]["members"] if m not in trainable]
    todo = [(n, s) for n in trainable for s in seeds if args.force or not is_finished(runs_dir, n, s)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | to train: {sorted({n for n, _ in todo}) or 'nothing (all finished)'}")

    if todo:
        needed = sorted({k for n, _ in todo for k in model_inputs(specs[n])})
        t0 = time.time()
        store = load_store(features_dir, needed, local_dir=local)
        print(f"loaded {needed} in {time.time() - t0:.0f} s | "
              + ", ".join(f"{s} {len(store.index(s)):,}" for s in ("train", "dev", "test")))
        for n, s in todo:
            train_one(store, specs[n], n, s, tcfg, runs_dir, device)
        del store
    for n in late:
        for s in seeds:
            late_fusion(runs_dir, n, specs[n], s)
    if any(n in FLOORS for n in names):
        meta, _ = load_meta(features_dir)
        fit_floors(meta, runs_dir)


if __name__ == "__main__":
    main()
