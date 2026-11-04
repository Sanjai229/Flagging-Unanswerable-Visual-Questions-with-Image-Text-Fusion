"""Evaluation: metrics, seed summaries, bootstrap confidence intervals, overfitting, subgroups, missing inputs,
calibration, failure cases and cost. Every function reads the run folders written by vizwiz.train.

Rules: each run's decision threshold is tuned on dev (macro-F1); all reported numbers are on test.
Positive class = unanswerable (y = 1).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

METRICS = ["macro_f1", "f1_unanswerable", "recall_unanswerable", "balanced_acc", "accuracy", "auroc", "auprc", "brier",
           "ece"]
THRESHOLDS = np.linspace(0.05, 0.95, 91)


# ---------------------------------------------------------------- loading
def load_runs(runs_dir, names=None):
    """{model: {seed: predictions DataFrame}} with rows in the same pair order for every model and seed."""
    runs = {}
    for f in sorted(Path(runs_dir).glob("*/seed*/predictions.csv")):
        name, seed = f.parent.parent.name, int(f.parent.name[4:])
        if names is None or name in names:
            p = pd.read_csv(f).sort_values(["split", "pair_id"], kind="stable").reset_index(drop=True)
            runs.setdefault(name, {})[seed] = p
    return runs


def load_summaries(runs_dir):
    rows = [json.loads(f.read_text()) for f in sorted(Path(runs_dir).glob("*/seed*/summary.json"))]
    return pd.DataFrame(rows)


def load_histories(runs_dir):
    frames = []
    for f in sorted(Path(runs_dir).glob("*/seed*/history.csv")):
        frames.append(pd.read_csv(f).assign(model=f.parent.parent.name, seed=int(f.parent.name[4:])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- metrics
def macro_f1_counts(tp, fp, fn, tn):
    """Macro-F1 from confusion counts (works on arrays, for fast bootstrapping)."""
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fn + fp) > 0)
    return (f1_pos + f1_neg) / 2


def macro_f1(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    return float(macro_f1_counts(np.sum(y & yhat), np.sum(~y & yhat), np.sum(y & ~yhat), np.sum(~y & ~yhat)))


def tune_threshold(p, y):
    """Threshold that maximises macro-F1. Only ever called on dev."""
    return float(THRESHOLDS[int(np.argmax([macro_f1(y, p >= t) for t in THRESHOLDS]))])


def expected_calibration_error(p, y, n_bins=15):
    """Sum over equal-width bins of (share of pairs) x |observed unanswerable rate - mean predicted probability|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def metrics(p, y, t):
    with warnings.catch_warnings():  # subgroups can hold a single class; those metrics are then NaN or trivial
        warnings.simplefilter("ignore")
        return _metrics(p, y, t)


def _metrics(p, y, t):
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    yhat = (p >= t).astype(int)
    tp, fp, fn = np.sum((y == 1) & (yhat == 1)), np.sum((y == 0) & (yhat == 1)), np.sum((y == 1) & (yhat == 0))
    return {"macro_f1": macro_f1(y, yhat),
            "f1_unanswerable": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision_unanswerable": tp / max(tp + fp, 1),
            "recall_unanswerable": tp / max(tp + fn, 1),
            "balanced_acc": balanced_accuracy_score(y, yhat),
            "accuracy": accuracy_score(y, yhat),
            "auroc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            "auprc": average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
            "brier": brier_score_loss(y, p),
            "ece": expected_calibration_error(p, y)}


def split_of(pred, split, col="p_unanswerable"):
    d = pred[pred.split == split]
    return d[col].values, d.y.values.astype(int)


def thresholds(runs):
    """{(model, seed): dev-tuned threshold}"""
    return {(m, s): tune_threshold(*split_of(p, "dev")) for m, seeds in runs.items() for s, p in seeds.items()}


def scores_by_seed(runs, thr, split="test", col="p_unanswerable"):
    rows = []
    for m, seeds in runs.items():
        for s, p in seeds.items():
            if col in p:
                rows.append({"model": m, "seed": s, "threshold": thr[(m, s)], **metrics(*split_of(p, split, col),
                                                                                      thr[(m, s)])})
    return pd.DataFrame(rows)


def mean_sd_table(scores, order, cols=METRICS, digits=3):
    """'mean +/- sd' over seeds for each model, in `order`."""
    order = [m for m in order if m in set(scores.model)]
    g = scores.groupby("model")[cols]
    mean, sd, n = g.mean().reindex(order), g.std().reindex(order), scores.groupby("model").size().reindex(order)
    out = pd.DataFrame(index=order)
    for c in cols:
        out[c] = [f"{a:.{digits}f}" + (f" +/- {b:.{digits}f}" if pd.notna(b) else "") for a, b in zip(mean[c], sd[c])]
    out["seeds"] = n.values
    return mean, out


# ---------------------------------------------------------------- bootstrap
def _resample_counts(yhat_list, y, boot):
    """Macro-F1 averaged over seeds, for every bootstrap resample (rows of `boot`)."""
    yb = y[boot]
    f1s = []
    for yhat in yhat_list:
        hb = yhat[boot]
        tp = np.sum(yb & hb, 1)
        fp = np.sum(~yb & hb, 1)
        fn = np.sum(yb & ~hb, 1)
        tn = np.sum(~yb & ~hb, 1)
        f1s.append(macro_f1_counts(tp, fp, fn, tn))
    return np.mean(f1s, 0)


def bootstrap(runs, thr, names, split="test", n_boot=1000, seed=42):
    """Percentile 95% CI (Efron & Tibshirani 1993) of each model's test macro-F1, averaged over its seeds.
    Every model is scored on the same resamples, so differences between models can be compared directly."""
    any_run = next(iter(runs[names[0]].values()))
    y = (any_run[any_run.split == split].y.values == 1)
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(y), (n_boot, len(y)))
    dist = {}
    for m in names:
        yhats = [(split_of(p, split)[0] >= thr[(m, s)]) for s, p in runs[m].items()]
        dist[m] = _resample_counts(yhats, y, boot)
    return dist


def ci_table(dist):
    return pd.DataFrame({m: {"mean": d.mean(), "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)}
                         for m, d in dist.items()}).T


def paired_gain(dist, a, b):
    """macro-F1(a) - macro-F1(b) on the same resamples: mean, 95% CI, and share of resamples where a <= b."""
    g = dist[a] - dist[b]
    return {"model": a, "vs": b, "gain": float(g.mean()), "ci_low": float(np.percentile(g, 2.5)),
            "ci_high": float(np.percentile(g, 97.5)), "share_not_better": float(np.mean(g <= 0))}


# ---------------------------------------------------------------- analyses
def overfitting_table(histories, summaries, order):
    """Train and dev log loss at the selected epoch, and how far dev loss rose after it."""
    rows = []
    for m in order:
        h = histories[histories.model == m]
        if h.empty:
            continue
        for s, hs in h.groupby("seed"):
            best = int(hs.dev_auprc.values.argmax())
            rows.append({"model": m, "seed": s, "best_epoch": best + 1, "epochs_run": len(hs),
                         "train_loss": hs.train_loss.values[best], "dev_loss": hs.dev_loss.values[best],
                         "gap": hs.dev_loss.values[best] - hs.train_loss.values[best],
                         "final_gap": hs.dev_loss.values[-1] - hs.train_loss.values[-1]})
    t = pd.DataFrame(rows)
    return t.groupby("model")[["best_epoch", "epochs_run", "train_loss", "dev_loss", "gap", "final_gap"]] \
        .mean().reindex([m for m in order if m in set(t.model)]).round(3)


def missing_input_table(runs, thr, names):
    """Test scores of fusion models with both inputs, without the question, and without the photo
    (same model, same dev-tuned threshold). Shows how much each model relies on each input."""
    rows = []
    for m in names:
        for s, p in runs.get(m, {}).items():
            if "p_no_question" not in p:
                continue
            for col, label in (("p_unanswerable", "both inputs"), ("p_no_question", "no question"),
                               ("p_no_image", "no photo")):
                r = metrics(*split_of(p, "test", col), thr[(m, s)])
                rows.append({"model": m, "seed": s, "input": label, "macro_f1": r["macro_f1"], "auprc": r["auprc"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.groupby(["model", "input"])[["macro_f1", "auprc"]].mean().unstack("input").round(3)


def subgroup_table(runs, thr, meta, column, names, split="test"):
    """Macro-F1 and AUPRC (mean over seeds) for each level of a meta column, e.g. sharp_bin or rare_question."""
    rows = []
    for m in names:
        for s, p in runs[m].items():
            d = p[p.split == split].merge(meta[["pair_id", column]], on="pair_id", how="left")
            for level, g in d.groupby(column):
                r = metrics(g.p_unanswerable.values, g.y.values.astype(int), thr[(m, s)])
                rows.append({"group": column, "level": level, "n": len(g), "unanswerable_rate": g.y.mean(),
                             "model": m, "seed": s, "macro_f1": r["macro_f1"], "auprc": r["auprc"],
                             "recall_unanswerable": r["recall_unanswerable"]})
    t = pd.DataFrame(rows)
    return t.groupby(["group", "level", "model"]).agg(
        n=("n", "first"), unanswerable_rate=("unanswerable_rate", "first"), macro_f1=("macro_f1", "mean"),
        auprc=("auprc", "mean"), recall_unanswerable=("recall_unanswerable", "mean")).reset_index()


def ensemble(runs, thr, name, split="test"):
    """Seed-averaged probability and threshold for one model (used for failure cases and curves)."""
    ps = [p[p.split == split] for p in runs[name].values()]
    out = ps[0][["pair_id", "y"]].copy()
    out["p"] = np.mean([p.p_unanswerable.values for p in ps], 0)
    if "gate_q" in ps[0]:
        out["gate_q"] = np.mean([p.gate_q.values for p in ps], 0)
    return out, float(np.mean([thr[(name, s)] for s in runs[name]]))


def failure_cases(runs, thr, meta, name, k=8):
    """The most confident mistakes of a model on test: missed unanswerable pairs (false negatives) and
    answerable pairs flagged as unanswerable (false positives)."""
    e, t = ensemble(runs, thr, name)
    e = e.merge(meta, on="pair_id", how="left", suffixes=("", "_meta"))
    fn = e[(e.y == 1) & (e.p < t)].nsmallest(k, "p").assign(error="missed unanswerable")
    fp = e[(e.y == 0) & (e.p >= t)].nlargest(k, "p").assign(error="false alarm")
    counts = {"false_negatives": int(((e.y == 1) & (e.p < t)).sum()),
              "false_positives": int(((e.y == 0) & (e.p >= t)).sum()), "threshold": t}
    return pd.concat([fn, fp]), counts


def error_profile(runs, thr, meta, name, columns=("sharp_bin", "rare_question", "answer_type")):
    """Share of each subgroup among all errors vs among all test pairs (where do mistakes concentrate?)."""
    e, t = ensemble(runs, thr, name)
    e = e.merge(meta, on="pair_id", how="left", suffixes=("", "_meta"))
    e["wrong"] = (e.p >= t).astype(int) != e.y.astype(int)
    rows = []
    for c in columns:
        for level, g in e.groupby(c):
            rows.append({"group": c, "level": level, "share_of_test": len(g) / len(e),
                         "share_of_errors": g.wrong.sum() / max(e.wrong.sum(), 1), "error_rate": g.wrong.mean()})
    return pd.DataFrame(rows).round(3)


def complementarity(runs, thr, a="question_only", b="image_only"):
    """How often only one of two single-modality models is right (mean over seeds)."""
    rows = []
    for s in sorted(set(runs[a]) & set(runs[b])):
        ok = {}
        for m in (a, b):
            p, y = split_of(runs[m][s], "test")
            ok[m] = (p >= thr[(m, s)]).astype(int) == y
        rows.append({"both_correct": np.mean(ok[a] & ok[b]), f"only_{a}": np.mean(ok[a] & ~ok[b]),
                     f"only_{b}": np.mean(~ok[a] & ok[b]), "both_wrong": np.mean(~ok[a] & ~ok[b]),
                     "exactly_one_correct": np.mean(ok[a] ^ ok[b]), "oracle_accuracy": np.mean(ok[a] | ok[b])})
    return pd.DataFrame(rows).mean().round(3).to_dict()


def operating_points(runs, thr, name, precisions=(0.6, 0.7, 0.8)):
    """Threshold trade-off for deployment: recall of unanswerable pairs at the dev-tuned threshold and at fixed
    precision levels (thresholds chosen on dev, applied to test; seed-averaged probabilities)."""
    dev, _ = ensemble(runs, thr, name, "dev")
    test, t = ensemble(runs, thr, name, "test")
    rows = [{"rule": "dev-tuned macro-F1 threshold", "threshold": t, **_pr_at(test, t)}]
    pr, rc, th = precision_recall_curve(dev.y, dev.p)
    for target in precisions:
        ok = np.where(pr[:-1] >= target)[0]
        if len(ok):
            tt = float(th[ok[0]])
            rows.append({"rule": f"precision >= {target:.0%} on dev", "threshold": tt, **_pr_at(test, tt)})
    return pd.DataFrame(rows).round(3)


def _pr_at(e, t):
    yhat = e.p.values >= t
    y = e.y.values == 1
    tp = np.sum(y & yhat)
    return {"precision_unanswerable": tp / max(yhat.sum(), 1), "recall_unanswerable": tp / max(y.sum(), 1),
            "share_flagged": yhat.mean()}


def gate_by_group(runs, thr, meta, name="gated", columns=("sharp_bin", "rare_question")):
    """Mean gate value (weight on the question) by subgroup and true label."""
    if name not in runs or "gate_q" not in next(iter(runs[name].values())):
        return pd.DataFrame()
    e, _ = ensemble(runs, thr, name)
    e = e.merge(meta, on="pair_id", how="left", suffixes=("", "_meta"))
    rows = []
    for c in columns:
        for (level, y), g in e.groupby([c, "y"]):
            rows.append({"group": c, "level": level, "label": "unanswerable" if y == 1 else "answerable",
                         "mean_gate_q": g.gate_q.mean(), "n": len(g)})
    return pd.DataFrame(rows).round(3)


def cost_table(summaries, order):
    cols = ["params", "epochs_run", "sec_per_epoch", "peak_vram_gb", "head_latency_ms"]
    s = summaries[summaries.model.isin(order)]
    t = s.groupby("model")[[c for c in cols if c in s]].mean().reindex([m for m in order if m in set(s.model)])
    return t.round(3)
