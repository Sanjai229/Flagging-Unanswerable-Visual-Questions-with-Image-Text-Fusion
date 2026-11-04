"""Fast end-to-end check on synthetic features (CPU, about a minute). Run from the repo root:
    python -m tests.test_smoke
It builds a tiny fake feature folder in the same format as notebooks/01_data_and_features.ipynb, trains every
model in configs/default.yaml for 2 epochs, and runs every evaluation function.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vizwiz import evaluate as ev
from vizwiz.train import load_config, main

D, D_CLIP, N_TOK, N_REG = 32, 16, 40, 49


def make_features(folder, rng):
    """Synthetic data whose label leaks weakly into both modalities, so models can learn something."""
    folder.mkdir(parents=True)
    splits = ["train"] * 300 + ["dev"] * 100 + ["test"] * 120
    n = len(splits)
    y = (rng.random(n) < 0.3).astype(int)
    direction = np.random.default_rng(1).choice([-1.0, 1.0], D)
    shift = (y[:, None] - 0.3) * 0.8 * direction
    votes = np.clip(y * 6 + rng.integers(0, 4, n), 0, 10)
    meta = pd.DataFrame({"pair_id": [f"VizWiz_{s}_{i:05d}.jpg" for i, s in enumerate(splits)],
                         "question": "what is this?", "official_split": [{"dev": "val"}.get(s, s) for s in splits],
                         "split": splits, "y": y, "unans_votes": votes, "vote_share": votes / 10,
                         "sharpness": rng.lognormal(5, 1, n) * (1 - 0.5 * y), "brightness": rng.normal(110, 20, n),
                         "megapixels": rng.uniform(0.5, 3, n), "file_kb": rng.uniform(100, 900, n),
                         "sharp_bin": rng.choice(["blurry", "medium", "sharp"], n),
                         "rare_question": rng.random(n) < 0.4, "answer_type": rng.choice(["other", "yes/no"], n)})
    meta.to_csv(folder / "meta.csv", index=False)
    mask = np.zeros((n, N_TOK), np.uint8)
    mask[:, :8] = 1
    arrays = {"img_cls": rng.normal(size=(n, D)) + shift, "img_tok": rng.normal(size=(n, N_REG, D)) + shift[:, None],
              "txt_mean": rng.normal(size=(n, D)) + shift, "txt_tok": rng.normal(size=(n, N_TOK, D)) + shift[:, None],
              "txt_mask": mask, "clip_img": rng.normal(size=(n, D_CLIP)) + shift[:, :D_CLIP],
              "clip_txt": rng.normal(size=(n, D_CLIP)) + shift[:, :D_CLIP]}
    manifest = {"n_rows": n, "meta": "meta.csv", "features": {}}
    for k, a in arrays.items():
        a = a.astype(np.uint8 if k == "txt_mask" else np.float16)
        np.save(folder / f"{k}_test.npy", a)
        manifest["features"][k] = {"file": f"{k}_test.npy", "shape": list(a.shape), "dtype": str(a.dtype)}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return meta


def run():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta = make_features(tmp / "features", rng)
        main(["--config", "configs/default.yaml", "--features-dir", str(tmp / "features"),
              "--runs-dir", str(tmp / "runs"), "--local-dir", "none", "--epochs", "2", "--seeds", "0", "1"])

        names = list(load_config("configs/default.yaml")["models"]) + ["majority", "photo_quality"]
        runs = ev.load_runs(tmp / "runs")
        assert set(runs) == set(names), sorted(set(names) ^ set(runs))
        for m, seeds in runs.items():
            for s, p in seeds.items():
                assert set(p.split) == {"dev", "test"} and len(p) == 220, (m, s)
                assert p.p_unanswerable.between(0, 1).all(), (m, s)
        assert "gate_q" in runs["gated"][0] and "p_no_image" in runs["cross_attention"][0]
        assert "p_no_question" not in runs["image_only"][0]

        thr = ev.thresholds(runs)
        scores = ev.scores_by_seed(runs, thr)
        mean, table = ev.mean_sd_table(scores, names)
        assert len(table) == len(names)
        dist = ev.bootstrap(runs, thr, ["image_only", "gated"], n_boot=50)
        assert ev.ci_table(dist).shape == (2, 3)
        ev.paired_gain(dist, "gated", "image_only")
        summaries, histories = ev.load_summaries(tmp / "runs"), ev.load_histories(tmp / "runs")
        assert not ev.overfitting_table(histories, summaries, names).empty
        assert not ev.missing_input_table(runs, thr, ["early", "gated", "cross_attention"]).empty
        assert not ev.subgroup_table(runs, thr, meta, "sharp_bin", ["image_only", "gated"]).empty
        cases, counts = ev.failure_cases(runs, thr, meta, "gated", k=3)
        assert len(cases) <= 6 and counts["threshold"] > 0
        assert not ev.error_profile(runs, thr, meta, "gated").empty
        assert "oracle_accuracy" in ev.complementarity(runs, thr)
        assert not ev.operating_points(runs, thr, "gated").empty
        assert not ev.gate_by_group(runs, thr, meta).empty
        assert not ev.cost_table(summaries, names).empty

        # Rerun: finished runs are skipped
        main(["--config", "configs/default.yaml", "--features-dir", str(tmp / "features"),
              "--runs-dir", str(tmp / "runs"), "--local-dir", "none", "--epochs", "2", "--seeds", "0", "--models",
              "early"])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    run()
