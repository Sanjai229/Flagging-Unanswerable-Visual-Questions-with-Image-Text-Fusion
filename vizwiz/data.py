"""Load the cached features written by notebooks/01_data_and_features.ipynb.

features/manifest.json lists every feature array (file, shape, encoder) and features/meta.csv holds one row
per image-question pair, in the same row order as the arrays.

Splits follow the official VizWiz partition: official train -> train, official validation -> dev
(early stopping, thresholds, late-fusion weight), official test -> test (final scores only).
"""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SPLITS = ("train", "dev", "test")
QUESTION_FEATURES = {"txt_cls", "txt_mean", "txt_tok", "clip_txt"}
IMAGE_FEATURES = {"img_cls", "img_tok", "clip_img"}
MASK_FEATURES = {"txt_mask"}
REQUIRED_META = ("pair_id", "question", "split", "y", "vote_share")


def modality(name):
    """'question', 'image' or None (masks and flags) for a feature name."""
    if name in QUESTION_FEATURES:
        return "question"
    if name in IMAGE_FEATURES:
        return "image"
    return None


@dataclass
class FeatureStore:
    """meta (one row per pair) plus the feature arrays as CPU tensors, row-aligned with meta."""
    meta: pd.DataFrame
    arrays: dict

    def index(self, split):
        """Row positions of one split: 'train', 'dev' or 'test'."""
        return np.where(self.meta["split"].values == split)[0]

    def dims(self):
        return {k: int(v.shape[-1]) for k, v in self.arrays.items()}

    def batch(self, idx, needs, device, blank=None):
        """Inputs a model asks for, on `device`. blank='question' or 'image' replaces that modality with zeros
        (masks are kept), to test how a fusion model copes when one input is missing."""
        idx = torch.as_tensor(np.asarray(idx))
        out = {}
        for k in needs:
            x = self.arrays[k][idx].to(device, non_blocking=True)
            if k in MASK_FEATURES:
                x = x.bool()
            else:
                x = x.float()
                if blank is not None and modality(k) == blank:
                    x = torch.zeros_like(x)
            out[k] = x
        return out

    def target(self, label):
        """Training target: 'hard' = official label y; 'soft' = share of crowd answers saying unanswerable."""
        if label == "hard":
            t = self.meta["y"]
        elif label == "soft":
            t = self.meta["vote_share"].fillna(self.meta["y"])
        else:
            raise ValueError("label must be 'hard' or 'soft'")
        return torch.from_numpy(t.values.astype(np.float32))


def read_manifest(features_dir):
    return json.loads((Path(features_dir) / "manifest.json").read_text())


def load_meta(features_dir):
    features_dir = Path(features_dir)
    manifest = read_manifest(features_dir)
    meta = pd.read_csv(features_dir / manifest.get("meta", "meta.csv"))
    missing = [c for c in REQUIRED_META if c not in meta.columns]
    if missing:
        raise ValueError(f"meta.csv is missing columns {missing}; rerun notebooks/01_data_and_features.ipynb")
    if len(meta) != manifest["n_rows"]:
        raise ValueError("meta.csv and manifest.json disagree on the number of rows")
    if not meta["pair_id"].is_unique:
        raise ValueError("pair_id must be unique")
    bad = set(meta["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split values {bad}")
    if meta.loc[meta.split.isin(["train", "dev"]), "y"].isna().any():
        raise ValueError("train and dev pairs must all have a label")
    meta["y"] = meta["y"].astype(np.float32)
    return meta, manifest


def load_store(features_dir, names, local_dir=None):
    """Read meta.csv and the arrays in `names` into memory. With local_dir, each array is first copied there
    (much faster than reading from Google Drive)."""
    meta, manifest = load_meta(features_dir)
    arrays = {}
    for name in names:
        if name not in manifest["features"]:
            raise KeyError(f"feature '{name}' is not in manifest.json; available: {sorted(manifest['features'])}")
        src = Path(features_dir) / manifest["features"][name]["file"]
        if local_dir is not None:
            dst = Path(local_dir) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, dst)
            src = dst
        a = np.load(src, mmap_mode="r")
        if len(a) != len(meta):
            raise ValueError(f"{src.name} has {len(a)} rows but meta.csv has {len(meta)}")
        arrays[name] = torch.from_numpy(np.array(a))  # copy into memory (writable, contiguous)
    return FeatureStore(meta=meta, arrays=arrays)
