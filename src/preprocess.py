"""Dataset loading & preprocessing utilities (GSM8K).

All datasets are cached in `.cache/`.
"""
from __future__ import annotations

from typing import Dict

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

CACHE_DIR = ".cache/"


def _lowercase_question(example: Dict):
    example["question"] = example["question"].lower()
    return example


def _load_gsm8k(cfg: DictConfig):
    ds = load_dataset("gsm8k", "main", cache_dir=CACHE_DIR)
    val_ratio = float(cfg.dataset.split_ratios.validation)
    split = ds["train"].train_test_split(test_size=val_ratio, seed=int(cfg.training.seed))
    train_ds, val_ds = split["train"], split["test"]
    test_ds = ds["test"]
    if bool(cfg.dataset.preprocessing.get("lower_case", False)):
        train_ds = train_ds.map(_lowercase_question)
        val_ds = val_ds.map(_lowercase_question)
        test_ds = test_ds.map(_lowercase_question)
    return {"train": train_ds, "validation": val_ds, "test": test_ds}


def load_and_preprocess(cfg: DictConfig):
    name = str(cfg.dataset.name).lower()
    if name == "gsm8k":
        return _load_gsm8k(cfg)
    raise NotImplementedError(name)
