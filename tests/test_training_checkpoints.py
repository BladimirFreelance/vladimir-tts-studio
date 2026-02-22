from __future__ import annotations

import os
from pathlib import Path

from training.checkpoints import (
    ensure_stable_ckpt_aliases,
    find_best_ckpt,
    find_latest_ckpt,
)


def test_find_latest_ckpt_prefers_newest_mtime(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    old = runs / "epoch=0.ckpt"
    new = runs / "epoch=1.ckpt"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")

    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))

    assert find_latest_ckpt(runs) == new


def test_find_best_ckpt_prefers_best_prefix(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    metric = runs / "epoch=3-val_loss=0.20.ckpt"
    explicit_best = runs / "best-epoch=2.ckpt"
    metric.write_text("metric", encoding="utf-8")
    explicit_best.write_text("best", encoding="utf-8")

    assert find_best_ckpt(runs) == explicit_best


def test_find_best_ckpt_uses_metric_when_no_best_alias(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    worse = runs / "epoch=5-val_loss=0.90.ckpt"
    better = runs / "epoch=4-val_loss=0.30.ckpt"
    worse.write_text("worse", encoding="utf-8")
    better.write_text("better", encoding="utf-8")

    assert find_best_ckpt(runs) == better


def test_ensure_stable_ckpt_aliases_copies_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    best = runs / "best-model.ckpt"
    last = runs / "epoch=9.ckpt"
    best.write_text("best-data", encoding="utf-8")
    last.write_text("last-data", encoding="utf-8")

    ensure_stable_ckpt_aliases(runs)

    assert (runs / "best.ckpt").read_text(encoding="utf-8") == "best-data"
    assert (runs / "last.ckpt").read_text(encoding="utf-8") == "last-data"
