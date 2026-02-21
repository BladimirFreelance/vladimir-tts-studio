from __future__ import annotations

import re
import shutil
from pathlib import Path

_METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


def _iter_ckpts(runs_dir: Path) -> list[Path]:
    return [
        path
        for path in runs_dir.glob("**/*.ckpt")
        if path.is_file() and path.name not in {"best.ckpt", "last.ckpt"}
    ]


def find_latest_ckpt(runs_dir: Path) -> Path:
    ckpts = sorted(_iter_ckpts(runs_dir), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError("No ckpt found in project runs folder")
    return ckpts[0]


def _metric_value(ckpt: Path) -> tuple[int, float] | None:
    metrics = {match.group(1).lower(): float(match.group(2)) for match in _METRIC_RE.finditer(ckpt.stem)}
    if not metrics:
        return None

    if "val_loss" in metrics:
        return (0, metrics["val_loss"])
    if "loss" in metrics:
        return (1, metrics["loss"])
    if "cer" in metrics:
        return (2, metrics["cer"])
    if "wer" in metrics:
        return (3, metrics["wer"])
    if "acc" in metrics:
        return (4, -metrics["acc"])
    if "accuracy" in metrics:
        return (4, -metrics["accuracy"])
    return None


def find_best_ckpt(runs_dir: Path) -> Path:
    explicit_best = sorted(
        [path for path in _iter_ckpts(runs_dir) if path.name.lower().startswith("best")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if explicit_best:
        return explicit_best[0]

    ranked = []
    for ckpt in _iter_ckpts(runs_dir):
        metric = _metric_value(ckpt)
        if metric is None:
            continue
        ranked.append((metric[0], metric[1], -ckpt.stat().st_mtime, ckpt))

    if ranked:
        ranked.sort()
        return ranked[0][3]

    return find_latest_ckpt(runs_dir)


def ensure_stable_ckpt_aliases(runs_dir: Path) -> None:
    if not runs_dir.exists():
        return

    best = None
    last = None
    try:
        best = find_best_ckpt(runs_dir)
    except FileNotFoundError:
        pass

    try:
        last = find_latest_ckpt(runs_dir)
    except FileNotFoundError:
        pass

    for source, alias_name in ((best, "best.ckpt"), (last, "last.ckpt")):
        if source is None:
            continue
        alias_path = runs_dir / alias_name
        if alias_path.resolve() == source.resolve():
            continue
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, alias_path)
