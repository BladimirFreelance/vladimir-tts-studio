#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(
            json.dumps(
                {
                    "path": None,
                    "epoch": None,
                    "global_step": None,
                    "error": "usage: python tools/ckpt_info.py <path_to_ckpt>",
                },
                ensure_ascii=False,
            )
        )
        return 1

    ckpt_path = Path(sys.argv[1]).expanduser()
    result = {
        "path": str(ckpt_path),
        "epoch": None,
        "global_step": None,
    }

    try:
        import torch

        data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(data, dict):
            epoch = data.get("epoch")
            global_step = data.get("global_step")
            if isinstance(epoch, (int, float)):
                result["epoch"] = int(epoch)
            if isinstance(global_step, (int, float)):
                result["global_step"] = int(global_step)
        else:
            result["error"] = f"unsupported checkpoint format: {type(data).__name__}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{exc.__class__.__name__}: {exc}"

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
