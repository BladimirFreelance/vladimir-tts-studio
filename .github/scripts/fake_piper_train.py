from __future__ import annotations

import pathlib
import sys


def main() -> int:
    args = sys.argv[1:]
    if "--trainer.default_root_dir" not in args:
        raise SystemExit("--trainer.default_root_dir is required")

    root = pathlib.Path(args[args.index("--trainer.default_root_dir") + 1])
    root.mkdir(parents=True, exist_ok=True)
    (root / "ci-smoke.ckpt").write_text("ci", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
