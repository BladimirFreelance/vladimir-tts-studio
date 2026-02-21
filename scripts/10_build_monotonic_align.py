from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MONO_DIR = (
    REPO_ROOT
    / "third_party"
    / "piper1-gpl"
    / "src"
    / "piper"
    / "train"
    / "vits"
    / "monotonic_align"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main() -> int:
    if os.name != "nt":
        print("[i] Non-Windows: monotonic_align build is usually handled by your toolchain.")
        return 0

    setup_py = MONO_DIR / "setup.py"
    if not setup_py.exists():
        print(f"[FAIL] setup.py not found: {setup_py}")
        print("[HINT] Did you clone third_party/piper1-gpl ?")
        return 2

    # Build deps (inside current venv)
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "cython", "ninja"])

    # Build in-place (may try to output into local piper/... path)
    try:
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=MONO_DIR)
    except subprocess.CalledProcessError:
        # Workaround: create local output folder expected by some setups
        (MONO_DIR / "piper" / "train" / "vits" / "monotonic_align").mkdir(parents=True, exist_ok=True)
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=MONO_DIR)

    produced = list((MONO_DIR / "piper" / "train" / "vits" / "monotonic_align").glob("core*.pyd"))
    if not produced:
        produced = list(MONO_DIR.glob("core*.pyd"))

    if not produced:
        print("[FAIL] No core*.pyd produced.")
        print("[HINT] Install Visual Studio 2022 Build Tools (Desktop development with C++ + Windows SDK).")
        return 3

    pyd = produced[0]
    print(f"[OK] Produced: {pyd}")

    # Training expects: .../monotonic_align/monotonic_align/core*.pyd
    target_pkg = MONO_DIR / "monotonic_align"
    target_pkg.mkdir(parents=True, exist_ok=True)
    target = target_pkg / pyd.name
    shutil.copy2(pyd, target)
    print(f"[OK] Copied to: {target}")

    # Minimal check: runtime phonemizer exists and .pyd is in the expected location.
    test_code = (
        "import importlib;"
        "importlib.import_module('piper.espeakbridge');"
        "print('OK espeakbridge');"
    )
    run([sys.executable, "-c", test_code], cwd=REPO_ROOT)

    # Verify pyd in target location
    if not list((target_pkg).glob('core*.pyd')):
        raise SystemExit('[FAIL] core*.pyd not found in monotonic_align/monotonic_align after copy')
    print('[OK] core*.pyd present in monotonic_align/monotonic_align')

    print("\nDONE: monotonic_align is built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
