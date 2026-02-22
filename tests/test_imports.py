from __future__ import annotations

import subprocess
import sys


def test_piper_imports_available() -> None:
    import importlib.util
    import pytest

    if importlib.util.find_spec("piper") is None or importlib.util.find_spec("piper.train.vits") is None:
        pytest.skip("piper not installed in this env")

    command = [
        sys.executable,
        "-c",
        "import piper; import piper.train; import piper.train.vits",
    ]
    subprocess.run(command, check=True)
