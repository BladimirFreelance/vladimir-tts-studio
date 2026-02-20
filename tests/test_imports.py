from __future__ import annotations

import subprocess
import sys


def test_piper_imports_available() -> None:
    command = [
        sys.executable,
        "-c",
        "import piper; import piper.train; import piper.train.vits",
    ]
    subprocess.run(command, check=True)
