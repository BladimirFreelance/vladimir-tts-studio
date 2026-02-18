import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import main

if __name__ == "__main__":
    command = Path(__file__).stem.split("_", 1)[1]
    mapping = {
        "prepare_dataset": "prepare",
        "run_studio": "record",
        "train": "train",
        "export_onnx": "export",
        "test_voice": "test",
        "doctor": "doctor",
    }
    sys.argv.insert(1, mapping[command])
    main()
