import json
import re
from pathlib import Path

from src.run import record

RUNS = Path(__file__).resolve().parent.parent / "runs"
NAME = "sample"
TEMPLATE = "One named template.\n"


def test_run_records_template(tmp_path):
    before = sorted(path.name for path in RUNS.iterdir())
    folder = record(
        symbol="SPY",
        window={"start": "2024-03-04", "end": "2024-03-05"},
        freq="1D",
        model="qwen3:4b-instruct",
        prompt=NAME,
        template=TEMPLATE,
        slippage=0.0005,
        cash=100_000,
        root=tmp_path,
    )
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["prompt"] == NAME
    assert (folder / "prompt.txt").read_text(encoding="utf-8") == TEMPLATE
    assert folder.parent == tmp_path
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", folder.name)
    assert sorted(path.name for path in RUNS.iterdir()) == before
