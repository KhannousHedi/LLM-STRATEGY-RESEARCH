import json

from src.prompt import PROMPTS_DIR, load_template, render

SYMBOL = "SPY"
CALENDAR_DATE = "2024-03-04"


def _file_fields(template):
    return {
        "feature_set": template.feature_set,
        "parameters": template.parameters,
        "wording": template.wording,
    }


def test_template_load_and_render(tmp_path):
    loaded = load_template("default")
    default_file = json.loads((PROMPTS_DIR / "default.json").read_text(encoding="utf-8"))
    assert _file_fields(loaded) == default_file

    cautious_path = tmp_path / "cautious.json"
    cautious_path.write_text(
        json.dumps(
            {
                "feature_set": "default",
                "parameters": {"ma_window": 10, "vol_window": 10, "volume_window": 10},
                "wording": loaded.wording,
            }
        ),
        encoding="utf-8",
    )
    other = load_template("cautious", root=tmp_path)
    assert _file_fields(other) == json.loads(cautious_path.read_text(encoding="utf-8"))

    row = {
        "r1": 0.0125,
        "r5": -0.034,
        "r20": 0.058,
        "dist_ma": -0.0075,
        "vol": 0.021,
        "vol_ratio": 1.35,
    }
    text = render(loaded, row, "out")
    for value in row.values():
        assert format(value, ".6f") in text
    assert "Position: out" in text
    assert SYMBOL not in text
    assert CALENDAR_DATE not in text
