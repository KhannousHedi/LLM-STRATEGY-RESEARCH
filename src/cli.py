"""Pass CLI arguments through to the download and the backtest.

``download`` takes one or more symbols and ``--start`` and ``--end``.
``features`` takes one feature set name and prints each column it
returns, with its definition. ``backtest`` takes one ``--symbol``,
``--start``, ``--end``, ``--freq``, ``--strategy``, an optional JSON
``--parameters`` file, and repeated ``--param key=value``. ``--freq``
defaults to ``1D``. ``--strategy`` defaults to ``llm``. A ``--param``
value that parses as JSON is that value. Otherwise it stays a string.
It replaces that key in the dict.

There is no symbol-count check. The strategy checks its own parameters.
Coverage warnings stay in the loader. A finished year file is left as it
is.
"""

import argparse
import json
from pathlib import Path

from src.download import download_bars


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download")
    download.add_argument("symbols", nargs="+")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)

    features = commands.add_parser("features")
    features.add_argument("feature_set")

    run = commands.add_parser("backtest")
    run.add_argument("--symbol", required=True)
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--freq", default="1D")
    run.add_argument("--strategy", default="llm")
    run.add_argument("--parameters", metavar="PATH")
    run.add_argument("--param", action="append", default=[], metavar="key=value")
    return parser


def _parameters(parser, args):
    parameters = {}
    if args.parameters is not None:
        try:
            parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parser.error(f"--parameters: {error}")
        if not isinstance(parameters, dict):
            parser.error("--parameters must contain a JSON object")
    for item in args.param:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error("--param must be key=value")
        try:
            parameters[key] = json.loads(value)
        except json.JSONDecodeError:
            parameters[key] = value
    return parameters


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        download_bars(args.symbols, args.start, args.end)
        return
    if args.command == "features":
        from src.features import load_feature_set

        feature_set = load_feature_set(args.feature_set)
        if not feature_set.columns:
            print(f"{args.feature_set}: no columns, timestamps only")
        for name in feature_set.columns:
            print(f"{name}: {feature_set.descriptions[name]}")
        return
    parameters = _parameters(parser, args)
    from src.run import backtest

    print(backtest(args.symbol, args.start, args.end, args.freq, args.strategy, parameters))


if __name__ == "__main__":
    main()
