"""Pass CLI arguments through to the download and the backtest.

``download`` takes one or more symbols and ``--start`` and ``--end``.
``backtest`` takes one ``--symbol``, ``--start``, ``--end``, ``--model``,
``--freq``, and ``--prompt``. ``--freq`` defaults to ``1D``. ``--prompt``
defaults to ``default``. ``--model`` defaults to ``qwen3:4b-instruct``.

There is no symbol-count check. A missing prompt file fails when
``load_template`` opens it. Coverage warnings and the download skip stay
in the loader and the download.
"""

import argparse

from src.download import download_bars
from src.llm import MODEL


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download")
    download.add_argument("symbols", nargs="+")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)

    run = commands.add_parser("backtest")
    run.add_argument("--symbol", required=True)
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--model", default=MODEL)
    run.add_argument("--freq", default="1D")
    run.add_argument("--prompt", default="default")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "download":
        download_bars(args.symbols, args.start, args.end)
        return
    from src.run import backtest

    print(backtest(args.symbol, args.start, args.end, args.model, args.freq, args.prompt))


if __name__ == "__main__":
    main()
