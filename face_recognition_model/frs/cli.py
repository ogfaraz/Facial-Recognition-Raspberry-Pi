from __future__ import annotations

import argparse

from frs.app import run_application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        "--register",
        metavar="NAME",
        help='Register a new face. E.g.: --register "Faraz"',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_application(args.register)


if __name__ == "__main__":
    main()
