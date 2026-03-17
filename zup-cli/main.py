#!/usr/bin/env python3
"""
Zup CLI — AI coding assistant powered by StackSpot AI.

Usage:
  python main.py                    # interactive REPL
  python main.py "your prompt"      # one-shot (non-interactive)
  python main.py --config           # configure credentials
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="zup",
        description="AI coding assistant powered by StackSpot AI",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Run a single prompt and exit (non-interactive)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Configure credentials interactively",
    )

    args = parser.parse_args()

    if args.config:
        from config import configure
        configure()
        return

    from repl import start_repl
    start_repl(initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
