"""Executable exact finite coefficient test; supply negative values with --a=-1/2."""
import argparse
import json
from pathlib import Path

from _common import ROOT
from experiments.src.affine_admissibility import admissibility_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--c', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = admissibility_report(args.a, args.b, args.c)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
