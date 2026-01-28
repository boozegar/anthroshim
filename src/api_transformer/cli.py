from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, Iterator

from .openai_to_anthropic import convert_openai_to_anthropic
from .openai_stream_to_anthropic_stream import iter_anthropic_events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="api-transformer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("openai-to-anthropic", help="Convert OpenAI response/input to Anthropic Messages")
    p1.add_argument("--in", dest="in_path", required=True)
    p1.add_argument("--out", dest="out_path", required=True)
    p1.add_argument("--mode", default="auto", choices=["auto", "input", "response", "output"])

    p2 = sub.add_parser(
        "openai-stream-to-anthropic-stream",
        help="Convert OpenAI streaming events (NDJSON) to Anthropic streaming events (NDJSON)",
    )
    p2.add_argument("--in", dest="in_path", required=True)
    p2.add_argument("--out", dest="out_path", required=True)
    p2.add_argument("--model", default="unknown")
    p2.add_argument("--message-id", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "openai-to-anthropic":
        with open(args.in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = convert_openai_to_anthropic(data, mode=args.mode)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=True, indent=2)
        return 0

    if args.cmd == "openai-stream-to-anthropic-stream":
        with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
            events = _iter_ndjson(fin)
            for ev in iter_anthropic_events(events, model=args.model, message_id=args.message_id):
                fout.write(json.dumps(ev, ensure_ascii=True) + "\n")
        return 0

    return 2


def _iter_ndjson(f) -> Iterator[Dict[str, Any]]:
    for line in f:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
