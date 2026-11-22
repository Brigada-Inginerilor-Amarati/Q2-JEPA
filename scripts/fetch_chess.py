"""Stream the start of the Lichess evaluations database (CC0) and keep N (data/chess.py) positions with 3 equally good moves.

pip install zstandard && python scripts/fetch_chess.py [--root data]

Reads https://database.lichess.org/lichess_db_eval.jsonl.zst as a stream and stops once it has enough positions
(a few MB, not the whole file). A position is kept if white is to move and its deepest multi-PV evaluation has at
least 3 lines without mate scores, the 3 best first moves are distinct and all within CP_SPREAD centipawns of the
best (moves must differ in their squares, not only the promotion piece). Writes <root>/chess/positions.npz: fen (n,), moves (n, 3) UCI strings, in file order (train/dev/test are
consecutive slices, data/chess.py).
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.data.chess import CP_SPREAD, MOVES, PATH, N, sha256

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def pick(entry: dict) -> list[str] | None:
    """The 3 equally good first moves of one entry, or None."""
    if entry["fen"].split()[1] != "w" or not entry.get("evals"):
        return None
    pvs = max(entry["evals"], key=lambda e: (len(e["pvs"]), e.get("depth", 0)))["pvs"]
    if len(pvs) < MOVES or any("cp" not in p for p in pvs[:MOVES]):
        return None
    moves = [p["line"].split()[0] for p in pvs[:MOVES]]
    best = pvs[0]["cp"]
    if len({m[:4] for m in moves}) < MOVES or any(abs(p["cp"] - best) > CP_SPREAD for p in pvs[:MOVES]):
        return None
    return moves


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data")
    args = ap.parse_args(argv)
    import zstandard

    fens, moves, seen = [], [], 0
    with urllib.request.urlopen(URL) as resp:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(resp), encoding="utf-8")
        for line in text:
            seen += 1
            entry = json.loads(line)
            m = pick(entry)
            if m:
                fens.append(entry["fen"])
                moves.append(m)
                if len(fens) == N:
                    break
    out = Path(args.root) / PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, fen=np.array(fens), moves=np.array(moves))
    print(f"kept {len(fens)} of {seen} positions -> {out}, sha256 {sha256(out)} (data/chess.py SHA256)")
    return 0 if len(fens) == N else 1


if __name__ == "__main__":
    raise SystemExit(main())
