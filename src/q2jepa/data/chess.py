"""Chess positions with several equally good moves (Lichess evaluations, CC0), rendered as 64x64 clips.

Each position (white to move) comes with its top 3 engine moves, all within CP_SPREAD centipawns of the best
(scripts/fetch_chess.py selects them). The 3 moves are the 3 equally likely futures. A board is rendered as a
3-frame clip so it fits the video encoders: frame 0 = white pieces, frame 1 = black pieces, frame 2 = all pieces.
A square is 8x8 px with rank 8 at the top; a piece is a centred 6x6 patch whose intensity encodes its type.
The observation is the position; each future is the position after one of the 3 moves. Future state: the
move's (from file, from rank, to file, to rank), 0..7.
"""

import hashlib
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch

MOVES = 3
CP_SPREAD = 30
INTENSITY = {"p": 0.3, "n": 0.45, "b": 0.55, "r": 0.7, "q": 0.85, "k": 1.0}
PATH = Path("chess/positions.npz")  # under dataset.root
N = 10208  # positions: train 10000, dev 80, test 128
# sha256 of the 10208 fetched positions (2026-11-22; the first 1008 equal the first fetch). The Lichess stream is
# not versioned, so a re-fetch must reproduce it.
SHA256 = "6876a1f330b2e331b2d7ff53e8f95332ff253c1de3b3fe32a48791559c722ce8"


def sha256(path: Path) -> str:
    """Hash of the positions and moves (not the file bytes: npz files carry timestamps)."""
    data = np.load(path)
    text = "\n".join(f"{f} {' '.join(m)}" for f, m in zip(data["fen"].tolist(), data["moves"].tolist()))
    return hashlib.sha256(text.encode()).hexdigest()


def parse_fen(fen: str) -> list[list[str]]:
    """Board as 8 rows (rank 8 first) of 8 cells: piece letter (upper = white) or ''."""
    rows = []
    for rank in fen.split()[0].split("/"):
        row = []
        for ch in rank:
            row.extend([""] * int(ch) if ch.isdigit() else [ch])
        rows.append(row)
    return rows


def square(name: str) -> tuple[int, int]:
    """'e4' -> (row, col) on the parsed board: row 0 = rank 8."""
    return 8 - int(name[1]), ord(name[0]) - ord("a")


def apply_move(board: list[list[str]], uci: str) -> list[list[str]]:
    """Board after a UCI move (e2e4, e7e8q). Handles castling, en passant and promotion; no legality check."""
    b = [row[:] for row in board]
    (r0, c0), (r1, c1) = square(uci[:2]), square(uci[2:4])
    piece = b[r0][c0]
    if piece.lower() == "p" and c0 != c1 and not b[r1][c1]:
        b[r0][c1] = ""  # en passant: the captured pawn is beside the start square
    if piece.lower() == "k" and r0 == r1 and b[r1][c1].lower() == "r" and b[r1][c1].isupper() == piece.isupper():
        c1 = 6 if c1 > c0 else 2  # castling written king-onto-rook (e1h1): same as e1g1
    if piece.lower() == "k" and abs(c1 - c0) == 2:  # castling: move the rook too
        rc0, rc1 = (7, 5) if c1 > c0 else (0, 3)
        b[r0][rc1], b[r0][rc0] = b[r0][rc0], ""
    b[r1][c1], b[r0][c0] = piece, ""
    if len(uci) == 5:
        b[r1][c1] = uci[4].upper() if piece.isupper() else uci[4].lower()
    return b


def move_state(uci: str) -> list[float]:
    """(from file, from rank, to file, to rank), each 0..7; king-onto-rook castling maps to the king's square."""
    to = {"e1h1": "g1", "e1a1": "c1", "e8h8": "g8", "e8a8": "c8"}.get(uci[:4], uci[2:4])
    return [ord(uci[0]) - 97, int(uci[1]) - 1, ord(to[0]) - 97, int(to[1]) - 1]


def render(board: list[list[str]]) -> np.ndarray:
    """Clip (3, 64, 64): white pieces, black pieces, all pieces."""
    clip = np.zeros((3, 64, 64), np.float32)
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p:
                v = INTENSITY[p.lower()]
                ys, xs = slice(8 * r + 1, 8 * r + 7), slice(8 * c + 1, 8 * c + 7)
                clip[0 if p.isupper() else 1, ys, xs] = v
                clip[2, ys, xs] = v
    return clip


PIECES = "PNBRQKpnbrqk"  # board code = 1 + index; 0 = empty


def codes(board: list[list[str]]) -> np.ndarray:
    """(8, 8) int64 piece codes of a parsed board."""
    return np.array([[PIECES.index(p) + 1 if p else 0 for p in row] for row in board], np.int64)


_LEVELS = torch.tensor([0.0] + [INTENSITY[p.lower()] for p in PIECES])
_MASK = torch.zeros(8, 8)
_MASK[1:7, 1:7] = 1.0  # centred 6x6 patch per square


def render_codes(c: torch.Tensor) -> torch.Tensor:
    """Clips (..., 3, 64, 64) from piece codes (..., 8, 8), on c's device; equals render() of the same board."""
    v = _LEVELS.to(c.device)[c]
    white, black = torch.where(c <= 6, v, 0.0), torch.where(c > 6, v, 0.0)
    x = torch.stack([white, black, white + black], -3)  # (..., 3, 8, 8); a square holds one piece
    up = x[..., :, None, :, None] * _MASK.to(c.device)[:, None, :]  # (..., 3, 8 rows, 8 py, 8 cols, 8 px)
    return up.reshape(*x.shape[:-2], 64, 64)


class ChessPositions:
    """Piece codes: obs (n, 8, 8), fut (n, 3 moves, 8, 8); state (n, 3, 4). Render with render_codes().

    Needs fetch_chess.py. Train has 10000 positions: with 800 the particle readout memorised its train scenes
    (diagnostic 2026-11-22), and chess shares the readout.
    """

    _SLICES: ClassVar[dict[str, slice]] = {
        "train": slice(0, 10000),
        "dev": slice(10000, 10080),
        "test": slice(10080, N),
    }

    def __init__(self, split: Literal["train", "dev", "test"], root: str = "data"):
        if split not in self._SLICES:
            raise ValueError(f"unknown split {split!r}")
        path = Path(root) / PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python scripts/fetch_chess.py --root {root}")
        if SHA256 != "PENDING" and sha256(path) != SHA256:
            raise ValueError(f"{path} differs from the pinned fetch (sha256): re-run scripts/fetch_chess.py")
        data = np.load(path)
        fens, moves = data["fen"][self._SLICES[split]], data["moves"][self._SLICES[split]]
        boards = [parse_fen(str(f)) for f in fens]
        self.fens, self.moves = fens, moves
        self.obs = torch.tensor(np.stack([codes(b) for b in boards]))
        self.fut = torch.tensor(np.stack([[codes(apply_move(b, str(m))) for m in ms] for b, ms in zip(boards, moves)]))
        self.state = torch.tensor([[move_state(str(m)) for m in ms] for ms in moves], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.obs)
