"""Fetch D3IL Avoiding from the official Drive link and convert it once to data/d3il/avoiding.npz.

uv run --with gdown python scripts/fetch_d3il.py

The download is untrusted: it stays in data/d3il/raw/, and nothing outside this script reads it. Only stdlib and numpy
are imported at the top because the conversion runs in its own `python -I` process.
"""

import argparse
import hashlib
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

DRIVE_ID = "1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8"  # ALRhub/d3il README
ZIP_SHA256 = "f75acb9528bf272b2780a4ae42e199f76960d675d8039f2ff351d0cd5a18816a"
RAW = Path("data/d3il/raw")
OUT = Path("data/d3il/avoiding.npz")
ZIP_NAME = "d3il_data.zip"
# Every class the 96 Avoiding pickles reference, per pickletools. numpy._core is numpy 2's name for numpy.core.
ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def safe_load(path: Path) -> object:
    """Unpickle allowing only plain containers and numpy arrays, so a hostile file cannot run code."""
    with path.open("rb") as f:
        return _NumpyOnly(f).load()


def download(raw: Path = RAW) -> Path:
    """Download once, check the pinned SHA-256, and extract only the Avoiding members."""
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / ZIP_NAME
    if not archive.exists():
        import gdown

        gdown.download(id=DRIVE_ID, output=str(archive), quiet=False)
    digest = sha256(archive)
    print(f"sha256 {digest}")
    if ZIP_SHA256 and digest != ZIP_SHA256:
        raise ValueError(f"{archive}: sha256 {digest} does not match pinned {ZIP_SHA256}")
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if "avoiding/" in m.lower()]
        z.extractall(raw, members)
    return raw


def extract_xy(obj: dict) -> np.ndarray:
    """Measured planar position (T, 2) in metres: c_pos, the field whose 96 starts coincide as App. B.4 says."""
    return obj["robot"]["c_pos"][:, :2]


def convert(raw: Path, out: Path) -> None:
    """Write xy (N, 2) and offsets (demos + 1,): demo i is xy[offsets[i]:offsets[i + 1]], demos in filename order."""
    paths = [extract_xy(safe_load(f)) for f in sorted(raw.glob("avoiding/data/*.pkl"))]
    offsets = np.cumsum([0, *map(len, paths)], dtype=np.int64)
    np.savez(out, xy=np.concatenate(paths).astype(np.float64), offsets=offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="only download and extract")
    parser.add_argument("--convert", nargs=2, metavar=("RAW", "OUT"), help="internal: runs under python -I")
    args = parser.parse_args()
    if args.convert:
        convert(Path(args.convert[0]), Path(args.convert[1]))
    else:
        download()
        if not args.download:
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--convert",
                str(RAW.resolve()),
                str(OUT.resolve()),
            ]
            with tempfile.TemporaryDirectory() as isolated:
                subprocess.run(command, cwd=isolated, check=True)
