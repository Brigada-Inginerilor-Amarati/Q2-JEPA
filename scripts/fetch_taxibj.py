"""Download OpenSTL's preprocessed TaxiBJ (559 MB zip, 2.6 GB dataset.npz) to <root>/taxibj/dataset.npz.

python scripts/fetch_taxibj.py [--root data]

Source: the OpenSTL v0.1.0 release (tools/prepare_data/download_taxibj.sh in chengtan9907/OpenSTL). The file is
checked against data/taxibj.py's SHA256.
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.data.taxibj import PATH, SHA256, URL


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data")
    args = ap.parse_args(argv)
    out = Path(args.root) / PATH
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        zpath = out.parent / "taxibj_dataset.zip"
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("dataset.npz"))
            with z.open(name) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
        zpath.unlink()
    got = file_sha256(out)
    if got != SHA256:
        raise SystemExit(f"{out}: sha256 {got} differs from the pinned {SHA256}")
    print(f"{out} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
