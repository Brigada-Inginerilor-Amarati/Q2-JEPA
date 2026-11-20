import hashlib
import os
import pickle
import zipfile

import numpy as np
import pytest

from scripts import fetch_d3il
from scripts.fetch_d3il import convert, download, extract_xy, safe_load, sha256


class _Hostile:
    def __reduce__(self):
        return (os.getcwd, ())


def _demo(path, xy, other=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.zeros((len(xy), 1))
    des = np.concatenate([xy, z], -1)
    cur = des + 1.0 if other is None else other
    path.write_bytes(pickle.dumps({"robot": {"des_c_pos": des, "c_pos": cur}, "time_stamp": np.arange(len(xy))}))


def test_safe_load_refuses_hostile_pickle(tmp_path):
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps(_Hostile()))
    with pytest.raises(pickle.UnpicklingError):
        safe_load(path)


def test_safe_load_reads_numpy_arrays(tmp_path):
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"a": {"b": np.arange(3.0)}}))
    np.testing.assert_array_equal(safe_load(path)["a"]["b"], np.arange(3.0))


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"d3il")
    assert sha256(path) == hashlib.sha256(b"d3il").hexdigest()


def test_extract_xy_uses_measured_position():
    xy = np.arange(6.0).reshape(3, 2)
    obj = {"robot": {"c_pos": np.concatenate([xy, np.zeros((3, 1))], -1), "des_c_pos": np.zeros((3, 3))}}
    np.testing.assert_array_equal(extract_xy(obj), xy)


def test_convert_writes_plain_arrays(tmp_path):
    a, b = np.arange(6.0).reshape(3, 2), np.arange(10.0, 18.0).reshape(4, 2)
    _demo(tmp_path / "avoiding/data/env_001.pkl", b, other=np.concatenate([b, np.zeros((4, 1))], -1))
    _demo(tmp_path / "avoiding/data/env_000.pkl", a, other=np.concatenate([a, np.zeros((3, 1))], -1))
    out = tmp_path / "avoiding.npz"
    convert(tmp_path, out)
    data = np.load(out, allow_pickle=False)
    assert data["offsets"].tolist() == [0, 3, 7] and data["xy"].dtype == np.float64
    np.testing.assert_array_equal(data["xy"], np.concatenate([a, b]))


def test_convert_refuses_hostile_demo(tmp_path):
    path = tmp_path / "avoiding/data/env_000.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(_Hostile()))
    with pytest.raises(pickle.UnpicklingError):
        convert(tmp_path, tmp_path / "out.npz")


def _fake_zip(raw):
    raw.mkdir()
    with zipfile.ZipFile(raw / fetch_d3il.ZIP_NAME, "w") as z:
        z.writestr("avoiding/data/env_000.pkl", b"x")
        z.writestr("sorting/2_modes.pkl", b"y")


def test_download_rejects_wrong_sha(tmp_path):
    _fake_zip(tmp_path / "raw")
    with pytest.raises(ValueError, match="sha256"):
        download(tmp_path / "raw")


def test_download_extracts_only_avoiding(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _fake_zip(raw)
    monkeypatch.setattr(fetch_d3il, "ZIP_SHA256", sha256(raw / fetch_d3il.ZIP_NAME))
    download(raw)
    assert (raw / "avoiding/data/env_000.pkl").exists() and not (raw / "sorting").exists()
