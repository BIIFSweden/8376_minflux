# tests/test_legacy_minflux.py
#
# Reading the legacy MINFLUX .npy layout (Imspector < 24.10, and what pyMINFLUX
# saves): one row per localization, iterations nested under `itr`. It is
# converted to the flat one-row-per-iteration layout the rest of pyflux reads.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pyflux.core import (  # noqa: E402
    MinfluxLayoutError,
    flatten_legacy_minflux,
    is_legacy_minflux,
    legacy_flat_dtype,
    load_minflux_npy,
)

ITERATION_DTYPE = [
    ("itr", "<i4"), ("tic", "<u8"), ("loc", "<f8", (3,)), ("lnc", "<f8", (3,)),
    ("eco", "<i4"), ("ecc", "<i4"), ("efo", "<f8"), ("efc", "<f8"), ("sta", "<i4"),
    ("cfr", "<f8"), ("dcr", "<f8"), ("ext", "<f8", (3,)), ("fbg", "<f8"),
]


def _legacy(n_iterations=10, trace_lengths=(1, 3, 4, 2), invalid=(0, -1), seed=0, fluo=False):
    """A legacy array shaped like a real Imspector export.

    Records are localizations, traces are runs of equal `tid`. Like the real
    files, a trace's later localizations re-measure only the last iterations:
    the earlier ones are NaN, with `cfr` reading -0.0. CFR varies at the
    second-to-last iteration and is constant at the last, as in a 3-D
    sequence. Records listed in *invalid* get ``vld = False``.
    """
    rng = np.random.default_rng(seed)
    top = [("itr", ITERATION_DTYPE, (n_iterations,)), ("sqi", "<u4"), ("gri", "<u4"),
           ("tim", "<f8"), ("tid", "<i4"), ("vld", "?"), ("act", "?"), ("dos", "<i4"), ("sky", "<i4")]
    if fluo:
        top.append(("fluo", "i1"))
    n = int(sum(trace_lengths))
    a = np.zeros(n, dtype=top)
    a["tid"] = np.repeat(np.arange(len(trace_lengths)) + 100, trace_lengths)
    a["tim"] = np.sort(rng.uniform(0, 600, n))
    a["vld"] = True
    for i in invalid:
        a["vld"][i] = False
    a["gri"] = rng.integers(0, 5, n)
    a["sqi"] = 7
    it = a["itr"]
    it["itr"] = np.arange(n_iterations)
    it["loc"] = rng.uniform(-5e-6, 5e-6, (n, n_iterations, 3))
    it["efo"] = rng.uniform(1e4, 1e5, (n, n_iterations))
    it["eco"] = rng.integers(10, 500, (n, n_iterations))
    it["cfr"] = -0.0
    it["cfr"][:, n_iterations - 2] = rng.uniform(0, 1, n)
    first_of_trace = np.r_[True, a["tid"][1:] != a["tid"][:-1]]
    later = np.flatnonzero(~first_of_trace)
    relocalize_from = n_iterations - 4 if n_iterations >= 4 else 0
    for name in ("loc", "lnc", "efo", "efc", "dcr", "fbg"):
        block = it[name]
        block[later[:, None], np.arange(relocalize_from)] = np.nan
    if fluo:
        a["fluo"] = rng.integers(1, 3, n)
    a["itr"] = it
    return a


# ----------------------------
# layout detection
# ----------------------------

def test_detects_the_nested_layout():
    assert is_legacy_minflux(_legacy().dtype)


def test_the_flat_layout_is_not_legacy():
    flat = np.zeros(3, dtype=[("itr", "<i4"), ("loc", "<f8", (3,)), ("vld", "?")])
    assert not is_legacy_minflux(flat.dtype)
    assert not is_legacy_minflux(np.zeros(3).dtype)


def test_flat_columns_follow_the_new_layout():
    names = legacy_flat_dtype(_legacy(fluo=True).dtype).names
    assert names == ("vld", "sta", "tim", "tid", "gri", "sqi", "itr", "loc", "lnc",
                     "eco", "ecc", "efo", "efc", "fbg", "cfr", "dcr", "fluo")


def test_a_missing_iteration_field_is_named():
    a = _legacy()
    dt = [(n, a.dtype.fields[n][0]) for n in a.dtype.names if n != "itr"]
    nested = [f for f in ITERATION_DTYPE if f[0] != "efo"]
    broken = np.zeros(2, dtype=[("itr", nested, (5,))] + dt)
    with pytest.raises(MinfluxLayoutError, match="itr.efo"):
        flatten_legacy_minflux(broken)


# ----------------------------
# conversion
# ----------------------------

def test_one_row_per_iteration_of_each_valid_localization():
    a = _legacy()
    flat = flatten_legacy_minflux(a)
    valid = a[a["vld"]]
    k = a["itr"].shape[1]
    assert len(flat) == len(valid) * k
    assert flat["vld"].all()
    assert np.array_equal(flat["itr"], np.tile(np.arange(k), len(valid)))
    assert np.array_equal(flat["tid"], np.repeat(valid["tid"], k))
    assert np.array_equal(flat["tim"], np.repeat(valid["tim"], k))
    np.testing.assert_array_equal(flat["loc"], valid["itr"]["loc"].reshape(-1, 3))
    np.testing.assert_array_equal(flat["efo"], valid["itr"]["efo"].ravel())
    np.testing.assert_array_equal(flat["cfr"], valid["itr"]["cfr"].ravel())


def test_selecting_an_iteration_gives_every_localization_once_in_order():
    a = _legacy()
    flat = flatten_legacy_minflux(a)
    valid = a[a["vld"]]
    k = a["itr"].shape[1]
    final = flat[flat["itr"] == k - 1]
    np.testing.assert_array_equal(final["loc"], valid["itr"]["loc"][:, -1])
    assert np.array_equal(final["tid"], valid["tid"])
    cfr_rows = flat[flat["itr"] == k - 2]
    # row j of both selections is the same localization, which is what pyflux's
    # (tid, occurrence) join relies on
    assert np.array_equal(cfr_rows["tid"], final["tid"])
    np.testing.assert_array_equal(cfr_rows["cfr"], valid["itr"]["cfr"][:, -2])


def test_invalid_localizations_can_be_kept():
    a = _legacy()
    flat = flatten_legacy_minflux(a, valid_only=False)
    assert len(flat) == len(a) * a["itr"].shape[1]
    assert flat["vld"].sum() == a["vld"].sum() * a["itr"].shape[1]


def test_chunking_does_not_change_the_result():
    a = _legacy(trace_lengths=(3, 1, 5, 2, 4), invalid=(0, 4, 9))
    whole = flatten_legacy_minflux(a, chunk=10_000)
    for size in (1, 2, 3, 7):
        part = flatten_legacy_minflux(a, chunk=size)
        for name in whole.dtype.names:
            np.testing.assert_array_equal(part[name], whole[name])


def test_aggregated_acquisitions_have_one_iteration():
    a = _legacy(n_iterations=1)
    flat = flatten_legacy_minflux(a)
    assert len(flat) == a["vld"].sum()
    assert (flat["itr"] == 0).all()


def test_pyminflux_fluorophore_ids_survive():
    a = _legacy(fluo=True)
    flat = flatten_legacy_minflux(a)
    assert np.array_equal(flat["fluo"][flat["itr"] == 0], a["fluo"][a["vld"]])


def test_no_valid_localizations_gives_an_empty_flat_array():
    a = _legacy()
    a["vld"] = False
    flat = flatten_legacy_minflux(a)
    assert len(flat) == 0
    assert "loc" in flat.dtype.names


def test_cancel_stops_the_conversion():
    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        flatten_legacy_minflux(_legacy(), check_cancel=cancel)


# ----------------------------
# loading from disk
# ----------------------------

def test_load_converts_a_legacy_file_from_a_memory_map(tmp_path):
    a = _legacy()
    path = tmp_path / "legacy.npy"
    np.save(path, a)
    flat, info = load_minflux_npy(str(path), chunk=2)
    assert info["layout"] == "legacy"
    assert info["n_localizations"] == len(a)
    assert info["n_valid"] == int(a["vld"].sum())
    # the first and last localizations are invalid and so not in `flat`; the
    # acquisition span still covers them
    assert info["total_tim"] == pytest.approx(a["tim"][-1] - a["tim"][0])
    assert flat["tim"].max() - flat["tim"].min() < info["total_tim"]
    expected = flatten_legacy_minflux(a)
    for name in expected.dtype.names:
        np.testing.assert_array_equal(flat[name], expected[name])


def test_load_returns_a_flat_file_unchanged(tmp_path):
    flat_in = np.zeros(5, dtype=[("tim", "<f8"), ("vld", "?"), ("loc", "<f8", (3,)), ("itr", "<i4"),
                                 ("tid", "<i4"), ("efo", "<f8"), ("cfr", "<f8")])
    flat_in["tim"] = np.arange(5)
    path = tmp_path / "flat.npy"
    np.save(path, flat_in)
    for chunk in (2, 100):
        out, info = load_minflux_npy(str(path), chunk=chunk)
        assert info == {"layout": "flat"}
        assert np.array_equal(out, flat_in)


# ----------------------------
# through pyflux's own loader
# ----------------------------

def test_the_file_worker_reads_a_legacy_file(tmp_path, qapp):
    pytest.importorskip("pytestqt")
    from pyflux.gui import FileWorker

    a = _legacy(trace_lengths=(1, 3, 4, 2, 5), invalid=(0, -1))
    path = tmp_path / "legacy.npy"
    np.save(path, a)

    params = dict(min_trace_len=3, z_corr=0.7, scale=1e9, save_folder=str(tmp_path),
                  bin_size=3000.0, base="legacy")
    worker = FileWorker(str(path), params)
    seen, statuses = [], []
    worker.need_efo.connect(seen.append)
    worker.status.connect(statuses.append)
    result = worker._process_one_file()

    assert result is not None and seen, statuses
    ctx = seen[0]
    valid = a[a["vld"]]
    counts = {t: int((valid["tid"] == t).sum()) for t in np.unique(valid["tid"])}
    kept_tids = [t for t, c in counts.items() if c >= 3]
    expected = valid[np.isin(valid["tid"], kept_tids)]

    assert ctx["last_iteration_loc"] == len(valid)
    assert ctx["after_trace"] == len(expected)
    assert ctx["total_tim"] == pytest.approx(a["tim"][-1] - a["tim"][0])
    arr = ctx["arr"]
    assert np.array_equal(arr["tid"], expected["tid"])
    # final-iteration positions, in nm, z corrected and y flipped as for flat files
    loc = expected["itr"]["loc"][:, -1] * 1e9
    np.testing.assert_allclose(arr["loc"][:, 0], loc[:, 0])
    np.testing.assert_allclose(arr["loc"][:, 1], -loc[:, 1])
    np.testing.assert_allclose(arr["loc"][:, 2], loc[:, 2] * 0.7)
    np.testing.assert_allclose(arr["efo"], expected["itr"]["efo"][:, -1])
    # CFR comes from the last iteration where it varies -- the second-to-last
    # here -- for each localization, not from its trace's first one
    np.testing.assert_allclose(arr["cfr"], expected["itr"]["cfr"][:, -2])
    assert any("legacy layout" in s for s in statuses)
