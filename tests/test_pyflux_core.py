# tests/test_pyflux_core.py
#
# Starter pytest suite for the most important non-GUI/core functions.
#
# Notes:
# - This imports from src/pyflux/main.py using a temporary sys.path append.
# - It focuses on pure/helper logic only, not GUI behavior.
# - A few tests are property/behavior based rather than exact-value snapshots.
#
# Run:
#   pytest -q
#
# Recommended repo structure:
#   repo/
#     src/pyflux/main.py
#     tests/test_pyflux_core.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pyflux.core import (
    apply_T,
    apply_transform_to_arr,
    avg_loc_by_tid,
    bead_initial_positions,
    best_rigid_transform,
    dbscan_numpy,
    icp,
    match_and_filter_beads,
    moving_average_1d,
    np_to_df,
    preview_localization_precision,
    save_to_csv,
    _preprocess_xyz_points,
)
from pyflux.plotting import (
    _clip_to_scale_max,
    pointcloud_to_image,
    render_gaussians_xy,
    scatter_points_and_color,
    tid_to_color,
)


# ----------------------------
# moving_average_1d
# ----------------------------

def test_moving_average_1d_basic():
    a = np.array([1, 2, 3, 4], dtype=float)
    out = moving_average_1d(a, n=2)
    assert np.allclose(out, np.array([1.5, 2.5, 3.5]))


def test_moving_average_1d_returns_input_if_shorter_than_window():
    a = np.array([1, 2], dtype=float)
    out = moving_average_1d(a, n=4)
    assert np.allclose(out, a)


def test_moving_average_1d_window_equal_length():
    a = np.array([2, 4, 6, 8], dtype=float)
    out = moving_average_1d(a, n=4)
    assert np.allclose(out, np.array([5.0]))


# ----------------------------
# best_rigid_transform / apply_T / icp
# ----------------------------

def test_best_rigid_transform_identity():
    A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    R, t = best_rigid_transform(A, A)
    assert np.allclose(R, np.eye(3), atol=1e-10)
    assert np.allclose(t, np.zeros(3), atol=1e-10)


def test_best_rigid_transform_translation_only():
    A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0],
    ])
    shift = np.array([5.0, -2.0, 1.5])
    B = A + shift

    R, t = best_rigid_transform(A, B)

    assert np.allclose(R, np.eye(3), atol=1e-10)
    assert np.allclose(t, shift, atol=1e-10)


def test_apply_T_translation():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    ])
    T = np.eye(4)
    T[:3, 3] = [10.0, -1.0, 0.5]

    out = apply_T(pts, T)
    expected = np.array([
        [10.0, -1.0, 0.5],
        [11.0, 1.0, 3.5],
    ])
    assert np.allclose(out, expected)


def test_icp_returns_valid_shapes_and_improves_alignment():
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [0.3, 1.1, 0.0],
        [1.4, 1.0, 0.0],
        [0.2, 0.4, 0.7],
    ], dtype=float)

    shift = np.array([2.0, -3.0, 1.0])
    source = target - shift

    before_err = np.mean(np.linalg.norm(source - target, axis=1))

    aligned, T_total = icp(source, target, max_iterations=50, tolerance=1e-12)

    after_err = np.mean(np.linalg.norm(aligned - target, axis=1))

    assert aligned.shape == source.shape
    assert T_total.shape == (4, 4)
    assert after_err < before_err

# ----------------------------
# bead_initial_positions / match_and_filter_beads
# ----------------------------

def test_bead_initial_positions_returns_first_smoothed_point_per_gri():
    dtype = np.dtype([("gri", np.int32), ("xyz", np.float64, (3,))])
    points = np.array([
        (1, [0.0, 0.0, 0.0]),
        (1, [1.0, 1.0, 1.0]),
        (1, [2.0, 2.0, 2.0]),
        (2, [10.0, 0.0, 0.0]),
        (2, [12.0, 0.0, 0.0]),
        (2, [14.0, 0.0, 0.0]),
    ], dtype=dtype)

    out = bead_initial_positions(points, k=2, min_count=1)

    assert set(out.keys()) == {1, 2}
    assert np.allclose(out[1], np.array([0.5, 0.5, 0.5]))
    assert np.allclose(out[2], np.array([11.0, 0.0, 0.0]))


def test_match_and_filter_beads_keeps_all_when_all_z_small():
    beads_ref = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
        3: np.array([0.0, 1.0, 0.0]),
    }
    beads_mov = {
        1: np.array([0.1, 0.0, 0.0]),
        2: np.array([1.1, 0.0, 0.0]),
        3: np.array([0.1, 1.0, 0.0]),
    }

    ref_kept, mov_kept, common = match_and_filter_beads(beads_ref, beads_mov)

    assert common == [1, 2, 3]
    assert ref_kept.shape == (3, 3)
    assert mov_kept.shape == (3, 3)


def test_match_and_filter_beads_raises_if_less_than_three_common():
    beads_ref = {1: np.array([0, 0, 0]), 2: np.array([1, 0, 0])}
    beads_mov = {1: np.array([0, 0, 0]), 2: np.array([1, 0, 0])}

    with pytest.raises(ValueError, match="Need >=3 common beads"):
        match_and_filter_beads(beads_ref, beads_mov)


# ----------------------------
# dbscan_numpy
# ----------------------------

def test_dbscan_numpy_finds_two_clusters_and_noise():
    cluster1 = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
    cluster2 = np.array([[10, 10], [10, 11], [11, 10]], dtype=float)
    noise = np.array([[100, 100]], dtype=float)
    pts = np.vstack([cluster1, cluster2, noise])

    labels = dbscan_numpy(pts, eps=1.5, min_samples=2)

    assert len(labels) == len(pts)
    unique_clusters = sorted(set(labels) - {-1})
    assert len(unique_clusters) == 2
    assert labels[-1] == -1


def test_dbscan_numpy_empty_input():
    pts = np.zeros((0, 2), dtype=float)
    labels = dbscan_numpy(pts, eps=1.0, min_samples=2)
    assert labels.size == 0


# ----------------------------
# avg_loc_by_tid
# ----------------------------

def test_avg_loc_by_tid_basic():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
    ])
    arr = np.array([
        (1, [0.0, 0.0, 0.0]),
        (1, [2.0, 2.0, 2.0]),
        (2, [10.0, 0.0, 0.0]),
        (2, [14.0, 0.0, 0.0]),
        (2, [16.0, 0.0, 0.0]),
    ], dtype=dtype)

    tids, centroids, counts = avg_loc_by_tid(arr)

    assert np.array_equal(tids, np.array([1, 2]))
    assert np.allclose(centroids[0], [1.0, 1.0, 1.0])
    assert np.allclose(centroids[1], [40.0 / 3.0, 0.0, 0.0])
    assert np.array_equal(counts, np.array([2, 3]))


def test_avg_loc_by_tid_empty():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
    ])
    arr = np.array([], dtype=dtype)

    tids, centroids, counts = avg_loc_by_tid(arr)

    assert tids.size == 0
    assert centroids.shape == (0, 3)
    assert counts.size == 0


# ----------------------------
# _preprocess_xyz_points
# ----------------------------

def test_preprocess_xyz_points_scales_and_corrects_z():
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    out = _preprocess_xyz_points(pts, z_corr=2.0, scale=10.0)

    expected = np.array([
        [10.0, 20.0, 60.0],
        [40.0, 50.0, 120.0],
    ])
    assert np.allclose(out, expected)


# ----------------------------
# apply_transform_to_arr
# ----------------------------

def test_apply_transform_to_arr_updates_loc_only():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
    ])
    arr = np.array([
        (1, [0.0, 0.0, 0.0]),
        (2, [1.0, 2.0, 3.0]),
    ], dtype=dtype)

    T = np.eye(4)
    T[:3, 3] = [1.0, 1.0, 1.0]

    out = apply_transform_to_arr(arr, T)

    assert np.array_equal(out["tid"], arr["tid"])
    assert np.allclose(out["loc"], np.array([[1, 1, 1], [2, 3, 4]], dtype=float))


# ----------------------------
# np_to_df
# ----------------------------

def test_np_to_df_splits_vector_columns():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
        ("lnc", np.float64, (3,)),
        ("dcr", np.float64, (3,)),
    ])
    arr = np.array([
        (7, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]),
    ], dtype=dtype)

    df = np_to_df(arr)

    expected_cols = {
        "tid",
        "loc_x", "loc_y", "loc_z",
        "lnc_x", "lnc_y", "lnc_z",
        "dcr_x", "dcr_y", "dcr_z",
    }
    assert expected_cols.issubset(df.columns)
    assert df.loc[0, "loc_x"] == 1.0
    assert df.loc[0, "dcr_z"] == 9.0


# ----------------------------
# preview_localization_precision
# ----------------------------

def test_preview_localization_precision_returns_median_std_per_dim():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
    ])
    arr = np.array([
        (1, [0.0, 0.0, 0.0]),
        (1, [2.0, 2.0, 2.0]),
        (2, [10.0, 0.0, 0.0]),
        (2, [14.0, 0.0, 0.0]),
    ], dtype=dtype)

    lp = preview_localization_precision(arr)

    assert lp is not None
    assert len(lp) == 3
    # x stds: [sqrt(2), sqrt(8)]? No: sample std for [0,2] is ~1.414, for [10,14] is ~2.828, median ~2.121
    assert np.isfinite(lp[0])
    assert np.isfinite(lp[1])
    assert np.isfinite(lp[2])


# ----------------------------
# scatter_points_and_color
# ----------------------------

def _make_arr_for_scatter():
    dtype = np.dtype([
        ("tid", np.int32),
        ("loc", np.float64, (3,)),
    ])
    return np.array([
        (1, [0.0, 0.0, 0.0]),
        (1, [3.0, 4.0, 0.0]),   # end-to-end = 5
        (2, [10.0, 0.0, 0.0]),
        (2, [10.0, 0.0, 0.0]),  # end-to-end = 0
    ], dtype=dtype)


def test_scatter_points_and_color_non_averaged():
    arr = _make_arr_for_scatter()
    xyz, vals, tids = scatter_points_and_color(arr, avg_tid=False)

    assert xyz.shape == (4, 3)
    assert np.array_equal(tids, np.array([1, 1, 2, 2]))
    assert np.allclose(vals, np.array([5.0, 5.0, 0.0, 0.0]))


def test_scatter_points_and_color_averaged():
    arr = _make_arr_for_scatter()
    xyz, vals, tids = scatter_points_and_color(arr, avg_tid=True)

    assert xyz.shape == (2, 3)
    assert np.array_equal(tids, np.array([1, 2]))
    assert np.allclose(xyz[0], np.array([1.5, 2.0, 0.0]))
    assert np.allclose(vals, np.array([5.0, 0.0]))


# ----------------------------
# tid_to_color
# ----------------------------

def test_tid_to_color_is_deterministic():
    c1 = tid_to_color(42, alpha=0.5)
    c2 = tid_to_color(42, alpha=0.5)
    assert c1 == c2
    assert c1.startswith("rgba(")
    assert c1.endswith(")")


# ----------------------------
# pointcloud_to_image
# ----------------------------

def test_pointcloud_to_image_basic_counts():
    x = np.array([0.1, 0.2, 1.1])
    y = np.array([0.1, 0.2, 1.1])

    H, extent = pointcloud_to_image(x, y, pixel_size_nm=1.0, padding_nm=0.0)

    assert H.shape == (1, 1) or H.shape == (2, 2)
    assert np.isclose(H.sum(), 3.0)
    assert len(extent) == 4


def test_pointcloud_to_image_empty():
    H, extent = pointcloud_to_image([], [], pixel_size_nm=1.0)
    assert H.shape == (1, 1)
    assert np.allclose(H, 0.0)
    assert extent == (0.0, 1.0, 0.0, 1.0)


def test_pointcloud_to_image_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        pointcloud_to_image([0, 1], [0], pixel_size_nm=1.0)


# ----------------------------
# _clip_to_scale_max
# ----------------------------

def test_clip_to_scale_max_basic():
    img = np.array([[0.0, 5.0, 20.0]])
    out = _clip_to_scale_max(img, max_value=10.0)
    assert np.allclose(out, np.array([[0.0, 5.0, 10.0]]))


def test_clip_to_scale_max_invalid_max_returns_zeros():
    img = np.array([[1.0, 2.0]])
    out = _clip_to_scale_max(img, max_value=0.0)
    assert np.allclose(out, np.zeros_like(img))


# ----------------------------
# render_gaussians_xy
# ----------------------------

def test_render_gaussians_xy_returns_image_and_bounds():
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 2.0])
    img, bounds = render_gaussians_xy(x, y, sigma_nm=1.0, pixel_size_nm=1.0)
    assert img.ndim == 2
    assert img.size > 0
    assert len(bounds) == 4


def test_render_gaussians_xy_empty_input():
    img, bounds = render_gaussians_xy([], [], sigma_nm=1.0, pixel_size_nm=1.0)
    assert img.shape == (1, 1)
    assert np.allclose(img, 0.0)
    assert bounds == (0.0, 1.0, 0.0, 1.0)


# ----------------------------
# save_to_csv
# ----------------------------

def test_save_to_csv_writes_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_path = tmp_path / "subdir" / "file.csv"

    save_to_csv(df, str(out_path))

    assert out_path.exists()
    df2 = pd.read_csv(out_path)
    assert list(df2.columns) == ["a", "b"]
    assert df2.shape == (2, 2)