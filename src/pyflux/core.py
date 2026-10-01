import os
from collections import deque

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree


def save_to_csv(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def moving_average_1d(a, n=4):
    a = np.asarray(a, dtype=float)
    if len(a) < n:
        return a
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def best_rigid_transform(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def apply_T(points, T):
    ones = np.ones((points.shape[0], 1))
    P = np.hstack([points, ones])
    return (P @ T.T)[:, :3]


def icp(source, target, max_iterations=50, tolerance=0.5e-9):
    src = source.copy()
    T_total = np.eye(4)
    prev_err = np.inf
    tree = cKDTree(target)
    for _ in range(max_iterations):
        dists, idx = tree.query(src, k=1)
        corr = target[idx]
        R, t = best_rigid_transform(src, corr)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        src = apply_T(src, T)
        T_total = T @ T_total
        mean_err = float(np.mean(dists))
        if abs(prev_err - mean_err) < tolerance:
            break
        prev_err = mean_err
    return src, T_total


def load_mbm_points(grd_mbm_path):
    g = zarr.open_group(grd_mbm_path, mode="r")
    points = g["points"][:]
    # DEBUGGING: Flip y to match MINFLUX data convention.
    points["xyz"][:, 1] *= -1.0
    return points


def bead_initial_positions(points, k=4, min_count=10):
    gri_vals = np.asarray(points["gri"])
    xyz = np.asarray(points["xyz"])
    out = {}
    for gri in np.unique(gri_vals):
        mask = gri_vals == gri
        if mask.sum() <= min_count:
            continue
        bead_xyz = xyz[mask]
        x = moving_average_1d(bead_xyz[:, 0], n=k)
        y = moving_average_1d(bead_xyz[:, 1], n=k)
        z = moving_average_1d(bead_xyz[:, 2], n=k)
        if len(x) == 0:
            continue
        out[int(gri)] = np.array([x[0], y[0], z[0]], dtype=float)
    return out


def match_and_filter_beads(beads_ref, beads_mov, return_diagnostics: bool = False):
    common = sorted(set(beads_ref.keys()) & set(beads_mov.keys()))
    if len(common) < 3:
        raise ValueError(f"Need >=3 common beads, got {len(common)}")

    ref = np.vstack([beads_ref[g] for g in common])
    mov = np.vstack([beads_mov[g] for g in common])

    z = ref[:, 2]
    if np.std(z) > 100e-9:
        keep = z < np.median(z) + 1.5 * np.std(z)
    else:
        keep = z < np.median(z) + 100e-9

    common_kept = [g for g, k in zip(common, keep) if k]
    if len(common_kept) < 3:
        raise ValueError(f"After z-filter need >=3 beads, got {len(common_kept)}")

    ref_kept = np.vstack([beads_ref[g] for g in common_kept])
    mov_kept = np.vstack([beads_mov[g] for g in common_kept])
    if return_diagnostics:
        diagnostics = {
            "common_ids": [int(g) for g in common],
            "keep_mask": np.asarray(keep, dtype=bool),
            "ref_common": np.asarray(ref, dtype=float),
            "mov_common": np.asarray(mov, dtype=float),
            "common_kept": [int(g) for g in common_kept],
        }
        return ref_kept, mov_kept, common_kept, diagnostics
    return ref_kept, mov_kept, common_kept


def dbscan_numpy(points, eps=200.0, min_samples=3):
    """
    Simple DBSCAN using cKDTree neighborhood queries.
    points: (N, D) array
    Returns labels (N,), where -1 means noise.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return np.array([], dtype=int)

    eps = float(max(eps, 1e-12))
    min_samples = int(max(1, min_samples))

    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=eps)

    labels = np.full(len(pts), -99, dtype=int)  # unvisited
    cluster_id = 0

    for i in range(len(pts)):
        if labels[i] != -99:
            continue

        if len(neighbors[i]) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = deque(j for j in neighbors[i] if j != i)
        seed_set = set(seeds)

        while seeds:
            j = seeds.popleft()

            if labels[j] == -1:
                labels[j] = cluster_id

            if labels[j] != -99:
                continue

            labels[j] = cluster_id

            if len(neighbors[j]) >= min_samples:
                for k in neighbors[j]:
                    if k not in seed_set:
                        seeds.append(k)
                        seed_set.add(k)

        cluster_id += 1

    labels[labels == -99] = -1
    return labels


def avg_loc_by_tid(arr):
    """
    Returns track-level centroids and counts.
    Output tuple:
      track_ids: (T,)
      centroids_xyz: (T, 3)
      n_localizations: (T,)
    """
    if arr is None or len(arr) == 0:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float), np.array([], dtype=int)

    names = arr.dtype.names or ()  # None for a plain, non-structured array
    if "tid" not in names or "loc" not in names:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float), np.array([], dtype=int)

    tids = np.asarray(arr["tid"])
    loc = np.asarray(arr["loc"], dtype=float)
    if loc.ndim != 2 or loc.shape[1] < 2:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float), np.array([], dtype=int)

    use_dims = min(3, loc.shape[1])
    valid = np.isfinite(tids)
    valid &= np.isfinite(loc[:, 0]) & np.isfinite(loc[:, 1])
    if use_dims >= 3:
        valid &= np.isfinite(loc[:, 2])

    tids = tids[valid]
    loc = loc[valid]
    if len(tids) == 0:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float), np.array([], dtype=int)

    tids = tids.astype(np.int64)
    unique_tids, inv_idx, counts = np.unique(tids, return_inverse=True, return_counts=True)

    centroids = np.zeros((len(unique_tids), 3), dtype=float)
    for d in range(use_dims):
        sums = np.bincount(inv_idx, weights=loc[:, d], minlength=len(unique_tids))
        centroids[:, d] = sums / np.maximum(counts, 1)

    return unique_tids.astype(int), centroids, counts.astype(int)


def _preprocess_xyz_points(points, *, z_corr=1.0, scale=1.0):
    pts = np.asarray(points, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] < 3:
        return pts
    pts[:, 2] *= float(z_corr)
    pts *= float(scale)
    return pts


def compute_mbm_transform(mbm_ref_dir, mbm_mov_dir, k=4, scale=1.0, z_corr=1.0, return_diagnostics: bool = False):
    pts_ref = load_mbm_points(mbm_ref_dir)
    pts_mov = load_mbm_points(mbm_mov_dir)

    beads_ref = bead_initial_positions(pts_ref, k=k)
    beads_mov = bead_initial_positions(pts_mov, k=k)

    diagnostics = None
    if return_diagnostics:
        ref_pts, mov_pts, common, diagnostics = match_and_filter_beads(
            beads_ref, beads_mov, return_diagnostics=True
        )
    else:
        ref_pts, mov_pts, common = match_and_filter_beads(beads_ref, beads_mov)

    # Match MBM bead coordinates to the same coordinate frame used by loaded MINFLUX data.
    ref_pts = _preprocess_xyz_points(ref_pts, z_corr=z_corr, scale=scale)
    mov_pts = _preprocess_xyz_points(mov_pts, z_corr=z_corr, scale=scale)

    if diagnostics is not None:
        diagnostics["ref_common_scaled"] = _preprocess_xyz_points(
            diagnostics["ref_common"], z_corr=z_corr, scale=scale
        )
        diagnostics["mov_common_scaled"] = _preprocess_xyz_points(
            diagnostics["mov_common"], z_corr=z_corr, scale=scale
        )

    _, T_total = icp(mov_pts, ref_pts, max_iterations=50, tolerance=0.5e-9 * scale)
    if return_diagnostics:
        return T_total, common, diagnostics
    return T_total, common


def apply_transform_to_arr(arr, T):
    out = arr.copy()
    out_loc = apply_T(np.asarray(out["loc"], dtype=float), T)
    out["loc"] = out_loc
    return out


def np_to_df(np_data):
    df = pd.DataFrame(np_data.tolist(), columns=np_data.dtype.names)

    column_to_split = ["loc", "lnc", "dcr"]
    vals = ["x", "y", "z"]

    for column in column_to_split:
        # Split only what is a vector in this file: `dcr` is two values per row
        # in the Imspector >= 24.10 layout and one in the legacy layout.
        if column not in df.columns or len(df) == 0 or not np_data.dtype.fields[column][0].shape:
            continue
        n = len(df[column].iloc[0])
        cols = [f"{column}_{vals[i]}" if i < len(vals) else f"{column}_{i}" for i in range(n)]
        split_data = pd.DataFrame(df[column].tolist(), columns=cols)
        df = pd.concat([df.drop(column, axis=1), split_data], axis=1)
    return df


def _camera_slim(cam: dict) -> dict:
    if not isinstance(cam, dict):
        return cam
    out = {}
    for k in ("eye", "up", "center", "projection"):
        if k in cam:
            out[k] = cam[k]
    return out


def preview_localization_precision(arr):
    if arr is None or len(arr) == 0:
        return None
    locs = arr["loc"]
    tids = arr["tid"]
    n_dim = locs.shape[1]
    meds = []
    ut = np.unique(tids)
    for d in range(n_dim):
        stds = []
        for tid in ut:
            pts = locs[tids == tid, d]
            if len(pts) >= 2:
                stds.append(float(np.std(pts, ddof=1)))
        meds.append(float(np.median(stds)) if stds else float("nan"))
    return tuple(meds)


# ----------------------------------------------------------------------
# MINFLUX .npy layouts
# ----------------------------------------------------------------------
#
# Imspector has written two .npy layouts.
#
# * Imspector >= 24.10 writes one row per *iteration*: `itr` is an integer
#   column and `loc`, `efo`, `cfr`, ... sit beside it.  pyflux's pipeline is
#   written against this layout.
# * Older versions write one row per *localization*, with the iterations nested
#   under `itr` as a structured sub-array of length K -- 5 for a 2-D sequence,
#   10 for 3-D, 1 for an aggregated acquisition.  The same layout is what
#   pyMINFLUX reads with its v1 reader and saves back out (adding `fluo`).
#
# A legacy file is flattened into the first layout, one row per (valid
# localization, iteration), so the rest of pyflux reads both without knowing
# which it got.  Every valid localization contributes exactly one row per
# iteration, in file order, so selecting one iteration and pairing it with
# another by (tid, occurrence) -- which is how the CFR of the final
# localizations is looked up -- matches rows of the same localization.
#
# Iterations a later localization of a trace did not re-measure are NaN in the
# legacy file (their `cfr` reads -0.0); they are carried over as they are, the
# same values pyMINFLUX would read.


class MinfluxLayoutError(ValueError):
    """The file is not a MINFLUX export pyflux can read."""


#: Output columns, in the order Imspector >= 24.10 writes them.  Columns only
#: one layout has are left out: `fnl`, `bot`, `eot` and `thi` do not exist in
#: the legacy layout, and `act`, `dos`, `sky`, `tic`, `ext`, `gvx`/`gvy`,
#: `eox`/`eoy`, `dmz` and `lcx`/`lcy`/`lcz` do not exist in the new one.
#: `fluo` is pyMINFLUX's fluorophore assignment, kept when present.
_FLAT_COLUMNS = (
    "vld", "sta", "tim", "tid", "gri", "sqi", "itr", "loc", "lnc",
    "eco", "ecc", "efo", "efc", "fbg", "cfr", "dcr", "fluo",
)
#: Columns that hold one value per localization and are repeated per iteration.
_PER_LOCALIZATION = frozenset({"vld", "tim", "tid", "gri", "sqi", "fluo"})
#: Columns that hold one value per iteration.
_PER_ITERATION = frozenset({"sta", "loc", "lnc", "eco", "ecc", "efo", "efc", "fbg", "cfr", "dcr"})

_LEGACY_REQUIRED = ("vld", "tim", "tid")
_LEGACY_REQUIRED_PER_ITERATION = ("loc", "efo", "cfr")


def is_legacy_minflux(dtype):
    """True when *dtype* is the legacy layout: iterations nested under `itr`."""
    names = getattr(dtype, "names", None) or ()
    if "itr" not in names:
        return False
    return dtype.fields["itr"][0].base.names is not None


def _legacy_shape(dtype):
    """``(K, iteration dtype)`` of a legacy layout, or a MinfluxLayoutError."""
    names = dtype.names or ()
    iteration = dtype.fields["itr"][0]
    iteration_names = iteration.base.names or ()
    missing = [f for f in _LEGACY_REQUIRED if f not in names]
    missing += [f"itr.{f}" for f in _LEGACY_REQUIRED_PER_ITERATION if f not in iteration_names]
    if missing:
        raise MinfluxLayoutError(f"legacy layout missing field(s): {', '.join(missing)}")
    n_iterations = int(np.prod(iteration.shape)) if iteration.shape else 1
    if n_iterations < 1:
        raise MinfluxLayoutError("legacy layout with no iterations")
    return n_iterations, iteration.base


def legacy_flat_dtype(dtype):
    """The flat dtype a legacy *dtype* is converted into."""
    _n, iteration = _legacy_shape(dtype)
    names = dtype.names
    fields = []
    for name in _FLAT_COLUMNS:
        if name == "itr":
            fields.append(("itr", "<i4"))
        elif name in _PER_LOCALIZATION and name in names:
            fields.append((name, dtype.fields[name][0]))
        elif name in _PER_ITERATION and name in iteration.names:
            fields.append((name, iteration.fields[name][0]))
    return np.dtype(fields)


def flatten_legacy_minflux(records, *, valid_only=True, chunk=100_000, check_cancel=None):
    """Convert a legacy-layout MINFLUX array to one row per iteration.

    Args:
        records: the legacy structured array.  A memory map works and is the
            point of *chunk*: only *chunk* localizations are ever copied out of
            it at once, so a multi-gigabyte export is converted without being
            loaded whole.
        valid_only: drop localizations with ``vld == False``.  pyflux discards
            them first thing, and in 3-D acquisitions they are most of the file.
        chunk: localizations converted per step.
        check_cancel: called between steps; raise from it to abort.

    Returns the flat array, rows ordered by localization, then iteration.
    """
    dtype = records.dtype
    if not is_legacy_minflux(dtype):
        raise MinfluxLayoutError("not the legacy layout: `itr` is not a nested array")
    n_iterations, _iteration = _legacy_shape(dtype)
    out_dtype = legacy_flat_dtype(dtype)
    per_localization = [n for n in out_dtype.names if n in _PER_LOCALIZATION]
    per_iteration = [n for n in out_dtype.names if n in _PER_ITERATION]
    chunk = max(1, int(chunk))
    n_records = len(records)

    def _cancel():
        if check_cancel is not None:
            check_cancel()

    # Count first, so the output is allocated once at its final size.
    if valid_only:
        n_kept = 0
        for start in range(0, n_records, chunk):
            _cancel()
            n_kept += int(np.count_nonzero(records["vld"][start:start + chunk]))
    else:
        n_kept = n_records

    out = np.empty(n_kept * n_iterations, dtype=out_dtype)
    iteration_index = np.arange(n_iterations, dtype=np.int32)
    row = 0
    for start in range(0, n_records, chunk):
        _cancel()
        block = records[start:start + chunk]
        if valid_only:
            keep = np.asarray(block["vld"], dtype=bool)
            if not keep.any():
                continue
            block = block[keep]
        else:
            block = np.asarray(block)
        m = len(block)
        if m == 0:
            continue
        rows = out[row:row + m * n_iterations]
        nested = np.asarray(block["itr"]).reshape(m, n_iterations)
        for name in per_localization:
            rows[name] = np.repeat(np.asarray(block[name]), n_iterations, axis=0)
        rows["itr"] = np.tile(iteration_index, m)
        for name in per_iteration:
            values = np.asarray(nested[name])
            rows[name] = values.reshape((m * n_iterations,) + values.shape[2:])
        row += m * n_iterations
    return out


def load_minflux_npy(path, *, chunk=100_000, check_cancel=None):
    """Load a MINFLUX .npy export in either Imspector layout.

    Returns ``(records, info)``.  *records* is always the flat layout -- one row
    per iteration -- that pyflux processes.  *info* holds ``layout`` ("flat" or
    "legacy"); for a legacy file also ``n_localizations`` and ``n_valid``, and
    ``total_tim``, the acquisition span over every localization, valid or not,
    which the converted array (valid localizations only) cannot give.

    The file is memory-mapped and copied in *chunk*-row steps, with
    *check_cancel* called between them: a plain ``np.load`` of a large file is
    one uninterruptible call.
    """
    try:
        mm = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception:
        # not memory-mappable (unusual layout, or a container) -- one shot
        mm = np.load(path, allow_pickle=False)

    if is_legacy_minflux(mm.dtype):
        n = len(mm)
        tim = mm["tim"]
        info = {
            "layout": "legacy",
            "n_localizations": int(n),
            "total_tim": float(tim[-1] - tim[0]) if n else 0.0,
        }
        flat = flatten_legacy_minflux(mm, valid_only=True, chunk=chunk, check_cancel=check_cancel)
        info["n_valid"] = int(len(flat) // max(1, _legacy_shape(mm.dtype)[0]))
        return flat, info

    info = {"layout": "flat"}
    if mm.ndim == 0 or len(mm) <= chunk:
        return np.asarray(mm).copy(), info
    out = np.empty(mm.shape, dtype=mm.dtype)
    for i in range(0, len(mm), chunk):
        if check_cancel is not None:
            check_cancel()
        out[i:i + chunk] = mm[i:i + chunk]
    return out, info
