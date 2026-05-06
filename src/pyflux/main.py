import tempfile
import os
from PySide6.QtCore import QUrl
import glob
import numpy as np
import pandas as pd
import sys
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import QUrl
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.widgets import SpanSelector
from PySide6.QtWebChannel import QWebChannel
import plotly.graph_objects as go
import plotly.io as pio
import json
from PySide6.QtGui import QColor
import hashlib
from scipy.spatial import cKDTree
import zarr
from matplotlib.colors import LinearSegmentedColormap
from functools import partial
from PySide6 import QtGui
from collections import deque

# -------------------- helpers --------------------
def save_to_csv(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

def moving_average_1d(a, n=4):
    a = np.asarray(a, dtype=float)
    if len(a) < n:
        return a
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

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
        T[:3,:3] = R
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
    points['xyz'][:, 1] *= -1.0
    return points


def bead_initial_positions(points, k=4, min_count=10):
    gri_vals = np.asarray(points["gri"])
    xyz = np.asarray(points["xyz"])
    out = {}
    for gri in np.unique(gri_vals):
        mask = (gri_vals == gri)
        if mask.sum() <= min_count:
            continue
        bead_xyz = xyz[mask]
        x = moving_average_1d(bead_xyz[:,0], n=k)
        y = moving_average_1d(bead_xyz[:,1], n=k)
        z = moving_average_1d(bead_xyz[:,2], n=k)
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

    z = ref[:,2]
    if np.std(z) > 100e-9:
        keep = (z < np.median(z) + 1.5*np.std(z))
    else:
        keep = (z < np.median(z) + 100e-9)

    common_kept = [g for g,k in zip(common, keep) if k]
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

def make_labeled_separator(text: str):
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 6, 0, 6)
    h.setSpacing(4)

    line1 = QtWidgets.QFrame()
    line1.setFrameShape(QtWidgets.QFrame.HLine)
    line1.setFrameShadow(QtWidgets.QFrame.Plain)

    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)

    line2 = QtWidgets.QFrame()
    line2.setFrameShape(QtWidgets.QFrame.HLine)
    line2.setFrameShadow(QtWidgets.QFrame.Plain)

    h.addWidget(line1, 1)
    h.addWidget(lbl, 0)
    h.addWidget(line2, 1)
    return w


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

    if "tid" not in arr.dtype.names or "loc" not in arr.dtype.names:
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
        if column in df.columns and len(df) > 0:
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


def scatter_points_and_color(arr, avg_tid: bool):
    """
    Returns xyz (N,3), vals (N,), tids_plot (N,)
    Coloring: end-to-end per tid.
    """
    if arr is None or len(arr) == 0:
        return None, None, None

    locs = arr["loc"]
    tids_all = arr["tid"]

    tids_unique = np.unique(tids_all)
    end_to_end = {}
    for tid in tids_unique:
        pts = locs[tids_all == tid]
        if len(pts) < 2:
            end_to_end[tid] = 0.0
        else:
            d = pts[-1] - pts[0]
            end_to_end[tid] = float(np.sqrt(np.sum(d * d)))

    if not avg_tid:
        xyz = locs
        tids_plot = tids_all
        vals = np.array([end_to_end[tid] for tid in tids_plot], dtype=float)
        return xyz, vals, tids_plot

    order = np.argsort(tids_all)
    a = arr[order]
    tids = a["tid"]
    locs_sorted = a["loc"]

    starts = np.r_[0, np.flatnonzero(tids[1:] != tids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(a)])

    loc_sum = np.add.reduceat(locs_sorted, starts, axis=0)
    loc_mean = loc_sum / counts[:, None]
    tids_plot = tids[starts]

    vals = np.array([end_to_end[tid] for tid in tids_plot], dtype=float)
    return loc_mean, vals, tids_plot


def tid_to_color(tid, alpha=1.0):
    """
    Deterministic pseudo-random color per tid.
    Returns CSS rgba(...) string.
    """
    # hash -> 3 bytes
    h = hashlib.md5(str(int(tid)).encode("utf-8")).digest()
    r, g, b = h[0], h[1], h[2]
    # soften a bit (avoid too-dark)
    r = int(0.25 * 255 + 0.75 * r)
    g = int(0.25 * 255 + 0.75 * g)
    b = int(0.25 * 255 + 0.75 * b)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def make_plotly_fig(arr, avg_tid: bool, is3d: bool, color_settings: dict = None, scalebar_nm = 100.0):
    """
    Build a Plotly figure for one dataset.

    color_settings dict (per file), expected keys:
      - mode: "solid" | "end-to-end" | "depth" | "tid"
      - solid: "cyan" | "green" | "magenta"              (used if mode=="solid")
      - lut: "Turbo" | "Viridis"                         (used if mode in {"end-to-end","depth"})
      - alpha: float [0..1]
      - size: int
    """
    if color_settings is None:
        color_settings = {
            "mode": DEFAULT_MODE,
            "solid": DEFAULT_SOLID,
            "lut": DEFAULT_LUT,
            "alpha": DEFAULT_ALPHA,
            "size": DEFAULT_SIZE_2D,
        }

    xyz, vals, tids_plot = scatter_points_and_color(arr, avg_tid)

    # No data
    if xyz is None or len(xyz) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)]
        )
        # minimal axis titles; theme will handle colors/background
        if is3d:
            fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        apply_plot_theme(fig, is3d=is3d)
        return fig

    # Defensive conversion
    xyz = np.asarray(xyz, dtype=float)
    vals = np.asarray(vals, dtype=float) if vals is not None else None
    tids_plot = np.asarray(tids_plot)

    # Ensure correct shape
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text=f"Bad loc shape: {xyz.shape}", x=0.5, y=0.5, showarrow=False)]
        )
        apply_plot_theme(fig, is3d=is3d)
        return fig

    x = xyz[:, 0].astype(float)
    y = xyz[:, 1].astype(float)
    z = xyz[:, 2].astype(float) if xyz.shape[1] >= 3 else np.zeros(len(x), dtype=float)

    # Sanitize - remove non-finite
    finite = np.isfinite(x) & np.isfinite(y)
    if is3d:
        finite = finite & np.isfinite(z)
    if vals is not None:
        finite = finite & np.isfinite(vals)

    if not np.all(finite):
        x = x[finite]
        y = y[finite]
        z = z[finite]
        if vals is not None:
            vals = vals[finite]
        tids_plot = tids_plot[finite]

    if len(x) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No valid data after filtering", x=0.5, y=0.5, showarrow=False)]
        )
        if is3d:
            fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        apply_plot_theme(fig, is3d=is3d)
        return fig

    trace = make_trace_for_arr(
        arr,
        avg_tid=avg_tid,
        is3d=is3d,
        color_settings=color_settings,
        name=None,
        show_colorbar=True,
    )

    if trace is None:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=is3d)
        return fig

    fig = go.Figure([trace])

    has_colorbar_2d = False

    # Keep your functional layout bits separate from theme
    if is3d:
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            )
        )
    else:
        fig.update_layout(dragmode="pan")
        fig.layout.margin.autoexpand = False

        # Track whether this 2D trace uses a colorbar.
        try:
            mk = fig.data[0].marker if len(fig.data) > 0 else None
            if mk is not None and bool(getattr(mk, "showscale", False)):
                has_colorbar_2d = True
        except Exception:
            pass

        # Equal aspect ratio if non-degenerate
        dx = float(np.nanmax(x) - np.nanmin(x)) if len(x) else 0.0
        dy = float(np.nanmax(y) - np.nanmin(y)) if len(y) else 0.0
        if dx > 0 and dy > 0:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

    apply_plot_theme(fig, is3d=is3d)
    if has_colorbar_2d:
        fig.update_layout(margin=dict(l=0, r=120, t=20, b=0))
    if not is3d:
        add_scalebar_2d(fig, length_nm=float(scalebar_nm))
    return fig


def make_plotly_fig_merged(arr_by_base: dict, avg_tid: bool, is3d: bool, color_settings_by_base: dict, scalebar_nm = 100):
    """
    Build one Plotly figure containing traces for every base in arr_by_base.

    color_settings_by_base[base] is a dict with keys:
      mode: "solid" | "end-to-end" | "depth"
      solid: "cyan"/"green"/"magenta"        (if mode=="solid")
      lut: "Turbo"/"Viridis"                (if mode in {"end-to-end","depth"})
      alpha: float [0..1]
      size: int
    """
    traces = []
    entries = []

    bases = list(arr_by_base.keys())
    if not bases:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=is3d)
        return fig

    for base in bases:
        arr = arr_by_base.get(base)
        if arr is None or len(arr) == 0:
            continue

        cs = (color_settings_by_base or {}).get(base, None)
        if cs is None:
            cs = {
                "mode": DEFAULT_MODE,
                "solid": DEFAULT_SOLID,
                "lut": DEFAULT_LUT,
                "alpha": DEFAULT_ALPHA,
                "size": DEFAULT_SIZE_2D,
            }
        else:
            # ensure missing keys get defaults
            cs = {
                "mode": cs.get("mode", DEFAULT_MODE),
                "solid": cs.get("solid", DEFAULT_SOLID),
                "lut": cs.get("lut", DEFAULT_LUT),
                "alpha": cs.get("alpha", DEFAULT_ALPHA),
                "size": cs.get("size", DEFAULT_SIZE_2D),
            }

        entries.append((base, arr, cs))

    # For merged depth mode, force one shared colorscale range across all datasets.
    global_depth_min = None
    global_depth_max = None
    for _base, arr, cs in entries:
        if cs.get("mode", DEFAULT_MODE) != MODE_DEPTH:
            continue
        xyz, _vals, _tids = scatter_points_and_color(arr, avg_tid)
        if xyz is None or len(xyz) == 0:
            continue
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            continue
        z = xyz[:, 2]
        z = z[np.isfinite(z)]
        if len(z) == 0:
            continue
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        global_depth_min = zmin if global_depth_min is None else min(global_depth_min, zmin)
        global_depth_max = zmax if global_depth_max is None else max(global_depth_max, zmax)

    colorbar_drawn = False
    for base, arr, cs in entries:
        cs_local = dict(cs)
        mode = cs_local.get("mode", DEFAULT_MODE)
        if mode == MODE_DEPTH and global_depth_min is not None and global_depth_max is not None:
            cs_local["cmin"] = global_depth_min
            cs_local["cmax"] = global_depth_max

        needs_colorbar = mode in {MODE_DEPTH, MODE_E2E}
        show_colorbar = needs_colorbar and (not colorbar_drawn)

        tr = make_trace_for_arr(
            arr,
            avg_tid=avg_tid,
            is3d=is3d,
            color_settings=cs_local,
            name=base,
            show_colorbar=show_colorbar,
        )
        if tr is not None:
            traces.append(tr)
            if show_colorbar:
                colorbar_drawn = True

    if not traces:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=is3d)
        return fig

    fig = go.Figure(traces)

    # Robust fallback: guarantee one visible colorbar for merged numeric colorscale traces.
    if not is3d:
        scale_owner = None
        for tr in fig.data:
            mk = getattr(tr, "marker", None)
            if mk is None:
                continue
            if getattr(mk, "colorscale", None) is None:
                continue
            try:
                c = np.asarray(getattr(mk, "color", []), dtype=float)
            except Exception:
                continue
            if c.size == 0 or not np.any(np.isfinite(c)):
                continue
            scale_owner = tr
            break

        if scale_owner is not None:
            for tr in fig.data:
                mk = getattr(tr, "marker", None)
                if mk is None or getattr(mk, "colorscale", None) is None:
                    continue
                mk.showscale = bool(tr is scale_owner)
                if tr is scale_owner:
                    title_text = "z"
                    try:
                        cb = getattr(mk, "colorbar", None)
                        if cb is not None and getattr(cb, "title", None) is not None:
                            t = getattr(cb.title, "text", None)
                            if isinstance(t, str) and t.strip():
                                title_text = t.strip()
                    except Exception:
                        pass
                    mk.colorbar = dict(title=title_text, x=1.0, xanchor="left", len=0.9, thickness=16)

    # Functional layout bits (legend placement, pan, aspect)
    if is3d:
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
    else:
        fig.update_layout(
            dragmode="pan",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.99,
                xanchor="left",
                x=0.0,
            ),
        )
        fig.layout.margin.autoexpand = False

        # keep equal aspect ratio only if ranges are non-degenerate
        try:
            all_x = []
            all_y = []
            for tr in fig.data:
                all_x.extend(list(tr.x) if tr.x is not None else [])
                all_y.extend(list(tr.y) if tr.y is not None else [])
            all_x = np.asarray(all_x, dtype=float)
            all_y = np.asarray(all_y, dtype=float)

            finite = np.isfinite(all_x) & np.isfinite(all_y)
            all_x = all_x[finite]
            all_y = all_y[finite]

            dx = float(np.nanmax(all_x) - np.nanmin(all_x)) if len(all_x) else 0.0
            dy = float(np.nanmax(all_y) - np.nanmin(all_y)) if len(all_y) else 0.0
            if dx > 0 and dy > 0:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
        except Exception:
            pass

    apply_plot_theme(fig, is3d=is3d)
    if colorbar_drawn:
        # Reserve space so a merged-mode colorbar is not clipped.
        fig.update_layout(margin=dict(l=0, r=120, t=20, b=0))
    if not is3d:
        add_scalebar_2d(fig, length_nm=float(scalebar_nm))
    return fig

def pointcloud_to_image(x_nm, y_nm, pixel_size_nm=4.0, padding_nm=0.0):
    """
    Bin a 2D point cloud (x_nm, y_nm) into an image where each pixel is
    pixel_size_nm wide/tall. Pixel intensity = number of points in that pixel.

    Returns
    -------
    H : (ny, nx) ndarray
        2D histogram (counts per pixel). Row 0 corresponds to lowest y-bin.
    extent : (xmin, xmax, ymin, ymax)
        Extent in nm for plotting.
    """
    x_nm = np.asarray(x_nm).ravel()
    y_nm = np.asarray(y_nm).ravel()
    if x_nm.size == 0:
        return np.zeros((1, 1), dtype=float), (0.0, 1.0, 0.0, 1.0)
    if x_nm.size != y_nm.size:
        raise ValueError("x_nm and y_nm must have the same length")

    xmin, xmax = float(np.min(x_nm)), float(np.max(x_nm))
    ymin, ymax = float(np.min(y_nm)), float(np.max(y_nm))

    xmin -= padding_nm; xmax += padding_nm
    ymin -= padding_nm; ymax += padding_nm

    # avoid degenerate ranges
    if xmax == xmin:
        xmax = xmin + pixel_size_nm
    if ymax == ymin:
        ymax = ymin + pixel_size_nm

    nx = int(np.ceil((xmax - xmin) / pixel_size_nm))
    ny = int(np.ceil((ymax - ymin) / pixel_size_nm))
    nx = max(nx, 1)
    ny = max(ny, 1)

    x_edges = xmin + np.arange(nx + 1) * pixel_size_nm
    y_edges = ymin + np.arange(ny + 1) * pixel_size_nm

    # histogram2d: first arg = y, second = x so H[ybin, xbin]
    H, _, _ = np.histogram2d(y_nm, x_nm, bins=(y_edges, x_edges))
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    return H, extent

def make_plotly_heatmap_from_arr(
    arr,
    pixel_size_nm: float,
    lut: str,
    title: str = None,
    *,
    scale_mode: str = "linear",   # "linear" (default) or "log"
    max_value: float = 10.0,
    show_colorbar: bool = True,
):
    """
    Build a single Plotly heatmap for the array's loc_x/loc_y (assumes nm units already).

    Parameters
    ----------
    arr : structured ndarray with field "loc" shaped (N, >=2)
    pixel_size_nm : float
        Binning pixel size in nm (global setting).
    lut : str
        Plotly colorscale name (must be in LUT_CHOICES).
    title : str
        Figure title.
    scale_mode : str
        "linear" or "log" (log uses log10(count+1)).
    """

    if arr is None or len(arr) == 0:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=False)
        return fig

    loc = np.asarray(arr["loc"], dtype=float)
    if loc.ndim != 2 or loc.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="Bad loc shape", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=False)
        return fig

    x = loc[:, 0]
    y = loc[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) == 0:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No valid points", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=False)
        return fig

    pixel_size_nm = float(pixel_size_nm)
    pixel_size_nm = max(pixel_size_nm, 0.1)

    H, extent = pointcloud_to_image(x, y, pixel_size_nm=pixel_size_nm, padding_nm=0.0)
    xmin, xmax, ymin, ymax = extent
    ny, nx = H.shape

    # pixel center coordinates
    x_centers = xmin + (np.arange(nx) + 0.5) * pixel_size_nm
    y_centers = ymin + (np.arange(ny) + 0.5) * pixel_size_nm

    if lut not in LUT_CHOICES:
        lut = DEFAULT_LUT_BIN
    if lut.startswith("cu"):
        lut = CUSTOM_LUTS.get(lut, DEFAULT_LUT_BIN)

    scale_mode = (scale_mode or "linear").strip().lower()
    if scale_mode not in ("linear", "log10(count+1)"):
        scale_mode = "linear"

    if scale_mode == "log10(count+1)":
        Z = np.log10(H + 1.0)
    else:
        Z = H.astype(float, copy=False)

    max_value = max(float(max_value), 1e-9)
    Z = _clip_to_scale_max(Z, max_value=max_value)

    fig = go.Figure(
        data=go.Heatmap(
            z=Z.tolist(),                 # JSON-safe
            x=x_centers.tolist(),
            y=y_centers.tolist(),
            colorscale=lut,
            colorbar=dict(title="value"),
            showscale=bool(show_colorbar),
            zmin=0.0,
            zmax=max_value,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x (nm)",
        yaxis_title="y (nm)",
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    apply_plot_theme(fig, is3d=False)
    fig.update_xaxes(showline=False, ticks="", showticklabels=False)
    fig.update_yaxes(showline=False, ticks="", showticklabels=False)

    return fig

def _xy_from_arr(arr):
    if arr is None or len(arr) == 0:
        return np.array([]), np.array([])
    loc = np.asarray(arr["loc"], dtype=float)
    if loc.ndim != 2 or loc.shape[1] < 2:
        return np.array([]), np.array([])
    x = loc[:, 0]
    y = loc[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _clip_hist_for_overlay(H, scale_mode: str, max_value: float = 10.0):
    scale_mode = (scale_mode or "linear").strip().lower()
    if scale_mode == "log10(count+1)":
        Z = np.log10(H + 1.0)
    else:
        Z = H.astype(float, copy=False)
    return _clip_to_scale_max(Z, max_value=max_value)

def _clip_to_scale_max(img: np.ndarray, max_value: float = 10.0):
    if img is None or np.size(img) == 0:
        return np.zeros((1, 1), dtype=float)
    max_value = float(max(max_value, 1e-9))
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(img, dtype=float)
    return np.clip(np.asarray(img, dtype=float), 0.0, max_value)

def _resolve_overlay_colorscale(lut: str):
    if lut not in LUT_CHOICES:
        lut = DEFAULT_LUT_BIN
    if lut.startswith("cu"):
        return CUSTOM_LUTS.get(lut, DEFAULT_LUT_BIN)
    return lut

def _plotly_color_to_mpl(color):
    if not isinstance(color, str):
        return color

    s = color.strip()
    low = s.lower()

    if low.startswith("rgb(") and low.endswith(")"):
        body = s[s.find("(") + 1:-1]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) >= 3:
            try:
                r = float(parts[0]) / 255.0
                g = float(parts[1]) / 255.0
                b = float(parts[2]) / 255.0
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                return (r, g, b)
            except Exception:
                return s

    if low.startswith("rgba(") and low.endswith(")"):
        body = s[s.find("(") + 1:-1]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) >= 4:
            try:
                r = float(parts[0]) / 255.0
                g = float(parts[1]) / 255.0
                b = float(parts[2]) / 255.0
                a = float(parts[3])
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                a = max(0.0, min(1.0, a))
                return (r, g, b, a)
            except Exception:
                return s

    return s

def _resolve_mpl_cmap(lut: str):
    if lut not in LUT_CHOICES:
        lut = DEFAULT_LUT_BIN
    if lut.startswith("cu"):
        colors = CUSTOM_LUTS.get(lut)
        if colors:
            mpl_colors = []
            for c in colors:
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    mpl_colors.append((float(c[0]), _plotly_color_to_mpl(c[1])))
                else:
                    mpl_colors.append(_plotly_color_to_mpl(c))
            return LinearSegmentedColormap.from_list(f"overlay_{lut}", mpl_colors)
        return matplotlib.colormaps.get_cmap(DEFAULT_LUT_BIN)
    try:
        return matplotlib.colormaps.get_cmap(lut)
    except Exception:
        return matplotlib.colormaps.get_cmap(DEFAULT_LUT_BIN)

def _make_rgba_intensity_colorscale(lut: str, steps: int = 256):
    cmap = _resolve_mpl_cmap(lut)
    out = []
    steps = max(2, int(steps))
    for i in range(steps):
        t = i / (steps - 1)
        r, g, b, _ = cmap(t)
        rr = int(round(255 * r))
        gg = int(round(255 * g))
        bb = int(round(255 * b))
        out.append([t, f"rgba({rr},{gg},{bb},1.0)"])
    return out

def _lut_rgb_image(lut: str, z01: np.ndarray):
    cmap = _resolve_mpl_cmap(lut)
    z = np.clip(np.asarray(z01, dtype=float), 0.0, 1.0)
    rgba = cmap(z)
    return np.asarray(rgba[..., :3], dtype=float)

def _compute_bounds_xy(x: np.ndarray, y: np.ndarray, sigma_nm: float, n_sigma: float = 3.0):
    if len(x) == 0:
        return (0.0, 1.0, 0.0, 1.0)
    pad = float(n_sigma) * float(sigma_nm)
    xmin = float(np.min(x) - pad)
    xmax = float(np.max(x) + pad)
    ymin = float(np.min(y) - pad)
    ymax = float(np.max(y) + pad)
    if xmax <= xmin:
        xmax = xmin + max(float(sigma_nm), 1.0)
    if ymax <= ymin:
        ymax = ymin + max(float(sigma_nm), 1.0)
    return xmin, xmax, ymin, ymax

def render_gaussians_xy(x, y, sigma_nm: float, pixel_size_nm: float, n_sigma: float = 3.0, bounds=None):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if len(x) == 0:
        return np.zeros((1, 1), dtype=float), (0.0, 1.0, 0.0, 1.0)

    sigma_nm = max(float(sigma_nm), 1e-9)
    pixel_size_nm = max(float(pixel_size_nm), 0.1)

    if bounds is None:
        xmin, xmax, ymin, ymax = _compute_bounds_xy(x, y, sigma_nm=sigma_nm, n_sigma=n_sigma)
    else:
        xmin, xmax, ymin, ymax = bounds

    nx = max(int(np.ceil((xmax - xmin) / pixel_size_nm)), 1)
    ny = max(int(np.ceil((ymax - ymin) / pixel_size_nm)), 1)

    xs = xmin + (np.arange(nx) + 0.5) * pixel_size_nm
    ys = ymin + (np.arange(ny) + 0.5) * pixel_size_nm

    img = np.zeros((ny, nx), dtype=float)
    r = float(n_sigma) * sigma_nm
    denom = 2.0 * sigma_nm * sigma_nm

    for xi, yi in zip(x, y):
        ix = np.where((xs >= xi - r) & (xs <= xi + r))[0]
        iy = np.where((ys >= yi - r) & (ys <= yi + r))[0]
        if ix.size == 0 or iy.size == 0:
            continue
        X, Y = np.meshgrid(xs[ix], ys[iy])
        img[np.ix_(iy, ix)] += np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / denom)

    return img, (float(xmin), float(xmax), float(ymin), float(ymax))

def make_plotly_gaussian_from_arr(
    arr,
    pixel_size_nm: float,
    lut: str,
    title: str = None,
    *,
    max_value: float = 10.0,
    show_colorbar: bool = True,
):
    x, y = _xy_from_arr(arr)
    if len(x) == 0:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No valid points", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=False)
        return fig

    sigma_nm = max(float(pixel_size_nm), 0.1)
    img, extent = render_gaussians_xy(
        x,
        y,
        sigma_nm=sigma_nm,
        pixel_size_nm=float(pixel_size_nm),
        n_sigma=3.0,
        bounds=None,
    )
    max_value = max(float(max_value), 1e-9)
    z = _clip_to_scale_max(img, max_value=max_value)

    xmin, xmax, ymin, ymax = extent
    ny, nx = z.shape
    x_centers = xmin + (np.arange(nx) + 0.5) * float(pixel_size_nm)
    y_centers = ymin + (np.arange(ny) + 0.5) * float(pixel_size_nm)

    cs = _resolve_overlay_colorscale(lut)

    fig = go.Figure(
        data=go.Heatmap(
            z=z.tolist(),
            x=x_centers.tolist(),
            y=y_centers.tolist(),
            colorscale=cs,
            zmin=0.0,
            zmax=max_value,
            colorbar=dict(title="value"),
            showscale=bool(show_colorbar),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x (nm)",
        yaxis_title="y (nm)",
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    apply_plot_theme(fig, is3d=False)
    fig.update_xaxes(showline=False, ticks="", showticklabels=False)
    fig.update_yaxes(showline=False, ticks="", showticklabels=False)
    return fig

def make_plotly_overlay_heatmap_from_two_arrs(
    arr_a,
    arr_b,
    pixel_size_nm: float,
    lut_a: str,
    lut_b: str,
    *,
    title: str = None,
    scale_mode: str = "linear",
    render_mode: str = "heatmap",
    max_value_a: float = 10.0,
    max_value_b: float = 10.0,
):
    x_a, y_a = _xy_from_arr(arr_a)
    x_b, y_b = _xy_from_arr(arr_b)

    if len(x_a) == 0 and len(x_b) == 0:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No valid points", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=False)
        return fig

    render_mode = (render_mode or "heatmap").strip().lower()
    pixel_size_nm = max(float(pixel_size_nm), 0.1)
    max_value_a = max(float(max_value_a), 1e-9)
    max_value_b = max(float(max_value_b), 1e-9)

    x_all = np.concatenate([x_a, x_b]) if (len(x_a) and len(x_b)) else (x_a if len(x_a) else x_b)
    y_all = np.concatenate([y_a, y_b]) if (len(y_a) and len(y_b)) else (y_a if len(y_a) else y_b)

    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))

    if render_mode == "gaussian":
        pad = 3.0 * pixel_size_nm
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad

    if xmax == xmin:
        xmax = xmin + pixel_size_nm
    if ymax == ymin:
        ymax = ymin + pixel_size_nm

    nx = max(int(np.ceil((xmax - xmin) / pixel_size_nm)), 1)
    ny = max(int(np.ceil((ymax - ymin) / pixel_size_nm)), 1)

    x_edges = xmin + np.arange(nx + 1) * pixel_size_nm
    y_edges = ymin + np.arange(ny + 1) * pixel_size_nm

    if render_mode == "gaussian":
        sigma_nm = pixel_size_nm
        bounds = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
        img_a, _ = render_gaussians_xy(
            x_a,
            y_a,
            sigma_nm=sigma_nm,
            pixel_size_nm=pixel_size_nm,
            n_sigma=3.0,
            bounds=bounds,
        )
        img_b, _ = render_gaussians_xy(
            x_b,
            y_b,
            sigma_nm=sigma_nm,
            pixel_size_nm=pixel_size_nm,
            n_sigma=3.0,
            bounds=bounds,
        )
        clip_a = _clip_to_scale_max(img_a, max_value=max_value_a)
        clip_b = _clip_to_scale_max(img_b, max_value=max_value_b)
        z_a = np.asarray(clip_a / (max_value_a + 1e-12), dtype=float)
        z_b = np.asarray(clip_b / (max_value_b + 1e-12), dtype=float)
    else:
        H_a = np.zeros((ny, nx), dtype=float)
        H_b = np.zeros((ny, nx), dtype=float)

        if len(x_a):
            H_a, _, _ = np.histogram2d(y_a, x_a, bins=(y_edges, x_edges))
        if len(x_b):
            H_b, _, _ = np.histogram2d(y_b, x_b, bins=(y_edges, x_edges))

        clip_a = _clip_hist_for_overlay(H_a, scale_mode, max_value=max_value_a)
        clip_b = _clip_hist_for_overlay(H_b, scale_mode, max_value=max_value_b)

        z_a = np.asarray(clip_a / (max_value_a + 1e-12), dtype=float)
        z_b = np.asarray(clip_b / (max_value_b + 1e-12), dtype=float)

    rgb_a = _lut_rgb_image(lut_a, z_a)
    rgb_b = _lut_rgb_image(lut_b, z_b)
    # Screen blend keeps per-channel LUT character and avoids neutral grey mixing.
    rgb = 1.0 - (1.0 - rgb_a) * (1.0 - rgb_b)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = np.asarray(np.round(rgb * 255.0), dtype=np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(
        z=rgb_u8.tolist(),
        x0=float(x_edges[0]),
        y0=float(y_edges[0]),
        dx=float(pixel_size_nm),
        dy=float(pixel_size_nm),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x (nm)",
        yaxis_title="y (nm)",
        dragmode="pan",
    )
    fig.update_xaxes(
        showline=False,
        ticks="",
        showticklabels=False,
        range=[float(x_edges[0]), float(x_edges[-1])],
    )
    fig.update_yaxes(
        showline=False,
        ticks="",
        showticklabels=False,
        range=[float(y_edges[0]), float(y_edges[-1])],
        scaleanchor="x",
        scaleratio=1,
    )
    apply_plot_theme(fig, is3d=False)
    fig.update_layout(plot_bgcolor="#000000")
    return fig

def make_trace_for_arr(arr, avg_tid: bool, is3d: bool, color_settings: dict, name: str = None, show_colorbar: bool = True):
    xyz, vals, tids_plot = scatter_points_and_color(arr, avg_tid)
    if xyz is None or len(xyz) == 0:
        return None

    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        return None

    x = xyz[:, 0].astype(float)
    y = xyz[:, 1].astype(float)
    z = xyz[:, 2].astype(float) if xyz.shape[1] >= 3 else np.zeros(len(x), dtype=float)

    # --- finite filtering ---
    if is3d:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    else:
        finite = np.isfinite(x) & np.isfinite(y)

    if vals is not None:
        finite = finite & np.isfinite(np.asarray(vals, dtype=float))

    if not np.all(finite):
        x, y, z = x[finite], y[finite], z[finite]
        if vals is not None:
            vals = np.asarray(vals, dtype=float)[finite]
        tids_plot = np.asarray(tids_plot)[finite]

    if len(x) == 0:
        return None

    cs = color_settings or {}
    mode = cs.get("mode", DEFAULT_MODE)

    alpha = float(cs.get("alpha", DEFAULT_ALPHA))
    alpha = max(0.0, min(1.0, alpha))

    size = int(cs.get("size", DEFAULT_SIZE_2D))
    size = max(1, size)

    # hover text
    text = [f"{name + ' ' if name else ''}tid={int(t)}" for t in tids_plot]

    # --- marker selection ---
    if mode == "solid":
        solid = cs.get("solid", DEFAULT_SOLID)
        col = SOLID_COLOR_MAP.get(solid, SOLID_COLOR_MAP[DEFAULT_SOLID])
        marker = dict(size=size, opacity=alpha, color=col)

    elif mode == "depth":
        lut = cs.get("lut", DEFAULT_LUT)
        if lut not in LUT_CHOICES:
            lut = DEFAULT_LUT
        if lut.startswith("cu"):
            lut = CUSTOM_LUTS.get(lut, DEFAULT_LUT)
        marker = dict(
            size=size,
            opacity=alpha,
            color=z.tolist(),
            colorscale=lut,
            showscale=bool(show_colorbar),
        )
        cmin = cs.get("cmin", None)
        cmax = cs.get("cmax", None)
        if cmin is not None and cmax is not None:
            marker["cmin"] = float(cmin)
            marker["cmax"] = float(cmax)
        if show_colorbar:
            marker["colorbar"] = dict(title="z", x=1.0, xanchor="left", len=0.9, thickness=16)

    elif mode == "tid":
        # one color per point based on its tid (deterministic random)
        # note: Plotly accepts list of color strings for marker.color
        color_list = [tid_to_color(t, alpha=alpha) for t in tids_plot]

        marker = dict(
            size=size,
            opacity=1.0,     # alpha already baked into rgba(); keep opacity=1 to avoid double-multiplying
            color=color_list,
        )

    else:  # "end-to-end"
        lut = cs.get("lut", DEFAULT_LUT)
        if lut not in LUT_CHOICES:
            lut = DEFAULT_LUT
        if lut.startswith("cu"):
            lut = CUSTOM_LUTS.get(lut, DEFAULT_LUT)

        cL = np.asarray(vals, dtype=float).tolist()
        marker = dict(
            size=size,
            opacity=alpha,
            color=cL,
            colorscale=lut,
            showscale=bool(show_colorbar),
        )
        if show_colorbar:
            marker["colorbar"] = dict(title="end-to-end", x=1.0, xanchor="left", len=0.9, thickness=16)

    # --- traces ---
    if is3d:
        return go.Scatter3d(
            x=x.tolist(), y=y.tolist(), z=z.tolist(),
            mode="markers",
            name=name,
            marker=marker,
            text=text,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>",
        )
    else:
        # When coloring by depth, include z in hover even in 2D
        if mode == "depth":
            customdata = z.astype(float).tolist()   # 1D list
            hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{customdata:.3f}<br>%{text}<extra></extra>"
        else:
            customdata = None
            hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}<extra></extra>"


        return go.Scattergl(
            x=x.tolist(), y=y.tolist(),
            mode="markers",
            name=name,
            marker=marker,
            text=text,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )

def apply_plot_theme(fig: go.Figure, *, is3d: bool):
    th = PLOT_THEME

    fig.update_layout(
        template=th.get("template", "none"),
        paper_bgcolor=th.get("paper_bg", "white"),
        font=dict(
            family=th.get("font_family", "Arial"),
            size=th.get("font_size", 12),
            color=th.get("font_color", "black"),
        ),
        margin=th.get("margin", dict(l=0, r=0, t=20, b=0)),
        legend=dict(
            bgcolor=th.get("legend_bg", "rgba(0,0,0,0)"),
            bordercolor=th.get("legend_border", "rgba(0,0,0,0)"),
            borderwidth=0 if th.get("legend_border", None) in (None, "rgba(0,0,0,0)") else 1,
        ),
    )

    if is3d:
        title_col = th.get("scene_axis_title_color", th.get("font_color", "black"))
        tick_col = th.get("scene_tick_color", "#888888")

        scene_axis_common = dict(
            showbackground=False,
            gridcolor=th.get("scene_grid_color", "#DDDDDD"),
            zerolinecolor=th.get("scene_zeroline_color", "#BBBBBB"),
            linecolor=th.get("scene_axis_line_color", "#888888"),
            tickcolor=tick_col,
            tickfont=dict(color=tick_col),
            title=dict(font=dict(color=title_col)),
        )

        fig.update_layout(
            scene=dict(
                bgcolor=th.get("scene_bg", th.get("paper_bg", "white")),
                xaxis=scene_axis_common,
                yaxis=scene_axis_common,
                zaxis=scene_axis_common,
            )
        )

    else:
        fig.update_layout(plot_bgcolor=th.get("plot_bg", th.get("paper_bg", "white")))

        fig.update_xaxes(
            gridcolor=th.get("grid_color", "#DDDDDD"),
            zerolinecolor=th.get("zeroline_color", "#BBBBBB"),
            linecolor=th.get("axis_line_color", "#888888"),
            tickcolor=th.get("tick_color", "#888888"),
            tickfont=dict(color=th.get("tick_color", "#888888")),
            title_font=dict(color=th.get("axis_title_color", th.get("font_color", "black"))),
        )
        fig.update_yaxes(
            gridcolor=th.get("grid_color", "#DDDDDD"),
            zerolinecolor=th.get("zeroline_color", "#BBBBBB"),
            linecolor=th.get("axis_line_color", "#888888"),
            tickcolor=th.get("tick_color", "#888888"),
            tickfont=dict(color=th.get("tick_color", "#888888")),
            title_font=dict(color=th.get("axis_title_color", th.get("font_color", "black"))),
        )

    return fig

def add_scalebar_2d(fig: go.Figure, length_nm: float = 100.0):
    """
    Adds a data-anchored scale bar of `length_nm` (same units as x/y) to the
    lower-left of the current data range.
    Works for 2D figures with numeric x/y.
    """
    if fig is None or not fig.data:
        return fig

    # collect all x/y from traces
    xs, ys = [], []
    for tr in fig.data:
        if hasattr(tr, "x") and tr.x is not None:
            xs.extend(tr.x)
        if hasattr(tr, "y") and tr.y is not None:
            ys.extend(tr.y)

    if not xs or not ys:
        return fig

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(xs) == 0:
        return fig

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        return fig

    # position: a bit inset from lower-left of data extent
    x0 = xmin + SCALEBAR_MARGIN_FRACTION * dx
    y0 = ymin + SCALEBAR_MARGIN_FRACTION * dy
    x1 = x0 + float(length_nm)

    # remove previous scalebar (if any)
    def _obj_name(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("name")
        return getattr(obj, "name", None)

    fig.layout.shapes = tuple(s for s in (fig.layout.shapes or []) if _obj_name(s) != "scalebar")
    fig.layout.annotations = tuple(a for a in (fig.layout.annotations or []) if _obj_name(a) != "scalebar_label")

    # add line
    fig.add_shape(
        type="line",
        xref="x", yref="y",
        x0=x0, y0=y0,
        x1=x1, y1=y0,
        line=dict(color=SCALEBAR_COLOR, width=SCALEBAR_LINE_WIDTH),
        name="scalebar",
    )

    # add label centered above line
    fig.add_annotation(
        x=(x0 + x1) / 2.0,
        y=y0 + 0.02 * dy,
        xref="x", yref="y",
        text=f"{length_nm:.0f} nm",
        showarrow=False,
        font=dict(color=SCALEBAR_COLOR, size=12),
        name="scalebar_label",
    )

    return fig

def apply_hist_theme(fig: Figure, ax):
    th = HIST_THEME

    fig.patch.set_facecolor(th["fig_bg"])
    ax.set_facecolor(th["ax_bg"])

    ax.set_axisbelow(True)  # grid behind bars/patches

    ax.title.set_color(th["text"])
    ax.xaxis.label.set_color(th["text"])
    ax.yaxis.label.set_color(th["text"])

    ax.tick_params(axis="both", colors=th["ticks"])

    for spine in ax.spines.values():
        spine.set_color(th["spines"])

    ax.grid(True, color=th["grid"], alpha=0.6, linewidth=0.8)

def apply_gui_theme(app: QtWidgets.QApplication):
    t = GUI_THEME
    qss = f"""
    /* Base */
    QWidget {{
        background-color: {t["bg"]};
        color: {t["text"]};
        selection-background-color: {t["selection_bg"]};
        selection-color: {t["selection_text"]};
    }}

    QMainWindow::separator {{
        background: {t["border"]};
        width: 1px;
        height: 1px;
    }}

    QFrame[role="separator"] {{
    color: #3C4043;
    background-color: #3C4043;
    min-height: 1px;
    max-height: 1px;
    }}

    /* Group boxes */
    QGroupBox {{
        background-color: {t["panel_bg"]};
        border: 1px solid {t["border"]};
        border-radius: 6px;
        margin-top: 10px;
        padding: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        color: {t["text"]};
    }}

    /* Matplotlib toolbar */
    QToolBar {{
        background-color: #25272A;   /* or GUI_THEME["panel_bg"] */
        border: 1px solid #3C4043;   /* or GUI_THEME["border"] */
        spacing: 4px;
        padding: 2px;
    }}

    QToolButton {{
        background-color: #303134;   /* button bg */
        border: 1px solid #5F6368;
        border-radius: 4px;
        padding: 2px;
        margin: 1px;
    }}

    QToolButton:hover {{
        background-color: #3A3B3C;
    }}

    QToolButton:pressed {{
        background-color: #2B2C2D;
    }}

    QToolBar QToolButton {{
        background-color: #919191;      /* your custom color */
        border: 1px solid #6e6e6e;
    }}

    QToolBar QToolButton:hover {{
        background-color: #24466c;
    }}   

    /* Inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {t["input_bg"]};
        color: {t["input_text"]};
        border: 1px solid {t["input_border"]};
        border-radius: 5px;
        padding: 3px 6px;
    }}

    QComboBox {{
        background-color: {t["input_bg"]};
        color: {t["input_text"]};
        border: 1px solid {t["input_border"]};
        border-radius: 5px;
        padding: 3px 18px 3px 8px;   /* right padding leaves room for arrow */
        min-height: 24px;            /* prevents cropping */
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 16px;                 /* was 18; 24–30 is usually safe */
        border-left: 1px solid {t["input_border"]};
    }}

    /* style the popup list */
    QComboBox QAbstractItemView {{
        background-color: {t["panel_bg"]};
        color: {t["text"]};
        selection-background-color: {t["selection_bg"]};
        selection-color: {t["selection_text"]};
        outline: 0;
        padding: 4px;
    }}

    /* Spinbox buttons (up/down) */
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: #303134;              /* choose */
        border-left: 1px solid #4A4D50;
        width: 16px;
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: #3A3B3C;
    }}

    /* Arrow color */
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 8px;
        height: 8px;
        /* this controls the arrow glyph color */
        color: #ffffff;
    }}

    /* Optional: pressed */
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
        background-color: #2B2C2D;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {t["button_bg"]};
        color: {t["button_text"]};
        border: 1px solid {t["button_border"]};
        border-radius: 6px;
        padding: 5px 10px;
    }}
    QPushButton:hover {{
        background-color: {t["button_bg_hover"]};
    }}
    QPushButton:pressed {{
        background-color: {t["button_bg_pressed"]};
    }}
    QPushButton:disabled {{
        color: {t["muted_text"]};
        border-color: {t["border"]};
        background-color: {t["panel_bg"]};
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: 1px solid {t["border"]};
        background-color: {t["panel_bg"]};
        border-radius: 6px;
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {t["button_bg"]};
        color: {t["button_text"]};
        border: 1px solid {t["button_border"]};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 5px 12px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: #1f2123;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {t["button_bg_hover"]};
    }}

    /* Tables */
    QTableWidget {{
        background-color: {t["panel_bg"]};
        gridline-color: {t["border"]};
        border: 1px solid {t["border"]};
    }}
    QHeaderView::section {{
        background-color: {t["panel_bg"]};
        color: {t["text"]};
        border: 1px solid {t["border"]};
        padding: 4px 6px;
    }}

    /* Scroll areas */
    QScrollArea {{
        border: none;
        background-color: {t["panel_bg"]};
    }}

    """
    app.setStyleSheet(qss)

def custom_LUT(colors=["#000000", "#ff0000"], bins=2**8):
    """
    colors: list of colors that define the color scale
    bins: range
    """

    cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=bins)
    xs = np.linspace(0, 1, bins)
    rgba = cm(xs)  # (n,4) floats in [0,1]
    colorscale = []
    for x, (r, g, b, a) in zip(xs, rgba):
        rr = int(round(r * 255))
        gg = int(round(g * 255))
        bb = int(round(b * 255))
        colorscale.append([float(x), f"#{rr:02x}{gg:02x}{bb:02x}"])
    return colorscale

# -------------------- Qt (GUI) theme --------------------

GUI_THEME = {
    "bg": "#202122",          # main window background
    "panel_bg": "#202122",    # groupboxes / panels
    "text": "#E8EAED",
    "muted_text": "#B0B3B8",
    "border": "#3C4043",

    "input_bg": "#1F1F1F",
    "input_text": "#E8EAED",
    "input_border": "#4A4D50",

    "button_bg": "#303134",
    "button_text": "#E8EAED",
    "button_border": "#5F6368",
    "button_bg_hover": "#3A3B3C",
    "button_bg_pressed": "#2B2C2D",

    "selection_bg": "#4C8BF5",
    "selection_text": "#FFFFFF",
}

# -------------------- constants --------------------

DEFAULT_ALPHA = 0.85
DEFAULT_SIZE_2D = 5
DEFAULT_SIZE_3D = 5

MODE_SOLID = "solid"
MODE_DEPTH = "depth"
MODE_TID = "tid"
MODE_E2E = "end-to-end"

MODE_CHOICES = [MODE_SOLID, MODE_DEPTH, MODE_TID, MODE_E2E]

SOLID_COLOR_MAP = {
    "cyan": "#00FFFF",
    "blue": "#0000FF",
    "red": "#FF0000",
    "yellow": "#FFFF00", 
    "orange": "#FF8800",
    "darkgreen": "#008800",
    "green": "#00CC00",
    "magenta": "#FF00FF",
}
SOLID_COLOR_CHOICES = list(SOLID_COLOR_MAP.keys())

CUSTOM_LUTS = {
    f"cu_{i}": custom_LUT(colors=["#000000", SOLID_COLOR_MAP[i]], bins=256)
    for i in SOLID_COLOR_CHOICES
}

LUT_CHOICES = ["turbo", "viridis", "cividis", "inferno", "magma", "plasma", 
               "electric", "hot", "hsv", "jet", "rainbow", "twilight", "icefire", 
               "piyg", "brbg", "rdbu", "brwnyl", "reds", "rdpu", "ylgn", "ylorbr", 
               "thermal","gray", "ice", "algae", "speed", "temps", 
               ]  # Plotly colorscale names
LUT_CHOICES = sorted(LUT_CHOICES + list(CUSTOM_LUTS.keys()))

DEFAULT_SOLID = "darkgreen"
DEFAULT_LUT = "twilight"
DEFAULT_LUT_BIN = "hot"
DEFAULT_MODE = "depth"

# -------------------- Plot theme --------------------

PLOT_THEME = {
    # overall
    "template": "none",                 # "none" to rely on your explicit colors; or "plotly_dark"/"plotly_white"
    "font_family": "Arial",
    "font_size": 12,
    "font_color": "#E6E6E6",

    # backgrounds
    "paper_bg": "#202122",              # outside plotting area
    "plot_bg":  "#202122",              # 2D plotting area
    "scene_bg": "#202122",              # 3D scene background

    # axes/grid (2D)
    "axis_line_color": "#8A8A8A",
    "grid_color": "#2F2F2F",
    "zeroline_color": "#444444",
    "tick_color": "#CFCFCF",
    "axis_title_color": "#E6E6E6",

    # axes/grid (3D)
    "scene_grid_color": "#2F2F2F",
    "scene_zeroline_color": "#444444",
    "scene_axis_line_color": "#8A8A8A",
    "scene_tick_color": "#CFCFCF",
    "scene_axis_title_color": "#E6E6E6",

    # legend
    "legend_bg": "rgba(0,0,0,0)",       # transparent
    "legend_border": "#666666",

    # margins
    "margin": dict(l=0, r=0, t=20, b=0),
}

DEFAULT_SCALEBAR_LENGTH_NM = 100.0
SCALEBAR_LENGTH_NM = DEFAULT_SCALEBAR_LENGTH_NM  
SCALEBAR_MARGIN_FRACTION = 0.05   # distance from lower-left as fraction of current view range
SCALEBAR_LINE_WIDTH = 4
SCALEBAR_COLOR = "#E6E6E6"


PLOTLY_HTML_BG = PLOT_THEME["paper_bg"]   # or PLOT_THEME["plot_bg"]

# -------------------- Matplotlib theme (histogram) --------------------

HIST_THEME = {
    "fig_bg":   "#202122",   # matches PLOT_THEME["paper_bg"]
    "ax_bg":    "#202122",   # matches PLOT_THEME["plot_bg"]
    "text":     "#E6E6E6",
    "grid":     "#2F2F2F",
    "spines":   "#8A8A8A",
    "ticks":    "#CFCFCF",

    # histogram style
    "hist_face": "#b8b8ff",
    "hist_edge": "#b8b8ff",

    # selection span style (SpanSelector)
    "span_face": "#f8f7ff",
    "span_alpha": 0.15,
    "span_edge":  "black",
    "span_lw":    2,
}

# -------------------- GUI components --------------------

class ColorSettingsPanel(QtWidgets.QGroupBox):
    """
    Table-based color settings:
      Columns: File | Mode | LUT/Solid | α | Size
      One row per file base.

    Emits changed(base, settings_dict).
    settings_dict keys:
      - mode: str  ("solid"|"depth"|"tid"|"end-to-end")
      - solid: str
      - lut: str
      - alpha: float
      - size: int
    """
    changed = Signal(str, object)  # (base, settings dict)

    COL_FILE  = 0
    COL_MODE  = 1
    COL_PALETTE = 2
    COL_ALPHA = 3
    COL_SIZE  = 4
    COL_MBM_SOURCE = 5

    def __init__(self, title="Color settings", parent=None):
        super().__init__(title, parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.table = QtWidgets.QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["File", "Mode", "LUT / Solid", "α", "Size", "mbm source"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        self.table.setAlternatingRowColors(False)

        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(False)

        hdr.setSectionResizeMode(self.COL_FILE, QtWidgets.QHeaderView.Interactive)
        self.table.setColumnWidth(self.COL_FILE, 260)

        hdr.setSectionResizeMode(self.COL_MODE, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_PALETTE, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_ALPHA, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_SIZE, QtWidgets.QHeaderView.Stretch)

        # make it scroll when many rows
        self.table.setMinimumHeight(170)
        self.table.setMaximumHeight(260)

        outer.addWidget(self.table, 1)

        # base -> dict of widgets
        self._row_by_base = {}  # base -> row index
        self._widgets_by_base = {}  # base -> dict(mode=..., palette=..., alpha=..., size=...)

    def _emit_row(self, base: str):
        w = self._widgets_by_base.get(base)
        if not w:
            return

        mode = w["mode"].currentText()
        alpha = float(w["alpha"].value())
        size = int(w["size"].value())

        # palette meaning depends on mode
        if mode == "solid":
            payload = {
                "mode": "solid",
                "solid": w["palette"].currentText(),
                "lut": DEFAULT_LUT,
                "alpha": alpha,
                "size": size,
            }
        elif mode == "tid":
            payload = {
                "mode": "tid",
                "solid": DEFAULT_SOLID,
                "lut": DEFAULT_LUT,
                "alpha": alpha,
                "size": size,
            }
        elif mode == "depth":
            payload = {
                "mode": "depth",
                "solid": DEFAULT_SOLID,
                "lut": w["palette"].currentText(),
                "alpha": alpha,
                "size": size,
            }
        else:  # "end-to-end"
            payload = {
                "mode": "end-to-end",
                "solid": DEFAULT_SOLID,
                "lut": w["palette"].currentText(),
                "alpha": alpha,
                "size": size,
            }

        self.changed.emit(base, payload)

    def _fill_palette_for_mode(self, base: str, mode: str, *, solid_default: str, lut_default: str):
        w = self._widgets_by_base[base]
        palette = w["palette"]

        palette.blockSignals(True)
        palette.clear()

        if mode == "solid":
            palette.setEnabled(True)
            palette.addItems(SOLID_COLOR_CHOICES)
            palette.setCurrentText(solid_default if solid_default in SOLID_COLOR_CHOICES else DEFAULT_SOLID)

        elif mode == "tid":
            palette.setEnabled(False)
            palette.addItem("—")

        else:  # "depth" or "end-to-end"
            palette.setEnabled(True)
            palette.addItems(LUT_CHOICES)
            palette.setCurrentText(lut_default if lut_default in LUT_CHOICES else DEFAULT_LUT)

        palette.blockSignals(False)

    def rebuild(self, base_names, current_settings_by_base: dict):
        self.table.setRowCount(0)
        self._row_by_base.clear()
        self._widgets_by_base.clear()

        for row, base in enumerate(base_names):
            self.table.insertRow(row)
            self._row_by_base[base] = row

            s = (current_settings_by_base or {}).get(base, {}) or {}
            mode = s.get("mode", DEFAULT_MODE)
            if mode not in MODE_CHOICES:
                mode = DEFAULT_MODE

            solid = s.get("solid", DEFAULT_SOLID)
            lut = s.get("lut", DEFAULT_LUT)

            alpha = float(s.get("alpha", DEFAULT_ALPHA))
            alpha = max(0.0, min(1.0, alpha))

            size = int(s.get("size", DEFAULT_SIZE_2D))
            size = max(1, min(50, size))

            # --- File cell (text) ---
            item = QtWidgets.QTableWidgetItem(base)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_FILE, item)

            # --- Mode combo ---
            mode_combo = QtWidgets.QComboBox()
            mode_combo.addItems(MODE_CHOICES)
            mode_combo.setCurrentText(mode)

            # --- Palette combo ---
            palette_combo = QtWidgets.QComboBox()

            # --- Alpha spin ---
            alpha_spin = QtWidgets.QDoubleSpinBox()
            alpha_spin.setRange(0.0, 1.0)
            alpha_spin.setDecimals(2)
            alpha_spin.setSingleStep(0.05)
            alpha_spin.setValue(alpha)

            # --- Size spin ---
            size_spin = QtWidgets.QSpinBox()
            size_spin.setRange(1, 50)
            size_spin.setValue(size)

            # --- mbm source ---
            src_chk = QtWidgets.QCheckBox()
            src_chk.setChecked(bool(s.get("mbm_source", False)))
            src_chk.stateChanged.connect(partial(self._on_source_toggled, base))

            cell = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignCenter)
            lay.addWidget(src_chk)

            self.table.setCellWidget(row, self.COL_MBM_SOURCE, cell)

            self._widgets_by_base[base] = dict(
                mode=mode_combo,
                palette=palette_combo,
                alpha=alpha_spin,
                size=size_spin,
                mbm_source=src_chk,     # self._widgets_by_base[base]["mbm_source"] = src_chk # Different??
            )

            # initial palette fill based on mode
            self._fill_palette_for_mode(base, mode, solid_default=solid, lut_default=lut)

            # connect signals (avoid lambda capturing row; capture base)
            mode_combo.currentTextChanged.connect(lambda m, b=base: self._on_mode_changed(b, m))
            palette_combo.currentTextChanged.connect(lambda _, b=base: self._emit_row(b))
            alpha_spin.valueChanged.connect(lambda _, b=base: self._emit_row(b))
            size_spin.valueChanged.connect(lambda _, b=base: self._emit_row(b))

            self.table.setCellWidget(row, self.COL_MODE, mode_combo)
            self.table.setCellWidget(row, self.COL_PALETTE, palette_combo)
            self.table.setCellWidget(row, self.COL_ALPHA, alpha_spin)
            self.table.setCellWidget(row, self.COL_SIZE, size_spin)

        self.table.resizeRowsToContents()

    def _on_mode_changed(self, base: str, mode: str):
        # keep existing palette selection if possible, otherwise fall back
        w = self._widgets_by_base.get(base)
        if not w:
            return

        current_palette = w["palette"].currentText()

        # infer defaults from stored current_palette when switching
        solid_default = current_palette if current_palette in SOLID_COLOR_CHOICES else DEFAULT_SOLID
        lut_default = current_palette if current_palette in LUT_CHOICES else DEFAULT_LUT

        self._fill_palette_for_mode(base, mode, solid_default=solid_default, lut_default=lut_default)
        self._emit_row(base)

    def _on_source_toggled(self, base: str, state: int):
        if state != Qt.Checked:
            self.changed.emit(base, {"mbm_source": False})
            return

        # uncheck all others
        for b, w in self._widgets_by_base.items():
            chk = w.get("mbm_source")
            if chk is None:
                continue
            if b != base:
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)

        self.changed.emit(base, {"mbm_source": True, "mbm_source_base": base})

class ParametersDialog(QtWidgets.QDialog):
    """Dialog to edit analysis parameters, bound to MainWindow spinboxes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.setModal(True)
        self.resize(900, 400)

        root = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QGridLayout()
        root.addLayout(form)

        self.min_trace = QtWidgets.QSpinBox()
        self.min_trace.setRange(1, 9999)

        self.zcorr = QtWidgets.QDoubleSpinBox()
        self.zcorr.setDecimals(6)
        self.zcorr.setRange(0, 1000)

        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setDecimals(1)
        self.scale.setRange(1, 1e15)

        self.bin_size = QtWidgets.QDoubleSpinBox()
        self.bin_size.setDecimals(1)
        self.bin_size.setRange(1, 1e9)

        self.cfr_bin_count = QtWidgets.QSpinBox()
        self.cfr_bin_count.setRange(5, 500)
        self.cfr_bin_count.setValue(50)

        self.scalebar_nm = QtWidgets.QDoubleSpinBox()
        self.scalebar_nm.setDecimals(1)
        self.scalebar_nm.setRange(1, 1e9)
        self.scalebar_nm.setSingleStep(10.0)

        form.addWidget(QtWidgets.QLabel("Min trace length:"), 0, 0)
        form.addWidget(self.min_trace, 0, 1)
        form.addWidget(QtWidgets.QLabel("Z correction factor:"), 0, 2)
        form.addWidget(self.zcorr, 0, 3)

        form.addWidget(QtWidgets.QLabel("Scale factor:"), 1, 0)
        form.addWidget(self.scale, 1, 1)
        form.addWidget(QtWidgets.QLabel("EFO histogram bin size:"), 1, 2)
        form.addWidget(self.bin_size, 1, 3)
        form.addWidget(QtWidgets.QLabel("Scale bar size (nm):"), 2, 0)
        form.addWidget(self.scalebar_nm, 2, 1)
        form.addWidget(QtWidgets.QLabel("CFR histogram bins:"), 2, 2)
        form.addWidget(self.cfr_bin_count, 2, 3)


        note = QtWidgets.QLabel("Data is usually recorded in meters, set scale factor to 1e9 to convert to nanometers.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #B0B3B8;")

        form.addWidget(note, 3, 0, 1, 4)

        self.mbm_table = QtWidgets.QTableWidget(0, 3)
        self.mbm_table.setHorizontalHeaderLabels(["File", "MBM folder", ""])
        self.mbm_table.verticalHeader().setVisible(False)
        self.mbm_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.mbm_table.horizontalHeader().setStretchLastSection(False)
        self.mbm_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.mbm_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.mbm_table.setColumnWidth(2, 120)
        self.mbm_table.setMinimumHeight(180)

        root.addWidget(QtWidgets.QLabel("MBM folders (per file):"))
        root.addWidget(self.mbm_table)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        root.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def set_values(self, *, min_trace_len, z_corr, scale, bin_size, cfr_bin_count, scalebar_nm):
        self.min_trace.setValue(int(min_trace_len))
        self.zcorr.setValue(float(z_corr))
        self.scale.setValue(float(scale))
        self.bin_size.setValue(float(bin_size))
        self.cfr_bin_count.setValue(int(cfr_bin_count))
        self.scalebar_nm.setValue(float(scalebar_nm))

    def values(self):
        return dict(
            min_trace_len=int(self.min_trace.value()),
            z_corr=float(self.zcorr.value()),
            scale=float(self.scale.value()),
            bin_size=float(self.bin_size.value()),
            cfr_bin_count=int(self.cfr_bin_count.value()),
            scalebar_nm=float(self.scalebar_nm.value()),
        )

    def set_mbm_rows(self, file_paths):
        self.mbm_table.setRowCount(0)
        for r, fp in enumerate(file_paths):
            base = os.path.splitext(os.path.basename(fp))[0]
            default_mbm = os.path.join(os.path.dirname(fp), "grd", "mbm")

            self.mbm_table.insertRow(r)
            self.mbm_table.setItem(r, 0, QtWidgets.QTableWidgetItem(base))

            le = QtWidgets.QLineEdit(default_mbm)
            self.mbm_table.setCellWidget(r, 1, le)

            btn = QtWidgets.QPushButton("Browse mbm…")
            btn.clicked.connect(lambda _, rr=r: self._browse_mbm(rr))
            self.mbm_table.setCellWidget(r, 2, btn)

    def _browse_mbm(self, row):
        le = self.mbm_table.cellWidget(row, 1)
        start = le.text().strip() if le else ""
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select MBM folder", start)
        if p and le:
            le.setText(p)

    def mbm_map(self):
        out = {}
        for r in range(self.mbm_table.rowCount()):
            base = self.mbm_table.item(r, 0).text()
            le = self.mbm_table.cellWidget(r, 1)
            out[base] = le.text().strip() if le else ""
        return out

class BinningWindow(QtWidgets.QMainWindow):
    """
    One window, one plot.
    Table: File | XX (multi-selection, max 2) | LUT
    - 1 selected file: heatmap or gaussian (based on mode)
    - 2 selected files: single merged overlay image from both binned maps
    """
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Binning")
        self.resize(1100, 700)

        self._mw = main_window
        self._widgets_by_base = {}
        self._selected_bases = []
        self._base_order = []
        self._max_value_applied_by_base = {}
        self._max_value_pending_by_base = {}
        self._max_value_spin_by_base = {}
        self._confocal_reset_pending = False
        self._confocal_max_applied = 750.0
        self._confocal_max_pending = 750.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 2, 2, 2)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        root.addLayout(top, 1)

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(4, 0, 0, 0)
        left.setSpacing(6)
        top.addLayout(left, 1)

        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["File", "XX", "LUT"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        self.table.setMinimumWidth(460)
        self.table.setMaximumWidth(580)
        self.table.setMaximumHeight(260)
        left.addWidget(self.table, 1)

        ctrl = QtWidgets.QGridLayout()
        ctrl.setContentsMargins(2, 0, 0, 0)
        ctrl.setHorizontalSpacing(8)
        ctrl.setVerticalSpacing(6)
        left.addLayout(ctrl)
        self.ctrl = ctrl

        ctrl.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["heatmap", "gaussian"])
        self.mode_combo.setCurrentText("heatmap")
        ctrl.addWidget(self.mode_combo, 0, 1)

        ctrl.addWidget(QtWidgets.QLabel("Pixel size (nm):"), 0, 2)
        self.px_spin = QtWidgets.QDoubleSpinBox()
        self.px_spin.setDecimals(1)
        self.px_spin.setRange(0.1, 1e6)
        self.px_spin.setSingleStep(1.0)
        self.px_spin.setValue(4.0)
        ctrl.addWidget(self.px_spin, 0, 3)

        ctrl.addWidget(QtWidgets.QLabel("Scale:"), 1, 0)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["linear", "log10(count+1)"])
        self.scale_combo.setCurrentText("linear")
        ctrl.addWidget(self.scale_combo, 1, 1)

        # small spacer to lower the confocal section a bit
        ctrl.setRowMinimumHeight(2, 8)

        self.confocal_sep = make_labeled_separator("Confocal")
        ctrl.addWidget(self.confocal_sep, 3, 0, 1, 6)

        self.confocal_btn = QtWidgets.QPushButton("Path")
        self.confocal_btn.clicked.connect(self._browse_confocal_path)
        ctrl.addWidget(self.confocal_btn, 4, 0)

        self.confocal_path_edit = QtWidgets.QLineEdit()
        self.confocal_path_edit.setPlaceholderText("Select .tif or .tiff file")
        ctrl.addWidget(self.confocal_path_edit, 4, 1, 1, 2)

        self.confocal_show_chk = QtWidgets.QCheckBox("show")
        self.confocal_show_chk.toggled.connect(self._on_confocal_show_toggled)
        ctrl.addWidget(self.confocal_show_chk, 4, 3)

        ctrl.addWidget(QtWidgets.QLabel("Scale:"), 4, 4)
        self.confocal_scale_spin = QtWidgets.QDoubleSpinBox()
        self.confocal_scale_spin.setDecimals(1)
        self.confocal_scale_spin.setRange(0.1, 10.0)
        self.confocal_scale_spin.setSingleStep(0.1)
        self.confocal_scale_spin.setValue(1.0)
        self.confocal_scale_spin.valueChanged.connect(lambda _: self.refresh_plot(keep_view=True))
        ctrl.addWidget(self.confocal_scale_spin, 4, 5)

        ctrl.addWidget(QtWidgets.QLabel("LUT:"), 5, 0)
        self.confocal_lut_combo = QtWidgets.QComboBox()
        self.confocal_lut_combo.addItems(LUT_CHOICES)
        self.confocal_lut_combo.setCurrentText(DEFAULT_LUT_BIN if DEFAULT_LUT_BIN in LUT_CHOICES else LUT_CHOICES[0])
        self.confocal_lut_combo.currentTextChanged.connect(lambda _: self.refresh_plot(keep_view=True))
        ctrl.addWidget(self.confocal_lut_combo, 5, 1)

        ctrl.addWidget(QtWidgets.QLabel("Max value:"), 5, 2)
        self.confocal_max_spin = QtWidgets.QDoubleSpinBox()
        self.confocal_max_spin.setDecimals(0)
        self.confocal_max_spin.setRange(0, 750)
        self.confocal_max_spin.setSingleStep(1.0)
        self.confocal_max_spin.setValue(750)
        self.confocal_max_spin.valueChanged.connect(self._on_confocal_max_value_changed)
        ctrl.addWidget(self.confocal_max_spin, 5, 3)

        ctrl.addWidget(QtWidgets.QLabel("rotate/flip:"), 6, 0)
        self.confocal_orient_combo = QtWidgets.QComboBox()
        self.confocal_orient_combo.addItems([
            "none",
            "flip vertical",
            "flip horizontal",
            "rotate 90",
            "rotate 180",
            "rotate 270",
        ])
        self.confocal_orient_combo.currentTextChanged.connect(lambda _: self.refresh_plot(keep_view=True))
        ctrl.addWidget(self.confocal_orient_combo, 6, 1, 1, 3)

        self.intensity_sep = make_labeled_separator("Intensity adjustment")
        ctrl.addWidget(self.intensity_sep, 7, 0, 1, 6)

        self._max_value_label_widgets = []
        self._max_value_spin_widgets = []
        self._max_value_placeholder = None

        self.apply_settings_btn = QtWidgets.QPushButton("Apply")
        self.apply_settings_btn.clicked.connect(self._on_apply_settings)

        self.save_tif_btn = QtWidgets.QPushButton("Save as tif")
        self.save_tif_btn.clicked.connect(self._save_selected_as_tif)

        self.actions_widget = QtWidgets.QWidget()
        self.actions_layout = QtWidgets.QHBoxLayout(self.actions_widget)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.actions_layout.setSpacing(8)
        self.actions_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.actions_layout.addWidget(self.apply_settings_btn)
        self.actions_layout.addWidget(self.save_tif_btn)
        self._sync_action_button_sizes()

        self.tone_hint_lbl = QtWidgets.QLabel("Values above each channel's Max value are clipped.")
        self.tone_hint_lbl.setStyleSheet("color: #B0B3B8;")

        ctrl.setColumnStretch(6, 1)

        self.view = PlotlyView()
        top.addWidget(self.view, 3)

        left.addStretch(1)

        self._rebuild_max_value_controls()

    def _sync_action_button_sizes(self):
        w = max(self.apply_settings_btn.sizeHint().width(), self.save_tif_btn.sizeHint().width())
        self.apply_settings_btn.setFixedWidth(w)
        self.save_tif_btn.setFixedWidth(w)

    def rebuild(self, base_names):
        prev_selected = [b for b in self._selected_bases if b in set(base_names)]
        self._base_order = list(base_names)
        self._max_value_applied_by_base = {b: v for b, v in self._max_value_applied_by_base.items() if b in set(base_names)}
        self._max_value_pending_by_base = {b: v for b, v in self._max_value_pending_by_base.items() if b in set(base_names)}

        self.table.setRowCount(0)
        self._widgets_by_base.clear()
        self._selected_bases = []

        for r, base in enumerate(base_names):
            self.table.insertRow(r)

            item = QtWidgets.QTableWidgetItem(f" {base}")
            self.table.setItem(r, 0, item)

            xx = QtWidgets.QCheckBox()
            xx.toggled.connect(lambda checked, b=base: self._on_xx_toggled(b, checked))
            xx_cell = QtWidgets.QWidget()
            xx_lay = QtWidgets.QHBoxLayout(xx_cell)
            xx_lay.setContentsMargins(0, 0, 0, 0)
            xx_lay.setAlignment(Qt.AlignCenter)
            xx_lay.addWidget(xx)
            self.table.setCellWidget(r, 1, xx_cell)

            lut = QtWidgets.QComboBox()
            lut.addItems(LUT_CHOICES)
            lut.setCurrentText(DEFAULT_LUT_BIN)
            lut.currentTextChanged.connect(lambda _, b=base: self._on_settings_changed(b))
            self.table.setCellWidget(r, 2, lut)

            self._widgets_by_base[base] = dict(xx=xx, lut=lut)

        self.table.resizeRowsToContents()

        for base in prev_selected[:2]:
            w = self._widgets_by_base.get(base)
            if not w:
                continue
            chk = w["xx"]
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.blockSignals(False)
            self._selected_bases.append(base)

        if not self._selected_bases and base_names:
            first = base_names[0]
            w = self._widgets_by_base.get(first)
            if w:
                chk = w["xx"]
                chk.blockSignals(True)
                chk.setChecked(True)
                chk.blockSignals(False)
                self._selected_bases = [first]

        self._rebuild_max_value_controls()

        self.refresh_plot(keep_view=False)

    def _on_settings_changed(self, base: str):
        if base in self._selected_bases:
            self.refresh_plot(keep_view=True)

    def _on_xx_toggled(self, base: str, checked: bool):
        if checked:
            if base not in self._selected_bases:
                if len(self._selected_bases) >= 2:
                    w = self._widgets_by_base.get(base)
                    if w and "xx" in w:
                        w["xx"].blockSignals(True)
                        w["xx"].setChecked(False)
                        w["xx"].blockSignals(False)
                    QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Select at most 2 files for overlay")
                    return
                self._selected_bases.append(base)
            self._rebuild_max_value_controls()
            self.refresh_plot(keep_view=True)
            return

        if base in self._selected_bases:
            self._selected_bases.remove(base)

        self._rebuild_max_value_controls()

        if not self._selected_bases:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text="No file selected", x=0.5, y=0.5, showarrow=False)])
            apply_plot_theme(fig, is3d=False)
            self.view.update_fig(fig, reset_view=True, is3d=False)
            return

        self.refresh_plot(keep_view=True)

    def _get_max_value_for_base(self, base: str) -> float:
        value = float(self._max_value_applied_by_base.get(base, 10.0))
        if not np.isfinite(value) or value <= 0:
            value = 10.0
        self._max_value_applied_by_base[base] = value
        return value

    def _get_pending_max_value_for_base(self, base: str) -> float:
        if base in self._max_value_pending_by_base:
            value = float(self._max_value_pending_by_base.get(base, 10.0))
        else:
            value = self._get_max_value_for_base(base)
        if not np.isfinite(value) or value <= 0:
            value = 10.0
        self._max_value_pending_by_base[base] = value
        return value

    def _on_apply_settings(self):
        for base in self._base_order:
            self._max_value_applied_by_base[base] = self._get_pending_max_value_for_base(base)
        self._confocal_max_applied = float(self._confocal_max_pending)
        self.refresh_plot(keep_view=True)

    def _on_confocal_max_value_changed(self, value: float):
        if not np.isfinite(value) or value < 0:
            return
        self._confocal_max_pending = float(value)

    def _is_tiff_path(self, path: str) -> bool:
        ext = os.path.splitext(str(path).strip())[1].lower()
        return ext in (".tif", ".tiff")

    def _browse_confocal_path(self):
        start = self.confocal_path_edit.text().strip()
        if not start:
            start = self._mw.data_edit.text().strip() if hasattr(self._mw, "data_edit") else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select confocal image",
            start,
            "TIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        if not self._is_tiff_path(path):
            QtWidgets.QMessageBox.critical(self, "Confocal image", "Selected file is not .tif/.tiff")
            return

        # New confocal file -> reset scale to default.
        if hasattr(self, "confocal_scale_spin"):
            self.confocal_scale_spin.blockSignals(True)
            self.confocal_scale_spin.setValue(1)
            self.confocal_scale_spin.blockSignals(False)

        if hasattr(self, "confocal_orient_combo"):
            self.confocal_orient_combo.blockSignals(True)
            self.confocal_orient_combo.setCurrentText("none")
            self.confocal_orient_combo.blockSignals(False)

        self.confocal_path_edit.setText(path)
        if self.confocal_show_chk.isChecked():
            self._confocal_reset_pending = True
            self.refresh_plot(keep_view=True)

    def _on_confocal_show_toggled(self, checked: bool):
        if checked:
            path = self.confocal_path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, "Confocal image", "Select a .tif/.tiff file first.")
                self.confocal_show_chk.blockSignals(True)
                self.confocal_show_chk.setChecked(False)
                self.confocal_show_chk.blockSignals(False)
                return
            if not self._is_tiff_path(path):
                QtWidgets.QMessageBox.critical(self, "Confocal image", "Selected file is not .tif/.tiff")
                self.confocal_show_chk.blockSignals(True)
                self.confocal_show_chk.setChecked(False)
                self.confocal_show_chk.blockSignals(False)
                return
            self._confocal_reset_pending = True
        self.refresh_plot(keep_view=True)

    def _load_confocal_image(self, path: str):
        try:
            import tifffile
            img = np.asarray(tifffile.imread(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Confocal image", f"Failed to load file:\n{exc}")
            return None

        if img.ndim < 2:
            QtWidgets.QMessageBox.critical(self, "Confocal image", "Confocal image must be at least 2D.")
            return None

        # Reduce to 2D robustly:
        # choose the two largest axes as spatial (y, x), then max-project all others.
        if img.ndim >= 3:
            shape = tuple(int(s) for s in img.shape)
            largest = np.argsort(shape)[-2:]
            spatial_axes = tuple(sorted(int(a) for a in largest))

            # Project non-spatial axes without reordering/flipping/rotating image axes.
            non_spatial_axes = tuple(a for a in range(img.ndim) if a not in spatial_axes)
            if len(non_spatial_axes) == 1 and img.shape[non_spatial_axes[0]] in (3, 4):
                color_axis = non_spatial_axes[0]
                img = np.mean(np.take(img, indices=(0, 1, 2), axis=color_axis), axis=color_axis)
            elif non_spatial_axes:
                img = np.max(img, axis=non_spatial_axes)

        if hasattr(self, "confocal_orient_combo"):
            orient = self.confocal_orient_combo.currentText().strip().lower()
            if orient == "flip vertical":
                img = np.flipud(img)
            elif orient == "flip horizontal":
                img = np.fliplr(img)
            elif orient == "rotate 90":
                img = np.rot90(img, 1)
            elif orient == "rotate 180":
                img = np.rot90(img, 2)
            elif orient == "rotate 270":
                img = np.rot90(img, 3)

        # Keep native 8-bit values unchanged.
        if img.dtype == np.uint8:
            return img

        # For non-8-bit images, remap intensities to 8-bit.
        imgf = np.asarray(img, dtype=float)
        finite = np.isfinite(imgf)
        if not np.any(finite):
            QtWidgets.QMessageBox.critical(self, "Confocal image", "Confocal image has no finite values.")
            return None
        imgf = np.where(finite, imgf, 0.0)

        vals = imgf[finite]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(imgf, dtype=np.uint8)

        img8 = np.clip((imgf - lo) / (hi - lo), 0.0, 1.0)
        return np.asarray(np.round(255.0 * img8), dtype=np.uint8)

    def _compose_with_confocal(self, left_fig: go.Figure) -> go.Figure:
        if not self.confocal_show_chk.isChecked():
            return left_fig
        if left_fig is None or len(getattr(left_fig, "data", [])) == 0:
            return left_fig

        conf_path = self.confocal_path_edit.text().strip()
        if not conf_path or not self._is_tiff_path(conf_path):
            return left_fig

        conf = self._load_confocal_image(conf_path)
        if conf is None:
            return left_fig

        try:
            xmin = xmax = ymin = ymax = None

            xr = None
            yr = None
            try:
                xr = list(getattr(getattr(left_fig.layout, "xaxis", None), "range", []) or [])
                yr = list(getattr(getattr(left_fig.layout, "yaxis", None), "range", []) or [])
            except Exception:
                xr = None
                yr = None

            if xr and len(xr) == 2 and yr and len(yr) == 2:
                xmin = float(min(xr[0], xr[1]))
                xmax = float(max(xr[0], xr[1]))
                ymin = float(min(yr[0], yr[1]))
                ymax = float(max(yr[0], yr[1]))
            else:
                xs = []
                ys = []
                for tr in left_fig.data:
                    tx = getattr(tr, "x", None)
                    ty = getattr(tr, "y", None)
                    if tx is not None:
                        try:
                            ax = np.asarray(tx, dtype=float).ravel()
                            ax = ax[np.isfinite(ax)]
                            if ax.size:
                                xs.append(ax)
                        except Exception:
                            pass
                    if ty is not None:
                        try:
                            ay = np.asarray(ty, dtype=float).ravel()
                            ay = ay[np.isfinite(ay)]
                            if ay.size:
                                ys.append(ay)
                        except Exception:
                            pass

                if not xs or not ys:
                    return left_fig

                x_all = np.concatenate(xs)
                y_all = np.concatenate(ys)
                xmin = float(np.min(x_all))
                xmax = float(np.max(x_all))
                ymin = float(np.min(y_all))
                ymax = float(np.max(y_all))

            width0 = max(float(xmax - xmin), 1e-9)
            height0 = max(float(ymax - ymin), 1e-9)

            scale_factor = float(self.confocal_scale_spin.value()) if hasattr(self, "confocal_scale_spin") else 1.0
            scale_factor = max(0.1, scale_factor)
            if abs(scale_factor - 1.0) > 1e-9:
                src_ny, src_nx = conf.shape
                dst_ny = max(1, int(np.round(src_ny * scale_factor)))
                dst_nx = max(1, int(np.round(src_nx * scale_factor)))
                # Nearest-neighbor resize keeps intensity values unchanged.
                y_idx = np.minimum((np.arange(dst_ny) / scale_factor).astype(np.int64), src_ny - 1)
                x_idx = np.minimum((np.arange(dst_nx) / scale_factor).astype(np.int64), src_nx - 1)
                conf = conf[np.ix_(y_idx, x_idx)]

            ny, nx = conf.shape

            # Infer binned raster shape; if confocal is larger, virtually pad binned extents.
            bin_ny, bin_nx = ny, nx
            for tr in left_fig.data:
                if getattr(tr, "type", "") != "heatmap":
                    continue
                zt = np.asarray(getattr(tr, "z", None))
                if zt.ndim == 2 and zt.size > 0:
                    bin_ny, bin_nx = int(zt.shape[0]), int(zt.shape[1])
                    break

            target_ny = max(int(bin_ny), int(ny))
            target_nx = max(int(bin_nx), int(nx))

            # Pad confocal to the center if binned is larger.
            pad_y = max(0, target_ny - int(ny))
            pad_x = max(0, target_nx - int(nx))
            if pad_y or pad_x:
                pad_top = pad_y // 2
                pad_bottom = pad_y - pad_top
                pad_left = pad_x // 2
                pad_right = pad_x - pad_left
                conf = np.pad(
                    conf,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
                ny, nx = conf.shape

            px_x = width0 / max(float(bin_nx), 1.0)
            px_y = height0 / max(float(bin_ny), 1.0)
            width = max(px_x * float(target_nx), 1e-9)
            height = max(px_y * float(target_ny), 1e-9)
            xmax_eff = float(xmin + width)
            ymax_eff = float(ymin + height)
            gap = 0.03 * width

            max_side = 1024
            if ny > max_side or nx > max_side:
                sy = max(1, int(np.ceil(ny / max_side)))
                sx = max(1, int(np.ceil(nx / max_side)))
                # Preserve sparse bright structures by max-pooling instead of strided decimation.
                pad_y = (-ny) % sy
                pad_x = (-nx) % sx
                if pad_y or pad_x:
                    conf = np.pad(conf, ((0, pad_y), (0, pad_x)), mode="edge")
                yy, xx = conf.shape
                conf = conf.reshape(yy // sy, sy, xx // sx, sx).max(axis=(1, 3))
                ny, nx = conf.shape

            conf_xmax = float(xmax_eff + gap + width)
            divider_x = float(xmax_eff + 0.5 * gap)

            fig = go.Figure(left_fig)
            for tr in fig.data:
                if getattr(tr, "type", "") == "heatmap":
                    tr.opacity = 1.0

            x0_conf = float(xmax_eff + gap)
            y0_conf = float(ymin)
            dx_conf = float(width / max(nx, 1))
            dy_conf = float(height / max(ny, 1))

            conf_u8 = np.asarray(conf, dtype=np.uint8)
            img_max = float(np.max(conf_u8)) if conf_u8.size > 0 else 0.0
            conf_max = img_max
            if hasattr(self, "confocal_max_spin"):
                spin = self.confocal_max_spin
                reset_for_new_image = bool(getattr(self, "_confocal_reset_pending", False))
                upper = max(750.0, img_max * 16.0, 10000.0)
                spin.blockSignals(True)
                spin.setRange(0.0, upper)
                if reset_for_new_image:
                    spin.setValue(img_max)
                    self._confocal_max_pending = float(img_max)
                    self._confocal_max_applied = float(img_max)
                spin.blockSignals(False)
                conf_max = float(self._confocal_max_applied)

            conf_max = max(float(conf_max), 1e-9)
            conf_norm = np.clip(conf_u8.astype(np.float32) / conf_max, 0.0, 1.0)
            conf_lut = self.confocal_lut_combo.currentText().strip() if hasattr(self, "confocal_lut_combo") else DEFAULT_LUT_BIN
            conf_rgb01 = _lut_rgb_image(conf_lut, conf_norm)
            conf_rgb_u8 = (255.0 * np.clip(conf_rgb01, 0.0, 1.0)).astype(np.uint8)

            try:
                from PIL import Image
                pil_img = Image.fromarray(conf_rgb_u8, mode="RGB")
                fig.add_layout_image(
                    dict(
                        source=pil_img,
                        xref="x",
                        yref="y",
                        x=x0_conf,
                        y=float(ymax_eff),
                        sizex=float(width),
                        sizey=float(height),
                        sizing="stretch",
                        opacity=1.0,
                        layer="above",
                    )
                )
            except Exception:
                fig.add_trace(
                    go.Heatmap(
                        z=conf_u8,
                        x0=x0_conf,
                        dx=dx_conf,
                        y0=y0_conf,
                        dy=dy_conf,
                        colorscale=_resolve_overlay_colorscale(conf_lut),
                        zmin=0.0,
                        zmax=255.0,
                        showscale=False,
                        hoverinfo="skip",
                        opacity=1.0,
                    )
                )

            fig.update_layout(dragmode="pan")
            fig.update_xaxes(range=[xmin, conf_xmax], autorange=False)
            fig.update_yaxes(range=[ymin, ymax_eff], autorange=False)
            fig.add_shape(
                type="line",
                x0=divider_x,
                x1=divider_x,
                y0=ymin,
                y1=ymax_eff,
                xref="x",
                yref="y",
                line=dict(color="#9AA0A6", width=2),
            )
            fig.update_xaxes(showline=False, ticks="", showticklabels=False)
            fig.update_yaxes(showline=False, ticks="", showticklabels=False)
            return fig
        except Exception:
            return left_fig

    def _on_max_value_changed(self, base: str, value: float):
        if not np.isfinite(value) or value <= 0:
            return
        self._max_value_pending_by_base[base] = float(value)

    def _rebuild_max_value_controls(self):
        for widget in self._max_value_label_widgets:
            self.ctrl.removeWidget(widget)
            widget.deleteLater()
        for widget in self._max_value_spin_widgets:
            self.ctrl.removeWidget(widget)
            widget.deleteLater()
        self._max_value_label_widgets = []
        self._max_value_spin_widgets = []

        if self._max_value_placeholder is not None:
            self.ctrl.removeWidget(self._max_value_placeholder)
            self._max_value_placeholder.deleteLater()
            self._max_value_placeholder = None

        self._max_value_spin_by_base = {}

        all_bases = list(self._base_order)
        max_rows_used = 1
        if not all_bases:
            self._max_value_placeholder = QtWidgets.QLabel("No images loaded.")
            self._max_value_placeholder.setStyleSheet("color: #B0B3B8;")
            self.ctrl.addWidget(self._max_value_placeholder, 8, 0, 1, 4)
        else:
            max_rows_used = max(1, (len(all_bases) + 1) // 2)

        for idx, base in enumerate(all_bases):
            row = 8 + (idx // 2)
            col_block = (idx % 2) * 2

            label = QtWidgets.QLabel(f"Max value {idx + 1}:")
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0.001, 1e9)
            spin.setSingleStep(1.0)
            spin.setValue(self._get_pending_max_value_for_base(base))
            spin.valueChanged.connect(lambda value, b=base: self._on_max_value_changed(b, value))

            self.ctrl.addWidget(label, row, col_block)
            self.ctrl.addWidget(spin, row, col_block + 1)
            self._max_value_label_widgets.append(label)
            self._max_value_spin_widgets.append(spin)
            self._max_value_spin_by_base[base] = spin

        actions_row = 8 + max_rows_used
        self._sync_action_button_sizes()
        self.ctrl.addWidget(self.actions_widget, actions_row, 0, 1, 4)

        spacer_row = actions_row + 1
        hint_row = actions_row + 2
        self.ctrl.setRowMinimumHeight(spacer_row, 14)
        self.ctrl.addWidget(self.tone_hint_lbl, hint_row, 0, 1, 4)

    def _selected_max_values(self, selected):
        out = []
        for base, _lut in selected:
            out.append(self._get_max_value_for_base(base))
        return out

    def _get_selected_settings(self):
        out = []
        for base in self._selected_bases:
            w = self._widgets_by_base.get(base)
            if not w:
                continue
            out.append((
                base,
                w["lut"].currentText() if "lut" in w else DEFAULT_LUT_BIN,
            ))
        return out

    def _get_saved_arr_for_base(self, base: str):
        arr = self._mw._filtered_by_base.get(base)
        if arr is None:
            return None
        if getattr(self._mw, "_mbm_enabled", False):
            arr = self._mw._aligned_arr_by_base.get(base, arr)
        return arr

    def _compute_selected_channel_images(
        self,
        selected,
        pixel_size_nm: float,
        scale_mode: str,
        render_mode: str,
        max_values,
    ):
        channel_xy = []
        channel_names = []

        for base, _lut in selected:
            arr = self._get_saved_arr_for_base(base)
            if arr is None:
                raise ValueError(f"No saved (EFO-filtered) data for: {base}")
            x, y = _xy_from_arr(arr)
            channel_xy.append((x, y))
            channel_names.append(base)

        non_empty = [(x, y) for (x, y) in channel_xy if len(x) > 0]
        if not non_empty:
            raise ValueError("No valid points in selected channel(s).")

        pixel_size_nm = max(float(pixel_size_nm), 0.1)
        render_mode = (render_mode or "heatmap").strip().lower()

        x_all = np.concatenate([x for x, _ in non_empty])
        y_all = np.concatenate([y for _, y in non_empty])

        xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))

        if render_mode == "gaussian":
            pad = 3.0 * pixel_size_nm
            xmin -= pad
            xmax += pad
            ymin -= pad
            ymax += pad

        if xmax == xmin:
            xmax = xmin + pixel_size_nm
        if ymax == ymin:
            ymax = ymin + pixel_size_nm

        nx = max(int(np.ceil((xmax - xmin) / pixel_size_nm)), 1)
        ny = max(int(np.ceil((ymax - ymin) / pixel_size_nm)), 1)

        x_edges = xmin + np.arange(nx + 1) * pixel_size_nm
        y_edges = ymin + np.arange(ny + 1) * pixel_size_nm

        images = []
        for idx, (x, y) in enumerate(channel_xy):
            max_value = max(float(max_values[idx]) if idx < len(max_values) else 10.0, 1e-9)
            if len(x) == 0:
                img_norm = np.zeros((ny, nx), dtype=float)
            elif render_mode == "gaussian":
                sigma_nm = pixel_size_nm
                bounds = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
                img, _ = render_gaussians_xy(
                    x,
                    y,
                    sigma_nm=sigma_nm,
                    pixel_size_nm=pixel_size_nm,
                    n_sigma=3.0,
                    bounds=bounds,
                )
                img_norm = _clip_to_scale_max(img, max_value=max_value)
            else:
                H = np.zeros((ny, nx), dtype=float)
                H, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges))
                img_norm = _clip_hist_for_overlay(H, scale_mode, max_value=max_value)

            images.append(np.asarray(img_norm, dtype=np.float32))

        return images, channel_names

    def _save_selected_as_tif(self):
        selected = self._get_selected_settings()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "Save as tif", "No channel selected.")
            return

        pixel_size_nm = float(self.px_spin.value()) if hasattr(self, "px_spin") else 4.0
        scale_mode = self.scale_combo.currentText().strip().lower() if hasattr(self, "scale_combo") else "linear"
        render_mode = self.mode_combo.currentText().strip().lower() if hasattr(self, "mode_combo") else "heatmap"
        max_values = self._selected_max_values(selected)

        try:
            images, channel_names = self._compute_selected_channel_images(
                selected,
                pixel_size_nm=pixel_size_nm,
                scale_mode=scale_mode,
                render_mode=render_mode,
                max_values=max_values,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save as tif", str(exc))
            return

        stack_cxy = np.stack(images, axis=0).astype(np.float32)

        default_name = "_".join(channel_names) + ".tif"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save selected channels as tif",
            default_name,
            "TIFF (*.tif *.tiff)",
        )
        if not out_path:
            return

        root, ext = os.path.splitext(out_path)
        if ext.lower() not in (".tif", ".tiff"):
            out_path = root + ".tif"

        try:
            import tifffile
            tifffile.imwrite(out_path, stack_cxy)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save as tif", f"Failed to save tif:\n{exc}")
            return

        QtWidgets.QMessageBox.information(
            self,
            "Save as tif",
            f"Saved {out_path}\nshape={tuple(stack_cxy.shape)}",
        )

    def refresh_plot(self, keep_view: bool = True):
        selected = self._get_selected_settings()
        if not selected:
            return

        pixel_size_nm = float(self.px_spin.value()) if hasattr(self, "px_spin") else 4.0
        scale_mode = self.scale_combo.currentText().strip().lower() if hasattr(self, "scale_combo") else "linear"
        render_mode = self.mode_combo.currentText().strip().lower() if hasattr(self, "mode_combo") else "heatmap"
        max_values = self._selected_max_values(selected)

        if len(selected) == 1:
            base, lut = selected[0]
            max_value = max_values[0] if max_values else 10.0
            arr = self._get_saved_arr_for_base(base)
            if arr is None:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(
                    text=f"No saved (EFO-filtered) data for: {base}<br>Apply filter first.",
                    x=0.5, y=0.5, showarrow=False
                )])
                apply_plot_theme(fig, is3d=False)
                self.view.update_fig(fig, reset_view=True, is3d=False)
                return

            if render_mode == "gaussian":
                fig = make_plotly_gaussian_from_arr(
                    arr,
                    pixel_size_nm=pixel_size_nm,
                    lut=lut,
                    title=None,
                    max_value=max_value,
                    show_colorbar=False,
                )
            else:
                fig = make_plotly_heatmap_from_arr(
                    arr,
                    pixel_size_nm=pixel_size_nm,
                    lut=lut,
                    title=None,
                    scale_mode=scale_mode,
                    max_value=max_value,
                    show_colorbar=False,
                )
        else:
            (base_a, lut_a), (base_b, lut_b) = selected[:2]
            arr_a = self._get_saved_arr_for_base(base_a)
            arr_b = self._get_saved_arr_for_base(base_b)

            missing = [b for b, a in ((base_a, arr_a), (base_b, arr_b)) if a is None]
            if missing:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(
                    text="No saved (EFO-filtered) data for:<br>" + "<br>".join(missing) + "<br>Apply filter first.",
                    x=0.5, y=0.5, showarrow=False
                )])
                apply_plot_theme(fig, is3d=False)
                self.view.update_fig(fig, reset_view=True, is3d=False)
                return

            fig = make_plotly_overlay_heatmap_from_two_arrs(
                arr_a,
                arr_b,
                pixel_size_nm=pixel_size_nm,
                lut_a=lut_a,
                lut_b=lut_b,
                title=None,
                scale_mode=scale_mode,
                render_mode=render_mode,
                max_value_a=max_values[0] if len(max_values) >= 1 else 10.0,
                max_value_b=max_values[1] if len(max_values) >= 2 else 10.0,
            )

        composed = self._compose_with_confocal(fig)
        force_reset_for_confocal = bool(getattr(self, "_confocal_reset_pending", False))
        self._confocal_reset_pending = False

        self.view.update_fig(
            composed,
            reset_view=(not keep_view) or force_reset_for_confocal,
            is3d=False,
            scalebar_nm=getattr(self._mw, "_scalebar_nm", 100.0),
        )

    def shutdown(self):
        try:
            if self.view is not None:
                self.view.shutdown()
        except Exception:
            pass


class DBSCANWindow(QtWidgets.QMainWindow):
    """DBSCAN on avg track centroids for one selected file at a time."""

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.setWindowTitle("DBSCAN")
        self.resize(1200, 760)

        self._mw = main_window
        self._base_buttons = {}
        self._current_base = None
        self._current_arr = None
        self._tracks_df = None
        self._labels = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left_wrap = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_wrap)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)
        left_wrap.setMinimumWidth(330)
        left_wrap.setMaximumWidth(420)
        root.addWidget(left_wrap, 0)

        files_group = QtWidgets.QGroupBox("Files")
        files_v = QtWidgets.QVBoxLayout(files_group)
        files_v.setContentsMargins(8, 8, 8, 8)
        files_v.setSpacing(6)

        self.files_scroll = QtWidgets.QScrollArea()
        self.files_scroll.setWidgetResizable(True)
        self.files_scroll.setMinimumWidth(300)
        self.files_scroll.setMaximumWidth(360)
        self.files_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.files_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.files_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.files_container = QtWidgets.QWidget()
        self.files_layout = QtWidgets.QVBoxLayout(self.files_container)
        self.files_layout.setContentsMargins(6, 8, 6, 8)
        self.files_layout.setSpacing(12)
        self.files_layout.addStretch(1)
        self.files_scroll.setWidget(self.files_container)
        files_v.addWidget(self.files_scroll, 1)
        left.addWidget(files_group, 0)

        viz_group = QtWidgets.QGroupBox("Visualization")
        viz_grid = QtWidgets.QGridLayout(viz_group)
        viz_grid.setContentsMargins(8, 8, 8, 8)
        viz_grid.setHorizontalSpacing(8)
        viz_grid.setVerticalSpacing(8)

        viz_grid.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.viz_mode_combo = QtWidgets.QComboBox()
        self.viz_mode_combo.addItems(MODE_CHOICES)
        self.viz_mode_combo.setCurrentText(DEFAULT_MODE if DEFAULT_MODE in MODE_CHOICES else MODE_CHOICES[0])
        viz_grid.addWidget(self.viz_mode_combo, 0, 1)

        viz_grid.addWidget(QtWidgets.QLabel("LUT / solid:"), 1, 0)
        self.viz_palette_combo = QtWidgets.QComboBox()
        viz_grid.addWidget(self.viz_palette_combo, 1, 1)

        self.viz_mode_combo.currentTextChanged.connect(self._on_viz_mode_changed)
        self.viz_palette_combo.currentTextChanged.connect(lambda _: self._refresh_tracks_plot())
        self._fill_viz_palette_for_mode(self.viz_mode_combo.currentText())

        left.addWidget(viz_group, 0)

        # Add clear visual separation between file list and DBSCAN action controls.
        left.addSpacing(14)

        params_group = QtWidgets.QGroupBox("DBSCAN")
        params_grid = QtWidgets.QGridLayout(params_group)
        params_grid.setContentsMargins(8, 8, 8, 8)
        params_grid.setHorizontalSpacing(8)
        params_grid.setVerticalSpacing(8)

        params_grid.addWidget(QtWidgets.QLabel("eps (nm):"), 0, 0)
        self.eps_spin = QtWidgets.QDoubleSpinBox()
        self.eps_spin.setDecimals(1)
        self.eps_spin.setRange(0.1, 1e6)
        self.eps_spin.setSingleStep(5.0)
        self.eps_spin.setValue(200.0)
        params_grid.addWidget(self.eps_spin, 0, 1)

        params_grid.addWidget(QtWidgets.QLabel("min_samples:"), 1, 0)
        self.min_samples_spin = QtWidgets.QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(3)
        params_grid.addWidget(self.min_samples_spin, 1, 1)

        params_grid.addWidget(QtWidgets.QLabel("Clustering mode:"), 2, 0)
        self.dimension_combo = QtWidgets.QComboBox()
        self.dimension_combo.addItems(["2D (x,y)", "3D (x,y,z)"])
        self.dimension_combo.setCurrentIndex(0)
        self.dimension_combo.currentTextChanged.connect(lambda _: self._refresh_tracks_plot())
        params_grid.addWidget(self.dimension_combo, 2, 1)

        self.run_btn = QtWidgets.QPushButton("Run DBSCAN")
        self.run_btn.clicked.connect(self.run_dbscan)
        params_grid.addWidget(self.run_btn, 3, 0, 1, 2)

        # Add extra breathing room between run action and log output.
        params_grid.setRowMinimumHeight(4, 12)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("Output will appear here...")
        params_grid.addWidget(self.log_text, 5, 0, 1, 2)

        self.clear_log_btn = QtWidgets.QPushButton("Clear log")
        self.clear_log_btn.clicked.connect(self.log_text.clear)

        self.save_btn = QtWidgets.QPushButton("Save DBSCAN")
        self.save_btn.clicked.connect(self.save_dbscan)

        # Match the button separation style used in other windows.
        for b in (self.run_btn, self.clear_log_btn, self.save_btn):
            b.setFixedHeight(40)

        params_grid.setRowMinimumHeight(6, 14)

        action_row = QtWidgets.QWidget()
        action_h = QtWidgets.QHBoxLayout(action_row)
        action_h.setContentsMargins(0, 6, 0, 8)
        action_h.setSpacing(12)
        action_h.addWidget(self.clear_log_btn)
        action_h.addWidget(self.save_btn)
        params_grid.addWidget(action_row, 7, 0, 1, 2)

        left.addWidget(params_group, 2)

        self.view = PlotlyView()
        self.view.setMinimumWidth(520)
        root.addWidget(self.view, 1)

        # Keep a stable split: controls panel narrow, viewer wide.
        root.setStretch(0, 1)
        root.setStretch(1, 3)

    def _append_log(self, text: str):
        self.log_text.appendPlainText(str(text))

    def _is_3d_mode(self) -> bool:
        return self.dimension_combo.currentText().startswith("3D")

    def _on_viz_mode_changed(self, mode: str):
        self._fill_viz_palette_for_mode(mode)
        self._refresh_tracks_plot()

    def _fill_viz_palette_for_mode(self, mode: str):
        self.viz_palette_combo.blockSignals(True)
        previous = self.viz_palette_combo.currentText()
        self.viz_palette_combo.clear()

        if mode == MODE_SOLID:
            self.viz_palette_combo.addItems(SOLID_COLOR_CHOICES)
            fallback = DEFAULT_SOLID if DEFAULT_SOLID in SOLID_COLOR_CHOICES else SOLID_COLOR_CHOICES[0]
        elif mode == MODE_TID:
            self.viz_palette_combo.addItem("—")
            fallback = "—"
        else:
            self.viz_palette_combo.addItems(LUT_CHOICES)
            fallback = DEFAULT_LUT if DEFAULT_LUT in LUT_CHOICES else LUT_CHOICES[0]

        target = previous if self.viz_palette_combo.findText(previous) >= 0 else fallback
        self.viz_palette_combo.setCurrentText(target)
        self.viz_palette_combo.setEnabled(mode != MODE_TID)
        self.viz_palette_combo.blockSignals(False)

    def _current_viz_settings(self) -> dict:
        mode = self.viz_mode_combo.currentText()
        if mode not in MODE_CHOICES:
            mode = DEFAULT_MODE

        palette = self.viz_palette_combo.currentText()
        lut = palette if palette in LUT_CHOICES else DEFAULT_LUT
        solid = palette if palette in SOLID_COLOR_CHOICES else DEFAULT_SOLID

        return {
            "mode": mode,
            "lut": lut,
            "solid": solid,
            "alpha": 0.9,
            "size": 7,
        }

    def _track_end_to_end_by_tid(self) -> dict:
        arr = self._current_arr
        if arr is None or len(arr) == 0 or arr.dtype.names is None:
            return {}
        if "tid" not in arr.dtype.names or "loc" not in arr.dtype.names:
            return {}

        tids_all = np.asarray(arr["tid"])
        locs = np.asarray(arr["loc"], dtype=float)
        if locs.ndim != 2 or locs.shape[1] < 2:
            return {}

        out = {}
        for tid in np.unique(tids_all):
            pts = locs[tids_all == tid]
            if len(pts) < 2:
                out[int(tid)] = 0.0
            else:
                d = pts[-1] - pts[0]
                out[int(tid)] = float(np.sqrt(np.sum(d * d)))
        return out

    def _refresh_tracks_plot(self):
        if self._tracks_df is None:
            return
        self._plot_tracks(self._tracks_df, labels=self._labels, title=self._current_plot_title())

    def _current_plot_title(self) -> str:
        if self._current_base is None:
            return "DBSCAN"
        if self._labels is None:
            return f"{self._current_base} - avg loc (tid)"
        eps = float(self.eps_spin.value())
        min_samples = int(self.min_samples_spin.value())
        return f"DBSCAN ({self._current_base}) - eps={eps}, min_samples={min_samples}"

    def _update_files_scroll_height(self, n_files: int):
        visible_rows = min(max(int(n_files), 1), 4)
        row_h = 38
        spacing = self.files_layout.spacing()
        margins = self.files_layout.contentsMargins()
        fixed_h = margins.top() + margins.bottom() + (visible_rows * row_h) + (max(0, visible_rows - 1) * spacing) + 4
        self.files_scroll.setFixedHeight(fixed_h)

    def rebuild(self, base_names):
        self._base_buttons.clear()

        while self.files_layout.count() > 0:
            item = self.files_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._btn_group = QtWidgets.QButtonGroup(self)
        self._btn_group.setExclusive(True)

        for base in base_names:
            btn = QtWidgets.QPushButton(base)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.setStyleSheet(
                """
                QPushButton {
                    text-align: center;
                    padding: 6px 10px;
                }
                QPushButton:checked {
                    background-color: #1F2123;
                    border: 1px solid #8AB4F8;
                    padding-top: 9px;
                    padding-left: 12px;
                }
                """
            )
            btn.clicked.connect(lambda checked, b=base: self.load_base(b) if checked else None)
            self._btn_group.addButton(btn)
            self.files_layout.addWidget(btn)
            self._base_buttons[base] = btn

        self.files_layout.addStretch(1)
        self._update_files_scroll_height(len(base_names))

        if self._current_base in self._base_buttons:
            self._base_buttons[self._current_base].setChecked(True)
            self.load_base(self._current_base)
        elif base_names:
            first = base_names[0]
            self._base_buttons[first].setChecked(True)
            self.load_base(first)
        else:
            self._current_base = None
            self._current_arr = None
            self._tracks_df = None
            self._labels = None
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text="No file loaded", x=0.5, y=0.5, showarrow=False)])
            apply_plot_theme(fig, is3d=False)
            self.view.update_fig(fig, reset_view=True, is3d=False)

    def _selected_arr(self, base: str):
        # Keep behavior aligned with multicolor viewer: refresh crop cache first,
        # then resolve through the same accessor.
        if getattr(self._mw, "_multicolor_crop_bounds", None):
            self._mw._recompute_multicolor_crop_cache()

        arr = self._mw._get_multicolor_arr_for_base(base)
        if arr is not None:
            return arr

        arr = self._mw._get_arr_for_base(base)
        if arr is not None:
            return arr

        if self._mw._current_ctx is not None and self._mw._current_ctx.get("base") == base:
            return self._mw._current_ctx.get("arr")

        return None

    def _avg_tracks_df(self, arr):
        if arr is None or len(arr) == 0:
            return pd.DataFrame(columns=["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations"])

        # Use the exact same averaging path as the multicolor viewer.
        xyz, _vals, tids_plot = scatter_points_and_color(arr, avg_tid=True)
        if xyz is None or len(xyz) == 0:
            return pd.DataFrame(columns=["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations"])

        xyz = np.asarray(xyz, dtype=float)
        tids_plot = np.asarray(tids_plot)
        if xyz.ndim != 2 or xyz.shape[1] < 2:
            return pd.DataFrame(columns=["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations"])

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2] if xyz.shape[1] >= 3 else np.zeros(len(x), dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[finite]
        y = y[finite]
        z = z[finite]
        tids_plot = tids_plot[finite]
        if len(tids_plot) == 0:
            return pd.DataFrame(columns=["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations"])

        # Track counts from the same array used to generate averages.
        nloc_by_tid = {}
        if "tid" in arr.dtype.names:
            tids_all = np.asarray(arr["tid"])
            ut, cnt = np.unique(tids_all, return_counts=True)
            nloc_by_tid = {int(t): int(c) for t, c in zip(ut, cnt)}

        n_localizations = np.array([nloc_by_tid.get(int(t), 0) for t in tids_plot], dtype=int)

        return pd.DataFrame(
            {
                "track_id": np.asarray(tids_plot).astype(int),
                "centroid_x": x.astype(float),
                "centroid_y": y.astype(float),
                "centroid_z": z.astype(float),
                "n_localizations": n_localizations,
            }
        )

    def _plot_tracks(self, tracks_df: pd.DataFrame, labels=None, title: str = None):
        fig = go.Figure()
        is3d = self._is_3d_mode()
        if tracks_df is None or len(tracks_df) == 0:
            fig.update_layout(annotations=[dict(text="No avg loc (tid) data", x=0.5, y=0.5, showarrow=False)])
            apply_plot_theme(fig, is3d=is3d)
            html = pio.to_html(
                fig,
                include_plotlyjs="cdn",
                full_html=False,
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "responsive": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "plot",
                        "scale": 10,
                    },
                },
            )
            self.view.web.setHtml(html, QUrl("about:blank"))
            return

        x = tracks_df["centroid_x"].to_numpy(dtype=float)
        y = tracks_df["centroid_y"].to_numpy(dtype=float)
        z = tracks_df["centroid_z"].to_numpy(dtype=float)
        tid = tracks_df["track_id"].to_numpy(dtype=int)
        nloc = tracks_df["n_localizations"].to_numpy(dtype=int)

        if labels is None:
            cs = self._current_viz_settings()
            mode = cs.get("mode", DEFAULT_MODE)
            marker = dict(size=7, opacity=0.9)

            if mode == MODE_SOLID:
                solid = cs.get("solid", DEFAULT_SOLID)
                marker["color"] = SOLID_COLOR_MAP.get(solid, SOLID_COLOR_MAP[DEFAULT_SOLID])
            elif mode == MODE_DEPTH:
                lut = cs.get("lut", DEFAULT_LUT)
                if lut not in LUT_CHOICES:
                    lut = DEFAULT_LUT
                marker["color"] = z
                marker["colorscale"] = CUSTOM_LUTS.get(lut, lut) if lut.startswith("cu") else lut
                marker["showscale"] = True
                marker["colorbar"] = dict(title="z (nm)")
            elif mode == MODE_E2E:
                lut = cs.get("lut", DEFAULT_LUT)
                if lut not in LUT_CHOICES:
                    lut = DEFAULT_LUT
                e2e_map = self._track_end_to_end_by_tid()
                cvals = np.array([e2e_map.get(int(t), 0.0) for t in tid], dtype=float)
                marker["color"] = cvals
                marker["colorscale"] = CUSTOM_LUTS.get(lut, lut) if lut.startswith("cu") else lut
                marker["showscale"] = True
                marker["colorbar"] = dict(title="end-to-end")
            else:  # MODE_TID
                marker["color"] = [tid_to_color(t, alpha=1.0) for t in tid]

            text = [f"track={int(t)}<br>n={int(n)}<br>z={zz:.2f}" for t, n, zz in zip(tid, nloc, z)]
            if is3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=marker,
                        text=text,
                        hovertemplate="%{text}<extra></extra>",
                        name="avg loc (tid)",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=marker,
                        text=text,
                        hovertemplate="%{text}<extra></extra>",
                        name="avg loc (tid)",
                        showlegend=False,
                    )
                )
        else:
            labels = np.asarray(labels, dtype=int)
            unique_labels = sorted(np.unique(labels))
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#4c78a8", "#f58518",
            ]
            for i, lab in enumerate(unique_labels):
                m = labels == lab
                if not np.any(m):
                    continue
                if lab == -1:
                    name = f"Noise ({int(np.sum(m))})"
                    color = "#9e9e9e"
                    opacity = 0.45
                else:
                    name = f"Cluster {int(lab)} ({int(np.sum(m))})"
                    color = palette[i % len(palette)]
                    opacity = 0.85
                text = [
                    f"cluster={int(lab)}<br>track={int(t)}<br>n={int(n)}<br>z={zz:.2f}"
                    for t, n, zz in zip(tid[m], nloc[m], z[m])
                ]
                if is3d:
                    fig.add_trace(
                        go.Scatter3d(
                            x=x[m],
                            y=y[m],
                            z=z[m],
                            mode="markers",
                            name=name,
                            marker=dict(size=5, color=color, opacity=opacity),
                            text=text,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=x[m],
                            y=y[m],
                            mode="markers",
                            name=name,
                            marker=dict(size=7, color=color, opacity=opacity),
                            text=text,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )

        finite = np.isfinite(x) & np.isfinite(y)
        xf = x[finite]
        yf = y[finite]
        zf = z[finite]
        if len(xf) > 0 and len(yf) > 0:
            xmin, xmax = float(np.min(xf)), float(np.max(xf))
            ymin, ymax = float(np.min(yf)), float(np.max(yf))
            dx = xmax - xmin
            dy = ymax - ymin
            pad_x = 0.05 * dx if dx > 0 else 5.0
            pad_y = 0.05 * dy if dy > 0 else 5.0
            if is3d:
                zmin, zmax = float(np.min(zf)), float(np.max(zf))
                dz = zmax - zmin
                pad_z = 0.05 * dz if dz > 0 else 5.0
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[xmin - pad_x, xmax + pad_x], autorange=False),
                        yaxis=dict(range=[ymin - pad_y, ymax + pad_y], autorange=False),
                        zaxis=dict(range=[zmin - pad_z, zmax + pad_z], autorange=False),
                        aspectmode="data",
                    )
                )
            else:
                fig.update_xaxes(range=[xmin - pad_x, xmax + pad_x], autorange=False)
                fig.update_yaxes(range=[ymin - pad_y, ymax + pad_y], autorange=False)

        # Diagnostics for render path
        n_traces = len(fig.data)
        n_points = int(len(x))
        self._append_log(f"- plotting traces: {n_traces}, points: {n_points}")

        fig.update_layout(
            title=title,
            xaxis_title="x (nm)",
            yaxis_title="y (nm)",
            legend_title="Label",
            dragmode="pan",
            showlegend=labels is not None,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
        )
        if is3d:
            fig.update_layout(
                scene=dict(
                    xaxis_title="x (nm)",
                    yaxis_title="y (nm)",
                    zaxis_title="z (nm)",
                )
            )
        apply_plot_theme(fig, is3d=is3d)
        fig.update_layout(
            title=dict(text=title, y=0.96, yanchor="top"),
            margin=dict(l=0, r=190, t=64, b=0),
        )
        if not is3d:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            add_scalebar_2d(fig, length_nm=float(getattr(self._mw, "_scalebar_nm", 100.0)))

        # Use plain Plotly HTML rendering in DBSCAN window to avoid sync/scalebar
        # side effects that can hide markers in this panel.
        html = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "plot",
                    "scale": 10,
                },
            },
        )
        self.view.web.setHtml(html, QUrl("about:blank"))

    def load_base(self, base: str):
        arr = self._selected_arr(base)
        self._current_base = base
        self._current_arr = arr
        self._labels = None

        if arr is None:
            self._tracks_df = pd.DataFrame(columns=["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations"])
            self._append_log(f"Loaded: {base}")
            self._append_log("No in-memory filtered/cropped data for this file yet. Run the file in main window first.")
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="No loaded data for selected file", x=0.5, y=0.5, showarrow=False)]
            )
            is3d = self._is_3d_mode()
            apply_plot_theme(fig, is3d=is3d)
            self.view.update_fig(fig, reset_view=True, is3d=is3d)
            return

        self._tracks_df = self._avg_tracks_df(arr)
        self._append_log(f"Loaded: {base}")

        n_locs = int(len(arr)) if arr is not None else 0
        n_tracks = int(len(self._tracks_df))
        self._append_log(f"Localizations: {n_locs}, avg loc tracks: {n_tracks}")
        self.run_diagnostics()

        self._plot_tracks(self._tracks_df, labels=None, title=f"{base} - avg loc (tid)")

    def run_diagnostics(self):
        if not self._current_base:
            self._append_log("Diagnostics: no file selected")
            return

        arr = self._current_arr
        self._append_log(f"Diagnostics for: {self._current_base}")
        if arr is None:
            self._append_log("- arr: None")
            return

        self._append_log(f"- arr length: {len(arr)}")
        names = list(arr.dtype.names) if arr.dtype.names is not None else []
        self._append_log(f"- fields: {names}")

        if "loc" in names:
            loc = np.asarray(arr["loc"])
            self._append_log(f"- loc shape: {tuple(loc.shape)}")
            try:
                lf = np.asarray(loc, dtype=float)
                if lf.ndim == 2 and lf.shape[1] >= 2:
                    fx = np.isfinite(lf[:, 0])
                    fy = np.isfinite(lf[:, 1])
                    fxy = fx & fy
                    self._append_log(f"- finite x/y: {int(np.sum(fxy))}/{len(lf)}")
            except Exception:
                pass

        try:
            xyz, _vals, tids_plot = scatter_points_and_color(arr, avg_tid=True)
            if xyz is None:
                self._append_log("- scatter_points_and_color(avg_tid=True): no data")
                return

            xyz = np.asarray(xyz, dtype=float)
            tids_plot = np.asarray(tids_plot)
            self._append_log(f"- avg xyz shape: {tuple(xyz.shape)}, tids: {len(tids_plot)}")

            if xyz.ndim == 2 and xyz.shape[1] >= 2 and len(xyz) > 0:
                x = xyz[:, 0]
                y = xyz[:, 1]
                z = xyz[:, 2] if xyz.shape[1] >= 3 else np.zeros(len(x), dtype=float)
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                self._append_log(f"- finite avg points: {int(np.sum(finite))}/{len(xyz)}")
                if np.any(finite):
                    xf = x[finite]
                    yf = y[finite]
                    zf = z[finite]
                    self._append_log(
                        f"- x range: [{float(np.min(xf)):.3f}, {float(np.max(xf)):.3f}], "
                        f"y range: [{float(np.min(yf)):.3f}, {float(np.max(yf)):.3f}], "
                        f"z range: [{float(np.min(zf)):.3f}, {float(np.max(zf)):.3f}]"
                    )
        except Exception as exc:
            self._append_log(f"- diagnostics error: {exc}")

    def refresh_current_plot(self):
        if not self._current_base:
            return
        self.load_base(self._current_base)

    def run_dbscan(self):
        if self._tracks_df is None or len(self._tracks_df) == 0:
            self._append_log("No data for DBSCAN.")
            return

        eps = float(self.eps_spin.value())
        min_samples = int(self.min_samples_spin.value())
        use_2d = not self._is_3d_mode()

        self._append_log("running dbscan...")
        if use_2d:
            points = self._tracks_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            dim_label = "2D"
        else:
            points = self._tracks_df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
            dim_label = "3D"

        labels = dbscan_numpy(points, eps=eps, min_samples=min_samples)
        self._labels = labels

        n_noise = int(np.sum(labels == -1))
        cluster_ids = sorted([int(cid) for cid in np.unique(labels) if int(cid) != -1])

        self._append_log(f"DBSCAN finished with eps={eps}, min_samples={min_samples}, mode={dim_label}")
        self._append_log(f"Clusters found: {len(cluster_ids)}")
        self._append_log(f"Noise points: {n_noise} / {len(labels)}")
        for cid in cluster_ids:
            self._append_log(f"  Cluster {cid}: {int(np.sum(labels == cid))} points")

        self._plot_tracks(
            self._tracks_df,
            labels=labels,
            title=f"DBSCAN ({self._current_base}) - eps={eps}, min_samples={min_samples}",
        )

    def save_dbscan(self):
        if not self._current_base or self._current_arr is None:
            QtWidgets.QMessageBox.warning(self, "Save DBSCAN", "Load a file first.")
            return
        if self._tracks_df is None or len(self._tracks_df) == 0:
            QtWidgets.QMessageBox.warning(self, "Save DBSCAN", "No avg loc (tid) data to save.")
            return
        if self._labels is None:
            QtWidgets.QMessageBox.warning(self, "Save DBSCAN", "Run DBSCAN first.")
            return

        save_folder = self._mw.out_edit.text().strip()
        if not save_folder:
            QtWidgets.QMessageBox.critical(self, "Save DBSCAN", "Please set an output folder in the main window.")
            return
        os.makedirs(save_folder, exist_ok=True)

        eps = float(self.eps_spin.value())
        min_samples = int(self.min_samples_spin.value())
        use_2d = self.dimension_combo.currentText().startswith("2D")
        dim = "2D" if use_2d else "3D"
        eps_token = f"{eps:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        stem = f"dbscan_{dim.lower()}_eps{eps_token}_min{min_samples}"
        base_prefix = f"{self._current_base}_{stem}"

        labels = np.asarray(self._labels, dtype=int)
        tracks = self._tracks_df.copy()
        tracks["cluster_id"] = labels

        tracks_out = tracks[["track_id", "centroid_x", "centroid_y", "centroid_z", "n_localizations", "cluster_id"]]
        tracks_path = os.path.join(save_folder, f"{base_prefix}_tracks_clustered.csv")
        tracks_out.to_csv(tracks_path, index=False)

        # propagate track labels back to all localizations
        arr = self._current_arr
        tid = np.asarray(arr["tid"]).astype(int) if "tid" in arr.dtype.names else np.array([], dtype=int)
        loc = np.asarray(arr["loc"], dtype=float) if "loc" in arr.dtype.names else np.zeros((0, 3), dtype=float)
        if loc.ndim != 2:
            loc = np.zeros((len(tid), 3), dtype=float)
        n = len(arr)

        x = loc[:, 0] if loc.shape[1] >= 1 else np.full(n, np.nan)
        y = loc[:, 1] if loc.shape[1] >= 2 else np.full(n, np.nan)
        z = loc[:, 2] if loc.shape[1] >= 3 else np.full(n, np.nan)

        if "tim" in arr.dtype.names:
            t = np.asarray(arr["tim"], dtype=float)
        else:
            t = np.arange(n, dtype=float)

        tid_to_cluster = {int(r.track_id): int(r.cluster_id) for r in tracks_out.itertuples(index=False)}
        loc_cluster = np.array([tid_to_cluster.get(int(tr), -1) for tr in tid], dtype=int)

        loc_df = pd.DataFrame(
            {
                "loc_id": np.arange(n, dtype=int),
                "x": x,
                "y": y,
                "z": z,
                "t": t,
                "track_id": tid,
                "cluster_id": loc_cluster,
            }
        )
        loc_path = os.path.join(save_folder, f"{base_prefix}_localizations_clustered.csv")
        loc_df.to_csv(loc_path, index=False)

        # cluster summary
        cl_rows = []
        for cid in sorted(int(c) for c in np.unique(labels) if int(c) >= 0):
            m = labels == cid
            trk = tracks_out.loc[m]
            if len(trk) == 0:
                continue

            cxyz = trk[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
            center = np.nanmean(cxyz, axis=0)
            d = np.sqrt(np.sum((cxyz - center[None, :]) ** 2, axis=1))
            radius = float(np.nanmean(d)) if len(d) else float("nan")

            cl_rows.append(
                {
                    "cluster_id": int(cid),
                    "n_tracks": int(len(trk)),
                    "n_localizations": int(np.sum(trk["n_localizations"].to_numpy(dtype=int))),
                    "centroid_x": float(center[0]),
                    "centroid_y": float(center[1]),
                    "centroid_z": float(center[2]),
                    "radius_estimate": radius,
                }
            )

        cluster_summary_df = pd.DataFrame(
            cl_rows,
            columns=[
                "cluster_id",
                "n_tracks",
                "n_localizations",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "radius_estimate",
            ],
        )
        summary_path = os.path.join(save_folder, f"{base_prefix}_cluster_summary.csv")
        cluster_summary_df.to_csv(summary_path, index=False)

        params = {
            "eps_nm": eps,
            "min_samples": min_samples,
            "dimension": dim,
            "clustering_level": "track_centroids",
            "centroid_definition": "mean_xyz",
            "source_file": self._current_base,
            "n_tracks": int(len(tracks_out)),
            "n_localizations": int(len(loc_df)),
        }
        params_path = os.path.join(save_folder, f"{base_prefix}_dbscan_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        self._append_log(f"Saved: {os.path.basename(tracks_path)}")
        self._append_log(f"Saved: {os.path.basename(loc_path)}")
        self._append_log(f"Saved: {os.path.basename(summary_path)}")
        self._append_log(f"Saved: {os.path.basename(params_path)}")
        self._append_log(f"Output folder: {save_folder}")

        QtWidgets.QMessageBox.information(self, "Save DBSCAN", "DBSCAN outputs saved.")

    def shutdown(self):
        try:
            if self.view is not None:
                self.view.shutdown()
        except Exception:
            pass


class PlotSyncBridge(QtCore.QObject):
    viewChanged = Signal(str, object)  # (source_id, payload dict)
    selectionChanged = Signal(str, object)  # (source_id, payload dict)

    @QtCore.Slot(str, "QVariant")
    def relayView(self, source_id, payload):
        # payload is a dict like {"mode":"2d","xRange":[...],"yRange":[...]} or {"mode":"3d","camera":{...}}
        self.viewChanged.emit(source_id, payload)

    @QtCore.Slot(str, "QVariant")
    def relaySelection(self, source_id, payload):
        # payload is a dict like {"mode":"2d","xRange":[...],"yRange":[...]}
        self.selectionChanged.emit(source_id, payload)


class MultiColorWindow(QtWidgets.QMainWindow):
    def __init__(self, on_view_changed, on_crop_requested=None, on_reset_crop_requested=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multicolor")
        self.resize(1200, 700)
        self._on_view_changed = on_view_changed
        self._on_crop_requested = on_crop_requested
        self._on_reset_crop_requested = on_reset_crop_requested
        self._last_selection_payload = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        # --- stacked widget with plots (create ONCE) ---
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)

        # --- crop buttons overlaid bottom-right ---
        self._crop_btn = QtWidgets.QPushButton("Crop", central)
        self._crop_btn.clicked.connect(self._crop_current_selection)
        self._crop_btn.setFixedHeight(24)
        self._crop_btn.setMaximumWidth(90)
        self._crop_btn.raise_()

        self._reset_crop_btn = QtWidgets.QPushButton("Reset Crop", central)
        self._reset_crop_btn.clicked.connect(self._reset_crop)
        self._reset_crop_btn.setFixedHeight(24)
        self._reset_crop_btn.setMaximumWidth(110)
        self._reset_crop_btn.raise_()

        # --- grid page ---
        self.grid_page = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_page)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.stack.addWidget(self.grid_page)

        # --- merged page ---
        self.merged_page = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(self.merged_page)
        mv.setContentsMargins(0, 0, 0, 0)

        self.merged_view = PlotlyView()
        self.merged_view.viewChanged.connect(self._on_view_changed)
        self.merged_view.selectionChanged.connect(self._on_plot_selection)
        mv.addWidget(self.merged_view, 1)

        self.stack.addWidget(self.merged_page)

        # state
        self._views = {}    # base -> PlotlyView
        self._mode = "grid" # or "merged"

    def set_mode(self, mode: str):
        mode = "merged" if mode == "merged" else "grid"
        self._mode = mode
        self.stack.setCurrentIndex(1 if mode == "merged" else 0)

    def mode(self):
        return self._mode

    def rebuild(self, base_names):
        # clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._views.clear()

        n = max(1, len(base_names))
        cols = 2 if n <= 4 else 3
        r = c = 0

        for base in base_names:
            box = QtWidgets.QGroupBox(base)
            vbox = QtWidgets.QVBoxLayout(box)
            vbox.setContentsMargins(4, 8, 4, 4)

            view = PlotlyView()
            view.viewChanged.connect(self._on_view_changed)
            view.selectionChanged.connect(self._on_plot_selection)
            vbox.addWidget(view, 1)

            self._views[base] = view
            self.grid.addWidget(box, r, c)

            c += 1
            if c >= cols:
                c = 0
                r += 1

    def update_one(self, base, fig, reset_view=False, is3d=False, scalebar_nm=100.0):
        view = self._views.get(base)
        if view is None:
            return
        if (not is3d) and fig is not None:
            add_scalebar_2d(fig, length_nm=float(scalebar_nm))
        view.update_fig(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def update_all(self, figs_by_base, reset_view=False, is3d=False, scalebar_nm=100.0):
        for base, fig in figs_by_base.items():
            self.update_one(base, fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (
            hasattr(self, "_crop_btn") and self._crop_btn is not None
            and hasattr(self, "_reset_crop_btn") and self._reset_crop_btn is not None
        ):
            m = 12  # margin from edges
            gap = 8
            btn_crop = self._crop_btn
            btn_reset = self._reset_crop_btn
            btn_crop.adjustSize()
            btn_reset.adjustSize()
            y = self.centralWidget().height() - btn_reset.height() - m
            x_reset = self.centralWidget().width() - btn_reset.width() - m
            x_crop = x_reset - btn_crop.width() - gap
            btn_crop.move(x_crop, y)
            btn_reset.move(x_reset, y)

    def _on_plot_selection(self, _source_id, payload):
        if not isinstance(payload, dict):
            return
        if payload.get("mode") != "2d":
            return
        xr = payload.get("xRange")
        yr = payload.get("yRange")
        if not (isinstance(xr, (list, tuple)) and isinstance(yr, (list, tuple)) and len(xr) == 2 and len(yr) == 2):
            return
        self._last_selection_payload = payload

    def _crop_current_selection(self):
        if self._on_crop_requested is None:
            return
        p = self._last_selection_payload
        if not isinstance(p, dict):
            QtWidgets.QMessageBox.information(self, "Crop", "Use Box select on a multicolor plot first.")
            return
        xr = p.get("xRange")
        yr = p.get("yRange")
        if not (isinstance(xr, (list, tuple)) and isinstance(yr, (list, tuple)) and len(xr) == 2 and len(yr) == 2):
            QtWidgets.QMessageBox.information(self, "Crop", "Box selection range is unavailable.")
            return
        try:
            payload = {
                "mode": "2d",
                "xRange": [float(min(xr[0], xr[1])), float(max(xr[0], xr[1]))],
                "yRange": [float(min(yr[0], yr[1])), float(max(yr[0], yr[1]))],
            }
        except Exception:
            QtWidgets.QMessageBox.information(self, "Crop", "Invalid box selection values.")
            return
        self._on_crop_requested(payload)

    def _reset_crop(self):
        if self._on_reset_crop_requested is None:
            return
        self._on_reset_crop_requested()

    def grid_sync_enabled(self) -> bool:
        return False

    def update_merged(self, fig, reset_view=False, is3d=False, scalebar_nm=100.0):
        self.merged_view.update_fig(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def shutdown(self):
        try:
            if self.merged_view is not None:
                self.merged_view.shutdown()
        except Exception:
            pass

        for view in self._views.values():
            try:
                view.shutdown()
            except Exception:
                pass

class PlotlyView(QtWidgets.QWidget):
    """Widget hosting Plotly in a QWebEngineView + emits view changes."""
    viewChanged = Signal(str, object)  # (source_id, payload)
    selectionChanged = Signal(str, object)  # (source_id, payload)

    def __init__(self, parent=None):
        super().__init__(parent)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.web = QWebEngineView(self)
        lay.addWidget(self.web)

        s = self.web.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self._plotly_loaded = False
        self._plotly_ready = False
        self._pending_fig = None
        self._id = hex(id(self))

        # WebChannel bridge
        self._bridge = PlotSyncBridge()
        self._bridge.viewChanged.connect(self.viewChanged.emit)
        self._bridge.selectionChanged.connect(self.selectionChanged.emit)

        self._channel = QWebChannel(self.web.page())
        self._channel.registerObject("plotSync", self._bridge)
        self.web.page().setWebChannel(self._channel)

        self.web.page().loadFinished.connect(self._on_load_finished)

        self._last_fig = None

        self.web.page().setBackgroundColor(QColor(PLOTLY_HTML_BG))
        self.ensure_page()

    def _on_download_requested(self, download: "QtWebEngineCore.QWebEngineDownloadRequest"):
        # Suggest a filename
        suggested = download.suggestedFileName() or "plot.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            suggested,
            "PNG Image (*.png);;All Files (*)"
        )
        if not path:
            download.cancel()
            return

        # Ensure .png extension if user omitted it
        if not os.path.splitext(path)[1]:
            path += ".png"

        download.setDownloadDirectory(os.path.dirname(path))
        download.setDownloadFileName(os.path.basename(path))
        download.accept()

    def _bootstrap_html(self):
        bg = PLOTLY_HTML_BG

        # scalebar constants
        L = float(SCALEBAR_LENGTH_NM)
        mfrac = float(SCALEBAR_MARGIN_FRACTION)
        sb_color = str(SCALEBAR_COLOR)
        sb_lw = int(SCALEBAR_LINE_WIDTH)
        sb_label = f"{int(round(L))} nm"

        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
    html, body {{
    margin:0; padding:0;
    width:100%; height:100%;
    overflow:hidden;
    background: {bg};
    }}
    #container {{
    width:100%; height:100%;
    background: {bg};
    }}
    #plot {{
    width:100%; height:100%;
    background: {bg};
    }}
    </style>
    </head>
    <body>
    <div id="container"><div id="plot"></div></div>

    <script>
    window._plotSync = {{ ready:false, bridge:null, sourceId:null, ignore:false, _scalebarUpdating:false, scalebarNm: 100.0 }};

    new QWebChannel(qt.webChannelTransport, function(channel) {{
    window._plotSync.bridge = channel.objects.plotSync;
    window._plotSync.ready = true;
    }});

    function updateScaleBar2D(gd) {{
    try {{
        if (!gd || !gd._fullLayout) return;
        if (gd._fullLayout.scene) return; // only 2D

        const xa = gd._fullLayout.xaxis;
        const ya = gd._fullLayout.yaxis;
        if (!xa || !ya) return;

        let xMin = null, xMax = null, yMin = null, yMax = null;
        if (xa.range && ya.range && xa.range.length >= 2 && ya.range.length >= 2) {{
            xMin = Number(xa.range[0]);
            xMax = Number(xa.range[1]);
            yMin = Number(ya.range[0]);
            yMax = Number(ya.range[1]);
        }}

        // Fallback for cases where axis ranges are not initialized yet.
        if (![xMin, xMax, yMin, yMax].every(v => Number.isFinite(v))) {{
            const xs = [];
            const ys = [];
            const data = (gd.data || []);
            for (const tr of data) {{
                if (Array.isArray(tr.x)) {{
                    for (const v of tr.x) {{
                        const n = Number(v);
                        if (Number.isFinite(n)) xs.push(n);
                    }}
                }}
                if (Array.isArray(tr.y)) {{
                    for (const v of tr.y) {{
                        const n = Number(v);
                        if (Number.isFinite(n)) ys.push(n);
                    }}
                }}
            }}
            if (!xs.length || !ys.length) return;
            xMin = Math.min(...xs);
            xMax = Math.max(...xs);
            yMin = Math.min(...ys);
            yMax = Math.max(...ys);
        }}

        const dx = xMax - xMin;
        const dy = yMax - yMin;
        if (!isFinite(dx) || !isFinite(dy) || dx === 0 || dy === 0) return;

        const L = (window._plotSync && typeof window._plotSync.scalebarNm === "number")
          ? window._plotSync.scalebarNm
          : {L};
        const m = {mfrac};

        const x0 = xMin + m * dx;
        const y0 = yMin + m * dy;
        const x1 = x0 + L;

        const sbName = "scalebar";
        const sbLabelName = "scalebar_label";

        const shapes = (gd.layout && gd.layout.shapes) ? gd.layout.shapes.slice() : [];
        const anns = (gd.layout && gd.layout.annotations) ? gd.layout.annotations.slice() : [];

        const shapes2 = shapes.filter(s => !(s && s.name === sbName));
        const anns2 = anns.filter(a => !(a && a.name === sbLabelName));

        shapes2.push({{
        type: "line",
        xref: "x",
        yref: "y",
        x0: x0, y0: y0,
        x1: x1, y1: y0,
        line: {{ color: "{sb_color}", width: {sb_lw} }},
        name: sbName
        }});

        anns2.push({{
        x: (x0 + x1) / 2.0,
        y: y0 + 0.03 * dy,
        xref: "x",
        yref: "y",
        text: Math.round(L).toString() + " nm",
        showarrow: false,
        font: {{ color: "{sb_color}", size: 12 }},
        name: sbLabelName
        }});

        window._plotSync._scalebarUpdating = true;
        Plotly.relayout(gd, {{ shapes: shapes2, annotations: anns2 }})
        .then(() => {{ window._plotSync._scalebarUpdating = false; }})
        .catch(() => {{ window._plotSync._scalebarUpdating = false; }});

    }} catch (e) {{
        console.log("updateScaleBar2D error", e);
    }}
    }}

    function installRelayoutHandler() {{
    const gd = document.getElementById('plot');
    if (!gd) return;

    if (gd.removeAllListeners) gd.removeAllListeners('plotly_relayout');

    let _lastSent = 0;
    let _timer = null;
    let _pending = null;

    function buildPayload(gd) {{
        const is3d = !!(gd._fullLayout && gd._fullLayout.scene);
        const payload = {{ mode: is3d ? "3d" : "2d" }};

        if (is3d) {{
        const cam = gd._fullLayout.scene && gd._fullLayout.scene.camera;
        if (cam) payload.camera = cam;
        }} else {{
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
        if (xa && xa.range && ya && ya.range) {{
            payload.xRange = xa.range;
            payload.yRange = ya.range;
        }}
        }}
        return payload;
    }}

    function maybeSend() {{
        _timer = null;
        if (!_pending) return;

        if (window._plotSync.ignore || window._plotSync._scalebarUpdating) {{
        _pending = null;
        return;
        }}

        const payload = _pending;
        _pending = null;

        if (window._plotSync.ready && window._plotSync.bridge) {{
        window._plotSync.bridge.relayView(window._plotSync.sourceId || "unknown", payload);
        }}
    }}

    gd.on('plotly_relayout', function(e) {{
        try {{
        // ALWAYS update scalebar for user pan/zoom (skip only if we are updating it ourselves)
        if (!window._plotSync._scalebarUpdating) {{
            updateScaleBar2D(gd);
        }}

        // broadcasting part
        if (window._plotSync.ignore || window._plotSync._scalebarUpdating) return;

        const now = Date.now();
        _pending = buildPayload(gd);

        const minInterval = 40;
        const dt = now - _lastSent;

        if (dt >= minInterval) {{
            _lastSent = now;
            maybeSend();
        }} else {{
            if (_timer) clearTimeout(_timer);
            _timer = setTimeout(() => {{
            _lastSent = Date.now();
            maybeSend();
            }}, minInterval - dt);
        }}
        }} catch(err) {{
        console.log("relayout hook error", err);
        }}
    }});
    }}

    function installSelectionHandler() {{
    const gd = document.getElementById('plot');
    if (!gd) return;

    if (gd.removeAllListeners) gd.removeAllListeners('plotly_selected');

    gd.on('plotly_selected', function(ev) {{
        try {{
        if (window._plotSync.ignore || !window._plotSync.ready || !window._plotSync.bridge) return;
        if (!gd._fullLayout || gd._fullLayout.scene) return; // 2D only

        let xr = null;
        let yr = null;

        if (ev && ev.range && ev.range.x && ev.range.y) {{
            xr = ev.range.x;
            yr = ev.range.y;
        }} else if (ev && ev.points && ev.points.length) {{
            const xs = ev.points.map(p => Number(p.x)).filter(v => Number.isFinite(v));
            const ys = ev.points.map(p => Number(p.y)).filter(v => Number.isFinite(v));
            if (xs.length && ys.length) {{
                xr = [Math.min(...xs), Math.max(...xs)];
                yr = [Math.min(...ys), Math.max(...ys)];
            }}
        }}

        if (!xr || !yr || xr.length < 2 || yr.length < 2) return;

        const payload = {{
            mode: "2d",
            xRange: [Number(xr[0]), Number(xr[1])],
            yRange: [Number(yr[0]), Number(yr[1])]
        }};
        window._plotSync.bridge.relaySelection(window._plotSync.sourceId || "unknown", payload);
        }} catch(err) {{
        console.log("selection hook error", err);
        }}
    }});
    }}
    </script>
    </body>
    </html>"""
        bg = PLOTLY_HTML_BG  # uses your constant
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
html, body {{
  margin:0; padding:0;
  width:100%; height:100%;
  overflow:hidden;
  background: {bg};
}}
#container {{
  width:100%; height:100%;
  background: {bg};
}}
#plot {{
  width:100%; height:100%;
  background: {bg};
}}
</style>
</head>
<body>
<div id="container"><div id="plot"></div></div>

<script>
window._plotSync = {{ ready:false, bridge:null, sourceId:null, ignore:false }};

new QWebChannel(qt.webChannelTransport, function(channel) {{
  window._plotSync.bridge = channel.objects.plotSync;
  window._plotSync.ready = true;
}});

function installRelayoutHandler() {{
  const gd = document.getElementById('plot');
  if (!gd) return;

  if (gd.removeAllListeners) gd.removeAllListeners('plotly_relayout');

  let _lastSent = 0;
  let _timer = null;
  let _pending = null;

  function buildPayload(gd) {{
    const is3d = !!(gd._fullLayout && gd._fullLayout.scene);
    const payload = {{ mode: is3d ? "3d" : "2d" }};

    if (is3d) {{
      const cam = gd._fullLayout.scene && gd._fullLayout.scene.camera;
      if (cam) payload.camera = cam;
    }} else {{
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      if (xa && xa.range && ya && ya.range) {{
        payload.xRange = xa.range;
        payload.yRange = ya.range;
      }}
    }}
    return payload;
  }}

  function maybeSend() {{
    _timer = null;
    if (!_pending) return;
    if (window._plotSync.ignore) {{ _pending = null; return; }}

    const payload = _pending;
    _pending = null;

    if (window._plotSync.ready && window._plotSync.bridge) {{
      window._plotSync.bridge.relayView(window._plotSync.sourceId || "unknown", payload);
    }}
  }}

    gd.on('plotly_relayout', function(e) {{
    try {{
        // Always keep scalebar in bottom-left of current view.
        // Skip if this relayout is the one we triggered to set the scalebar.
        if (!window._plotSync._scalebarUpdating) {{
        updateScaleBar2D(gd);
        }}

        // Everything below is only for broadcasting view sync
        if (window._plotSync.ignore || window._plotSync._scalebarUpdating) return;

        const now = Date.now();
        _pending = buildPayload(gd);

        const minInterval = 40;
        const dt = now - _lastSent;

        if (dt >= minInterval) {{
        _lastSent = now;
        maybeSend();
        }} else {{
        if (_timer) clearTimeout(_timer);
        _timer = setTimeout(() => {{
            _lastSent = Date.now();
            maybeSend();
        }}, minInterval - dt);
        }}
    }} catch(err) {{
        console.log("relayout hook error", err);
    }}
    }});
}}
</script>
</body>
</html>"""

    def ensure_page(self):
        if self._plotly_loaded:
            return
        self._plotly_loaded = True
        self._plotly_ready = False
        self.web.setHtml(self._bootstrap_html(), QUrl("about:blank"))

    def _on_load_finished(self, ok: bool):
        self._plotly_ready = bool(ok)
        if ok and self._pending_fig is not None:
            fig, reset_view, is3d = self._pending_fig
            self._pending_fig = None
            self.update_fig(fig, reset_view=reset_view, is3d=is3d)

    def update_fig(self, fig, reset_view=False, is3d=False, scalebar_nm=None):
        self.ensure_page()
        if not self._plotly_ready:
            self._pending_fig = (fig, reset_view, is3d)
            return
        self._last_fig = fig
        fig_json = pio.to_json(fig, validate=False)
        source_id = self._id
        sb = 100.0 if scalebar_nm is None else float(scalebar_nm)
        js = f"""
    (async function() {{
    try {{
        const fig = {fig_json};
        const targetIs3D = {'true' if is3d else 'false'};
        const resetView = {'true' if reset_view else 'false'};

        const container = document.getElementById('container');
        let gd = document.getElementById('plot');
        if (!container || !gd) return "error: container missing";

        // --- capture current view from existing plot (if any) ---
        let saved = null;
        if (!resetView && gd._fullLayout) {{
        const is3dNow = !!(gd._fullLayout.scene);
        if (is3dNow) {{
            const cam = gd._fullLayout.scene && gd._fullLayout.scene.camera;
            if (cam) saved = {{ mode: "3d", camera: cam }};
        }} else {{
            const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
            if (xa && xa.range && ya && ya.range) {{
            saved = {{ mode: "2d", xRange: xa.range, yRange: ya.range }};
            }}
        }}
        }}

        // Decide if we must rebuild (switch 2d<->3d or no plot yet)
        const currently3D = !!(gd._fullLayout && gd._fullLayout.scene);
        const modeChanged = (targetIs3D !== currently3D);

        const config = {{
            scrollZoom: true,
            displaylogo: false,
            responsive: true,
            toImageButtonOptions: {{
                format: "png",
                filename: "plot",
                scale: 10
            }}
        }};

        // Prevent relayout events from being broadcast while we update
        window._plotSync.sourceId = "{source_id}";
        window._plotSync.ignore = true;
        window._plotSync.scalebarNm = {sb};
        if (modeChanged || !gd.data) {{
        try {{ Plotly.purge(gd); }} catch(e) {{}}
        container.innerHTML = '<div id="plot" style="width:100%;height:100%;"></div>';
        gd = document.getElementById('plot');
        }}

        // Ensure layout objects exist
        fig.layout = fig.layout || {{}};

        // --- re-apply saved view onto new layout (if compatible) ---
        if (saved && saved.mode === "2d" && !targetIs3D) {{
        fig.layout.xaxis = fig.layout.xaxis || {{}};
        fig.layout.yaxis = fig.layout.yaxis || {{}};
        fig.layout.xaxis.range = saved.xRange;
        fig.layout.yaxis.range = saved.yRange;
        fig.layout.xaxis.autorange = false;
        fig.layout.yaxis.autorange = false;
        }} else if (saved && saved.mode === "3d" && targetIs3D) {{
        fig.layout.scene = fig.layout.scene || {{}};
        fig.layout.scene.camera = saved.camera;
        }} else {{
        // If no saved view or incompatible (2d<->3d switch), allow autorange
        if (targetIs3D) {{
            fig.layout.scene = fig.layout.scene || {{}};
            fig.layout.scene.xaxis = fig.layout.scene.xaxis || {{}};
            fig.layout.scene.yaxis = fig.layout.scene.yaxis || {{}};
            fig.layout.scene.zaxis = fig.layout.scene.zaxis || {{}};
            fig.layout.scene.xaxis.autorange = true;
            fig.layout.scene.yaxis.autorange = true;
            fig.layout.scene.zaxis.autorange = true;
        }} else {{
            fig.layout.xaxis = fig.layout.xaxis || {{}};
            fig.layout.yaxis = fig.layout.yaxis || {{}};
            fig.layout.xaxis.autorange = true;
            fig.layout.yaxis.autorange = true;
        }}
        }}

        // Plot
        if (modeChanged || !gd.data) {{
        await Plotly.newPlot(gd, fig.data, fig.layout, config);
        }} else {{
        await Plotly.react(gd, fig.data, fig.layout, config);
        }}

        // ensure scalebar placed for current initial view
        updateScaleBar2D(gd);

        // Install relayout handler after plot exists
        installRelayoutHandler();
        installSelectionHandler();

        // Re-enable broadcasting (next tick so Plotly internal relayouts won't echo)
        setTimeout(() => {{ window._plotSync.ignore = false; }}, 60);

        return "ok";
    }} catch(e) {{
        console.error(e);
        try {{ window._plotSync.ignore = false; }} catch(_) {{}}
        return "error: " + e.toString();
    }}
    }})();
    """
        self.web.page().runJavaScript(js)

    def apply_view(self, payload: dict):
        """Apply external view state to this plot (ranges/camera) without re-broadcast."""
        self.ensure_page()
        if not self._plotly_ready:
            return

        payload_json = json.dumps(payload)
        js = f"""
(function() {{
  const gd = document.getElementById('plot');
  if (!gd || !gd._fullLayout) return "no gd";

  const p = {payload_json};
  window._plotSync.ignore = true;
  try {{
    if (p.mode === "2d" && p.xRange && p.yRange) {{
      Plotly.relayout(gd, {{
        "xaxis.range": p.xRange,
        "yaxis.range": p.yRange,
        "xaxis.autorange": false,
        "yaxis.autorange": false
      }});
    }} else if (p.mode === "3d" && p.camera) {{
      Plotly.relayout(gd, {{
        "scene.camera": p.camera
      }});
    }}
  }} finally {{
    const delay = (p.mode === "3d") ? 50 : 10;
    setTimeout(() => {{ window._plotSync.ignore = false; }}, delay);
  }}
  return "applied";
}})();
"""
        self.web.page().runJavaScript(js)
    
    def shutdown(self):
        try:
            if self.web is not None:
                page = self.web.page()
                self.web.setParent(None)
                self.web.deleteLater()
                if page is not None:
                    page.deleteLater()
        except Exception:
            pass



# -------------------- worker --------------------
class FileWorker(QtCore.QThread):
    need_efo = Signal(object)           # ctx dict -> GUI should show histogram + scatter and let user choose
    status = Signal(str)
    done_one = Signal(object)           # result dict (final stats row items), None if skipped

    def __init__(self, file_path, params, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.params = params
        self._chosen_range = None
        self._cancelled = False
        self._wait = QtCore.QWaitCondition()
        self._mutex = QtCore.QMutex()
    
    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("cancelled")
        
    def cancel(self):
        self.requestInterruption()
        QtCore.QMutexLocker(self._mutex)
        self._cancelled = True
        self._chosen_range = None
        self._wait.wakeAll()

    def set_range_and_continue(self, xmin, xmax):
        self._mutex.lock()
        self._chosen_range = (float(xmin), float(xmax))
        self._wait.wakeAll()
        self._mutex.unlock()

    def run(self):
        try:
            res = self._process_one_file()
            self.done_one.emit(res)
        except Exception as e:
            self.status.emit(f"ERROR: {e}")
            self.done_one.emit(None)


    def _process_one_file(self):
        p = self.params
        file = self.file_path
        base = os.path.splitext(os.path.basename(file))[0]
        self.status.emit(f"Loading: {base}")

        MFX_Data = np.load(file, allow_pickle=False)
        MFX_Data = MFX_Data.copy()
        if self.isInterruptionRequested():
            return None
        self._check_cancel()

        total_tim = MFX_Data["tim"][-1] - MFX_Data["tim"][0]
        total_loc = len(MFX_Data)

        MFX_Data = MFX_Data[MFX_Data["vld"] == True]
        if self.isInterruptionRequested():
            return None
        self._check_cancel()
        if len(MFX_Data) == 0:
            self.status.emit(f"Skipping {base}: no valid localizations.")
            return None

        MFX_Data["loc"][:, -1] *= p["z_corr"]
        MFX_Data["loc"] *= p["scale"]
        MFX_Data["loc"][:, 1] *= -1.0

        last_itr = int(np.max(MFX_Data["itr"]))
        MFX_Data_vld_fnl = MFX_Data[MFX_Data["itr"] == last_itr].copy()
        last_iteration_loc = len(MFX_Data_vld_fnl)
        self._check_cancel()

        unique_tids, inv_idx, locs_per_tid = np.unique(
            MFX_Data_vld_fnl["tid"], return_inverse=True, return_counts=True
        )
        MFX_Data_vld_fnl_filt = MFX_Data_vld_fnl[locs_per_tid[inv_idx] >= p["min_trace_len"]]
        after_trace = len(MFX_Data_vld_fnl_filt)
        if after_trace == 0:
            self.status.emit(f"Skipping {base}: no data after trace filtering.")
            return None

        if "cfr" not in MFX_Data.dtype.names:
            self.status.emit(f"Skipping {base}: no CFR field.")
            return None

        # last valid CFR iteration = last itr with std(cfr) > 0
        iters = np.sort(np.unique(MFX_Data["itr"]))
        valid_cfr_iters = [it for it in iters if np.std(MFX_Data["cfr"][MFX_Data["itr"] == it]) > 0.0]
        last_cfr_itr = int(valid_cfr_iters[-1]) if len(valid_cfr_iters) else last_itr

        max_itr = np.max(MFX_Data_vld_fnl_filt["itr"])
        efo_vals = MFX_Data_vld_fnl_filt["efo"][MFX_Data_vld_fnl_filt["itr"] == max_itr]

        # get CFR for final tids from last valid CFR iteration
        final_tids = pd.DataFrame({"tid": MFX_Data_vld_fnl_filt["tid"]})
        cfr_df = pd.DataFrame({
            "tid": MFX_Data[MFX_Data["itr"] == last_cfr_itr]["tid"],
            "cfr": MFX_Data[MFX_Data["itr"] == last_cfr_itr]["cfr"],
        })
        merged = final_tids.merge(cfr_df, on="tid", how="left")
        cfr_vals = merged["cfr"].dropna().to_numpy()

        # optionally also overwrite cfr in the returned array for downstream filtering/saving
        if len(merged) == len(MFX_Data_vld_fnl_filt):
            cfr_arr = merged["cfr"].to_numpy()
            valid = np.isfinite(cfr_arr)
            MFX_Data_vld_fnl_filt = MFX_Data_vld_fnl_filt[valid].copy()
            MFX_Data_vld_fnl_filt["cfr"] = cfr_arr[valid]
            efo_vals = efo_vals[valid]
            cfr_vals = cfr_arr[valid]

        self._check_cancel()
        if len(efo_vals) == 0:
            self.status.emit(f"Skipping {base}: no EFO values.")
            return None
        if len(cfr_vals) == 0:
            self.status.emit(f"Skipping {base}: no CFR values.")
            return None

        ctx = dict(
            base=base,
            arr=MFX_Data_vld_fnl_filt,
            efo_vals=efo_vals,
            cfr_vals=cfr_vals,
            total_tim=total_tim,
            total_loc=total_loc,
            last_iteration_loc=last_iteration_loc,
            after_trace=after_trace,
        )
        self.need_efo.emit(ctx)
        return dict(display_name=base, ctx=ctx)

# -------------------- main window --------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MINFLUX Filter GUI (PyQt + Matplotlib + Plotly)")
        self.resize(1500, 950)
        self._plotly_tmp = None
        # state
        self._all_files = []
        self._current_index = 0
        self._current_ctx = None
        self._current_worker = None
        self._last_span = (None, None)
        self._current_is_3d = False  # Track current plot mode
        self._arr_by_base = {}        # base -> currently active array for plotting (trace-filtered or EFO-filtered)
        self._base_by_file = {}       # file_path -> base (optional convenience)
        self._multicolor_win = None
        self._multicolor_crop_bounds = None  # (xmin, xmax, ymin, ymax) for multicolor-only crop
        self._multicolor_cropped_by_base = {}  # base -> cropped array used only in multicolor viewer

        self._ctx_by_base = {}          # base -> ctx (from worker)
        self._efo_range_by_base = {}    # base -> (xmin, xmax) selected
        self._cfr_range_by_base = {}    # base -> (xmin, xmax) selected
        self._filtered_by_base = {}     # base -> array after EFO+CFR filtering (what will be saved)

        # widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self._binning_win = None
        self._dbscan_win = None

        # ---- top area: controls + output ----
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, 3)

        left = QtWidgets.QVBoxLayout()
        top.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        top.addLayout(right, 2)

        self._plot_arr = None          # array currently shown in scatter (may be EFO-filtered)
        self._plot_is_filtered = False

        # folder rows
        self.data_edit = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_folder)
        setdef_btn = QtWidgets.QPushButton("Set default")
        setdef_btn.clicked.connect(self.set_default_output)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("MINFLUX data folder:"), 0, 0)
        grid.addWidget(self.data_edit, 0, 1)
        grid.addWidget(browse_btn, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Output folder:"), 1, 0)
        grid.addWidget(self.out_edit, 1, 1)
        grid.addWidget(setdef_btn, 1, 2)
        left.addLayout(grid)

        self._closing = False
        self._plotly_loaded = False
        self._pending_fig = None

        # base -> dict(mode="solid"/"end-to-end", solid="cyan"/..., lut="Turbo"/...)
        self._color_settings_by_base = {}

        self._scalebar_nm = 100.0

        # params
        # ---- parameters (stored on MainWindow, edited via dialog) ----
        self.min_trace = QtWidgets.QSpinBox()
        self.min_trace.setRange(1, 9999)
        self.min_trace.setValue(3)

        self.zcorr = QtWidgets.QDoubleSpinBox()
        self.zcorr.setDecimals(6)
        self.zcorr.setRange(0, 1000)
        self.zcorr.setValue(0.7)

        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setDecimals(1)
        self.scale.setRange(1, 1e15)
        self.scale.setValue(1e9)

        self.bin_size = QtWidgets.QDoubleSpinBox()
        self.bin_size.setDecimals(1)
        self.bin_size.setRange(1, 1e9)
        self.bin_size.setValue(3000)

        self.cfr_bin_count = QtWidgets.QSpinBox()
        self.cfr_bin_count.setRange(5, 500)
        self.cfr_bin_count.setValue(50)

        self._workers = set()

        # run row
        runrow = QtWidgets.QHBoxLayout()
        left.addLayout(runrow)

        self.params_btn = QtWidgets.QPushButton("Parameters")
        self.params_btn.clicked.connect(self.open_parameters_dialog)
        runrow.addWidget(self.params_btn)

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self.run_start)
        runrow.addWidget(self.run_btn)

        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        runrow.addWidget(self.file_combo, 1)

        # --- color settings panel + right-side controls row (under Parameters/Run) ---
        color_and_controls = QtWidgets.QHBoxLayout()
        left.addLayout(color_and_controls)

        # Left: embedded color settings (scrollable)
        self.color_panel = ColorSettingsPanel(parent=self)
        self.color_panel.changed.connect(self.on_color_settings_changed)
        color_and_controls.addWidget(self.color_panel, 5)   # change here the width of the color settings panel

        # Right: stack the other controls vertically
        right_controls = QtWidgets.QVBoxLayout()
        color_and_controls.addLayout(right_controls, 1)
        right_controls.setContentsMargins(0, 18, 0, 0)  # push down; tweak 18->24 etc

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.avg_tid = QtWidgets.QCheckBox("avg loc (tid)")
        self.avg_tid.stateChanged.connect(lambda _: self._refresh_all_plots_same_data())
        row.addWidget(self.avg_tid, 1)

        self.is3d = QtWidgets.QCheckBox("3D")
        self.is3d.stateChanged.connect(lambda _: self._refresh_all_plots_same_data())
        row.addWidget(self.is3d, 0)

        right_controls.addLayout(row)

        # --- Binning + DBSCAN buttons (above multicolor section) ---
        VIEWER_W = 150
        BTN_H = 34

        def tune_btn(b, w):
            b.setFixedHeight(BTN_H)
            b.setFixedWidth(w)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)

        half_w = (VIEWER_W - btn_row.spacing()) // 2

        self.binning_btn = QtWidgets.QPushButton("Binning")
        self.binning_btn.clicked.connect(self.open_binning_window)
        tune_btn(self.binning_btn, half_w)
        btn_row.addWidget(self.binning_btn)

        self.dbscan_btn = QtWidgets.QPushButton("DBSCAN")
        self.dbscan_btn.clicked.connect(self.open_dbscan_window)
        tune_btn(self.dbscan_btn, half_w)
        btn_row.addWidget(self.dbscan_btn)

        btn_wrap = QtWidgets.QWidget()
        btn_wrap.setLayout(btn_row)
        right_controls.addWidget(btn_wrap, 0, Qt.AlignHCenter)

        # --- Multicolor section (divider + controls) ---
        mc = QtWidgets.QWidget()
        mc_lay = QtWidgets.QVBoxLayout(mc)
        mc_lay.setContentsMargins(0, 0, 0, 0)
        mc_lay.setSpacing(0)   # tighter default spacing between items

        # divider “Multicolor”
        mc_lay.addWidget(make_labeled_separator("Multicolor"))

        # reduce spacing between separator and viewer a bit more
        mc_lay.addSpacing(0)

        # Viewer (full width)
        self.multi_btn = QtWidgets.QPushButton("Viewer")
        self.multi_btn.clicked.connect(self.open_multicolor)
        tune_btn(self.multi_btn, VIEWER_W)
        mc_lay.addWidget(self.multi_btn, 0, Qt.AlignHCenter)

        # increase spacing between viewer and align/reset row
        mc_lay.addSpacing(16)

        # Row: Align mbm + Reset (together same width as Viewer)
        roww = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(roww)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)  # reduced spacing between Align and Reset

        half_w = (VIEWER_W - row.spacing()) // 2

        self.mbm_align_btn = QtWidgets.QPushButton("Align")
        self.mbm_align_btn.clicked.connect(self.on_mbm_align_clicked)
        tune_btn(self.mbm_align_btn, half_w)

        self.mbm_reset_btn = QtWidgets.QPushButton("Reset")
        self.mbm_reset_btn.clicked.connect(self.on_mbm_reset_clicked)
        tune_btn(self.mbm_reset_btn, half_w)

        row.addWidget(self.mbm_align_btn)
        row.addWidget(self.mbm_reset_btn)
        mc_lay.addWidget(roww, 0, Qt.AlignHCenter)

        # Merged checkbox (centered)
        chk_wrap = QtWidgets.QWidget()
        chk_lay = QtWidgets.QHBoxLayout(chk_wrap)
        chk_lay.setContentsMargins(0, 0, 0, 0)
        chk_lay.addStretch(1)

        self.merged_chk = QtWidgets.QCheckBox("Merged")
        self.merged_chk.setChecked(False)
        self.merged_chk.stateChanged.connect(self.on_merged_toggled)
        chk_lay.addWidget(self.merged_chk, 0)

        chk_lay.addStretch(1)
        mc_lay.addWidget(chk_wrap)

        right_controls.addWidget(mc)
        
        # output table
        outbox = QtWidgets.QGroupBox("Output")
        right.addWidget(outbox)
        ov = QtWidgets.QVBoxLayout(outbox)

        self.out_table = QtWidgets.QTableWidget(0, 2)
        self.out_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.out_table.horizontalHeader().setStretchLastSection(True)
        self.out_table.setColumnWidth(0, 200)  # ADD THIS LINE - set first column width
        self.out_table.verticalHeader().setVisible(False)
        self.out_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        font = self.out_table.font()
        font.setPointSize(12)  # Adjust size as needed (default is usually 13)
        self.out_table.setFont(font)
        self.out_table.verticalHeader().setDefaultSectionSize(20)  # Row height in pixels

        ov.addWidget(self.out_table)

        # ---- bottom area: histogram + plotly ----
        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom, 5)

        # histogram panel
        hist_box = QtWidgets.QGroupBox("Filter selection")
        bottom.addWidget(hist_box, 1)
        hv = QtWidgets.QVBoxLayout(hist_box)

        self.filter_tabs = QtWidgets.QTabWidget()
        hv.addWidget(self.filter_tabs)

        self.efo_tab = QtWidgets.QWidget()
        self.filter_tabs.addTab(self.efo_tab, "EFO")
        efo_v = QtWidgets.QVBoxLayout(self.efo_tab)
        self.fig_efo = Figure(figsize=(5, 4.5), dpi=100)
        self.ax_efo = self.fig_efo.add_subplot(111)
        self.canvas_efo = FigureCanvas(self.fig_efo)
        apply_hist_theme(self.fig_efo, self.ax_efo)
        self.canvas_efo.draw()
        efo_v.addWidget(self.canvas_efo)
        self.nav_toolbar_efo = NavigationToolbar2QT(self.canvas_efo, self)
        self.nav_toolbar_efo.setIconSize(QtCore.QSize(16, 16))
        self.nav_toolbar_efo.setFixedHeight(28)
        efo_v.addWidget(self.nav_toolbar_efo)
        self.efo_range_lbl = QtWidgets.QLabel("Selected EFO range: —")
        efo_v.addWidget(self.efo_range_lbl)

        self.cfr_tab = QtWidgets.QWidget()
        self.filter_tabs.addTab(self.cfr_tab, "CFR")
        cfr_v = QtWidgets.QVBoxLayout(self.cfr_tab)
        self.fig_cfr = Figure(figsize=(5, 4.5), dpi=100)
        self.ax_cfr = self.fig_cfr.add_subplot(111)
        self.canvas_cfr = FigureCanvas(self.fig_cfr)
        apply_hist_theme(self.fig_cfr, self.ax_cfr)
        self.canvas_cfr.draw()
        cfr_v.addWidget(self.canvas_cfr)
        self.nav_toolbar_cfr = NavigationToolbar2QT(self.canvas_cfr, self)
        self.nav_toolbar_cfr.setIconSize(QtCore.QSize(16, 16))
        self.nav_toolbar_cfr.setFixedHeight(28)
        cfr_v.addWidget(self.nav_toolbar_cfr)
        self.cfr_range_lbl = QtWidgets.QLabel("Selected CFR range: —")
        cfr_v.addWidget(self.cfr_range_lbl)

        ctrl = QtWidgets.QHBoxLayout()
        hv.addLayout(ctrl)
        ctrl.addWidget(QtWidgets.QLabel("Use tabs to select EFO and CFR ranges."), 1)

        self.apply_btn = QtWidgets.QPushButton("Apply filter")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_preview)
        ctrl.addWidget(self.apply_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset filter")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_filter)
        ctrl.addWidget(self.reset_btn)

        self.back_btn = QtWidgets.QPushButton("Back")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self.go_back)
        ctrl.addWidget(self.back_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.go_next)
        ctrl.addWidget(self.next_btn)

        self.save_all_btn = QtWidgets.QPushButton("Save all")
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.clicked.connect(self.save_all)
        ctrl.addWidget(self.save_all_btn)

        self._span_efo = None
        self._span_efo_xmin = None
        self._span_efo_xmax = None
        self._span_cfr = None
        self._span_cfr_xmin = None
        self._span_cfr_xmax = None

        self._mbm_dir_by_base = {}          # base -> mbm folder
        self._mbm_source_base = None        # selected source base
        self._mbm_enabled = False

        self._raw_arr_by_base = {}          # base -> current array (EFO-filtered if applied)
        self._aligned_arr_by_base = {}      # base -> aligned array (same filtering), computed when enabled
        self._T_by_base = {}                # base -> 4x4 transform to source

        # plotly panel
        # plotly panel
        plot_box = QtWidgets.QGroupBox("Scatter plot")
        bottom.addWidget(plot_box, 1) 
        pv = QtWidgets.QVBoxLayout(plot_box)
        self.plot_view = PlotlyView()
        self.plot_view.viewChanged.connect(self.on_any_plot_view_changed)
        pv.addWidget(self.plot_view)
        # status
        self.statusBar().showMessage("Ready.")

    def _plotly_bootstrap_html(self):
            return """<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        html, body { margin:0; padding:0; width:100%; height:100%; overflow:hidden; }
        #container { width:100%; height:100%; }
        #plot { width:100%; height:100%; }
    </style>
    </head>
    <body>
    <div id="container"><div id="plot"></div></div>
    <script>
        window._plotState = { camera3d: null, range2d: null, currentMode: null };
    </script>
    </body>
    </html>"""

    def ensure_plotly_page(self):
        if self._plotly_loaded:
            return
        self._plotly_loaded = True
        self._plotly_ready = False
        self.web.setHtml(self._plotly_bootstrap_html(), QUrl("about:blank"))

    def open_binning_window(self):
            if self._binning_win is None:
                self._binning_win = BinningWindow(self, self)

            bases = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]
            self._binning_win.rebuild(bases)

            self._binning_win.show()
            self._binning_win.raise_()
            self._binning_win.activateWindow()

    def open_dbscan_window(self):
            if self._dbscan_win is None:
                self._dbscan_win = DBSCANWindow(self, self)

            bases = self._dbscan_available_bases()
            self._dbscan_win.rebuild(bases)

            self._dbscan_win.show()
            self._dbscan_win.raise_()
            self._dbscan_win.activateWindow()

    def _on_plotly_load_finished(self, ok: bool):
        self._plotly_ready = bool(ok)
        if ok and self._pending_fig is not None:
            fig = self._pending_fig
            self._pending_fig = None
            self.update_plotly_fig(fig)

    def reset_session(self):
        """Reset all analysis state so a new folder starts from scratch."""
        # cancel any running worker
        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
                self._current_worker.wait(1000)
        except Exception:
            pass
        self._current_worker = None

        # clear state caches
        self._current_index = 0
        self._current_ctx = None
        self._plot_arr = None
        self._plot_is_filtered = False

        self._ctx_by_base.clear()
        self._efo_range_by_base.clear()
        self._cfr_range_by_base.clear()
        self._filtered_by_base.clear()

        self._raw_arr_by_base.clear()
        self._arr_by_base.clear()
        self._aligned_arr_by_base.clear()
        self._T_by_base.clear()
        self._multicolor_crop_bounds = None
        self._multicolor_cropped_by_base.clear()

        self._mbm_enabled = False
        self._mbm_source_base = None

        # reset UI controls
        self.apply_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)

        self.efo_range_lbl.setText("Selected EFO range: —")
        self.cfr_range_lbl.setText("Selected CFR range: —")
        self.set_output([])

        # clear histogram
        try:
            self.ax_efo.clear()
            apply_hist_theme(self.fig_efo, self.ax_efo)
            self.canvas_efo.draw()
            self.ax_cfr.clear()
            apply_hist_theme(self.fig_cfr, self.ax_cfr)
            self.canvas_cfr.draw()
        except Exception:
            pass

        # clear main plot
        try:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
            apply_plot_theme(fig, is3d=False)
            self.plot_view.update_fig(fig, reset_view=True, is3d=False)
        except Exception:
            pass

        # re-enable Run
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Ready.")

    def on_mbm_align_clicked(self):
        # infer source if needed
        if not self._mbm_source_base:
            for b, w in self.color_panel._widgets_by_base.items():
                chk = w.get("mbm_source")
                if chk and chk.isChecked():
                    self._mbm_source_base = b
                    break

        if not self._mbm_source_base:
            self.statusBar().showMessage("MBM align: no source selected")
            return

        # ensure default MBM paths even if Parameters was never confirmed with OK
        self._ensure_default_mbm_paths()

        # validate source path exists
        src = self._mbm_source_base
        src_path = self._mbm_dir_by_base.get(src, "")
        if not src_path or not os.path.isdir(src_path):
            QtWidgets.QMessageBox.critical(
                self,
                "MBM alignment",
                f"MBM path does not exist for source:\n\n{src}\n{src_path}\n\n"
                "Open Parameters to set MBM folders, or ensure /grd/mbm exists."
            )
            return

        # validate that every non-source dataset we might align has a valid path;
        # we can either error-out immediately, or skip missing ones. You asked to prompt error.
        missing = []
        for base in self._filtered_by_base.keys() or self._raw_arr_by_base.keys():
            if base == src:
                continue
            p = self._mbm_dir_by_base.get(base, "")
            if not p or not os.path.isdir(p):
                missing.append((base, p))

        if missing:
            msg = "MBM path does not exist for:\n\n" + "\n".join([f"{b}: {p}" for b, p in missing[:12]])
            if len(missing) > 12:
                msg += f"\n... and {len(missing) - 12} more"
            QtWidgets.QMessageBox.critical(self, "MBM alignment", msg)
            return

        # require data loaded
        if not (self._filtered_by_base or self._raw_arr_by_base):
            self.statusBar().showMessage("MBM align: no data loaded yet")
            return

        self._apply_mbm_alignment()

    def _ensure_default_mbm_paths(self):
        """
        Ensure self._mbm_dir_by_base has an entry for each base.
        Default: <folder containing the .npy>/grd/mbm
        """
        if not hasattr(self, "_mbm_dir_by_base") or self._mbm_dir_by_base is None:
            self._mbm_dir_by_base = {}

        # build base -> file_path map from _all_files
        for fp in self._all_files:
            base = os.path.splitext(os.path.basename(fp))[0]
            if base in self._mbm_dir_by_base and self._mbm_dir_by_base[base]:
                continue
            self._mbm_dir_by_base[base] = os.path.join(os.path.dirname(fp), "grd", "mbm")

    def on_mbm_reset_clicked(self):
        self._remove_mbm_alignment()
        self.statusBar().showMessage("MBM alignment reset")


    def on_color_settings_changed(self, base: str, settings: dict):
        if not isinstance(settings, dict):
            return

        # --- MBM source selection event ---
        if "mbm_source" in settings:
            if settings.get("mbm_source"):
                self._mbm_source_base = base
            else:
                if getattr(self, "_mbm_source_base", None) == base:
                    self._mbm_source_base = None

            # if user changes source while mbm is enabled, re-apply
            if self._mbm_enabled:
                self._apply_mbm_alignment()
            return

        mode = settings.get("mode", DEFAULT_MODE)
        if mode not in MODE_CHOICES:
            mode = DEFAULT_MODE

        solid = settings.get("solid", DEFAULT_SOLID)
        if solid not in SOLID_COLOR_CHOICES:
            solid = DEFAULT_SOLID

        lut = settings.get("lut", DEFAULT_LUT)
        if lut not in LUT_CHOICES:
            lut = DEFAULT_LUT

        alpha = float(settings.get("alpha", DEFAULT_ALPHA))
        alpha = max(0.0, min(1.0, alpha))

        size = int(settings.get("size", DEFAULT_SIZE_2D))
        size = max(1, min(50, size))

        self._color_settings_by_base[base] = {
            "mode": mode,
            "solid": solid,
            "lut": lut,
            "alpha": alpha,
            "size": size,
        }

        self.redraw_scatter(reset_view=False)
        self._refresh_multicolor_contents(reset_view=False)

        
    def open_parameters_dialog(self):
        if not self._all_files:
            self.refresh_files()
        dlg = ParametersDialog(self)
        dlg.move(self.geometry().center() - dlg.rect().center())
        dlg.set_values(
            min_trace_len=self.min_trace.value(),
            z_corr=self.zcorr.value(),
            scale=self.scale.value(),
            bin_size=self.bin_size.value(),
            cfr_bin_count=self.cfr_bin_count.value(),
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )

        dlg.set_mbm_rows(self._all_files)   # <-- required
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._mbm_dir_by_base = dlg.mbm_map()
            vals = dlg.values()
            self.cfr_bin_count.setValue(vals["cfr_bin_count"])
            self.min_trace.setValue(vals["min_trace_len"])
            self.zcorr.setValue(vals["z_corr"])
            self.scale.setValue(vals["scale"])
            self.bin_size.setValue(vals["bin_size"])
            self._scalebar_nm = float(vals["scalebar_nm"])
            
            self.redraw_scatter(reset_view=False)
            self._refresh_multicolor_contents(reset_view=False)
            if self._binning_win is not None and self._binning_win.isVisible():
                self._binning_win.refresh_plot(keep_view=True)

            # if a histogram is currently shown, refreshing it to new bin size can be helpful:
            if self._current_ctx is not None:
                self.on_need_efo(self._current_ctx)

    def _refresh_all_plots_same_data(self):
        self.redraw_scatter(reset_view=False)
        self._refresh_multicolor_contents(reset_view=False)
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return

    def open_multicolor(self):
        if self._multicolor_win is None:
            self._multicolor_win = MultiColorWindow(
                self.on_any_plot_view_changed,
                self._on_multicolor_crop_requested,
                self._on_multicolor_reset_crop_requested,
                self,
            )

        bases = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]
        self._multicolor_win.rebuild(bases)

        mode = "merged" if getattr(self, "merged_chk", None) and self.merged_chk.isChecked() else "grid"
        self._multicolor_win.set_mode(mode)

        self._multicolor_win.show()
        self._multicolor_win.raise_()
        self._multicolor_win.activateWindow()

        # IMPORTANT: defer initial drawing so QWebEngineViews can finish bootstrapping
        QtCore.QTimer.singleShot(0, lambda: self._refresh_multicolor_contents(reset_view=True))
        QtCore.QTimer.singleShot(120, lambda: self._refresh_multicolor_contents(reset_view=False))

    def on_merged_toggled(self):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        mode = "merged" if self.merged_chk.isChecked() else "grid"
        self._multicolor_win.set_mode(mode)

        # keep same window size; just swap content
        self._refresh_multicolor_contents(reset_view=False)

    def _refresh_multicolor_contents(self, reset_view=False):
        """Refresh either grid figs or merged fig depending on mode."""
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return

        self._recompute_multicolor_crop_cache()

        is3d = self.is3d.isChecked()
        avg = self.avg_tid.isChecked()
        sb = float(getattr(self, "_scalebar_nm", 100.0))
        if self._multicolor_win.mode() == "merged":
            # Build merged from whatever is in _arr_by_base (trace-filtered or EFO-filtered)
            # preserve insertion order from _all_files:
            arr_by_base = {}
            for f in self._all_files:
                base = os.path.splitext(os.path.basename(f))[0]
                arr = self._get_multicolor_arr_for_base(base)
                if arr is not None:
                    arr_by_base[base] = arr

            fig = make_plotly_fig_merged(
                arr_by_base,
                avg_tid=avg,
                is3d=is3d,
                color_settings_by_base=self._color_settings_by_base,
                scalebar_nm=getattr(self, "_scalebar_nm", 100.0)
            )
            self._multicolor_win.update_merged(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)
        else:
            figs = {}
            for f in self._all_files:
                base = os.path.splitext(os.path.basename(f))[0]
                cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})
                arr = self._get_multicolor_arr_for_base(base)
                if arr is None:
                    continue
                figs[base] = make_plotly_fig(arr, avg, is3d, color_settings=cs, scalebar_nm=getattr(self, "_scalebar_nm", 100.0))

            self._multicolor_win.update_all(figs, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)

    def _get_arr_for_base(self, base: str):
        arr = self._arr_by_base.get(base)
        if arr is not None:
            return arr
        # fallback: if the currently loaded ctx matches, use it
        if self._current_ctx is not None and self._current_ctx.get("base") == base:
            return self._current_ctx.get("arr")
        return None

    def _on_multicolor_crop_requested(self, payload: dict):
        if not isinstance(payload, dict):
            return
        xr = payload.get("xRange")
        yr = payload.get("yRange")
        if not (isinstance(xr, (list, tuple)) and isinstance(yr, (list, tuple)) and len(xr) == 2 and len(yr) == 2):
            QtWidgets.QMessageBox.information(self, "Crop", "Use Box select on a multicolor plot first.")
            return
        try:
            xmin = float(min(xr[0], xr[1]))
            xmax = float(max(xr[0], xr[1]))
            ymin = float(min(yr[0], yr[1]))
            ymax = float(max(yr[0], yr[1]))
        except Exception:
            QtWidgets.QMessageBox.information(self, "Crop", "Invalid crop selection.")
            return

        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            QtWidgets.QMessageBox.information(self, "Crop", "Crop range must be finite.")
            return

        self._multicolor_crop_bounds = (xmin, xmax, ymin, ymax)
        self._recompute_multicolor_crop_cache()
        self._refresh_multicolor_contents(reset_view=True)
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

        n = int(sum(len(v) for v in self._multicolor_cropped_by_base.values() if v is not None))
        self.statusBar().showMessage(f"Multicolor crop applied: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}], kept {n} locs")

    def _on_multicolor_reset_crop_requested(self):
        if not getattr(self, "_multicolor_crop_bounds", None):
            return
        self._multicolor_crop_bounds = None
        self._multicolor_cropped_by_base.clear()
        self._refresh_multicolor_contents(reset_view=True)
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()
        self.statusBar().showMessage("Multicolor crop reset")

    def _recompute_multicolor_crop_cache(self):
        self._multicolor_cropped_by_base = {}
        bounds = getattr(self, "_multicolor_crop_bounds", None)
        if not bounds:
            return

        xmin, xmax, ymin, ymax = bounds
        for f in self._all_files:
            base = os.path.splitext(os.path.basename(f))[0]
            arr = self._get_arr_for_base(base)
            if arr is None:
                continue
            if len(arr) == 0:
                self._multicolor_cropped_by_base[base] = arr
                continue

            try:
                loc = np.asarray(arr["loc"], dtype=float)
                if loc.ndim != 2 or loc.shape[1] < 2:
                    self._multicolor_cropped_by_base[base] = arr
                    continue
                x = loc[:, 0]
                y = loc[:, 1]
                m = np.isfinite(x) & np.isfinite(y)
                m &= (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                self._multicolor_cropped_by_base[base] = arr[m]
            except Exception:
                self._multicolor_cropped_by_base[base] = arr

    def _get_multicolor_arr_for_base(self, base: str):
        if getattr(self, "_multicolor_crop_bounds", None):
            return self._multicolor_cropped_by_base.get(base)
        return self._get_arr_for_base(base)

    def on_any_plot_view_changed(self, source_id, payload):
        # Synchronization intentionally disabled: each viewer keeps independent view state.
        return

    def _multicolor_update_base(self, base, reset_view=False):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        self._recompute_multicolor_crop_cache()
        arr = self._get_multicolor_arr_for_base(base)
        if arr is None:
            return
        cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})
        fig = make_plotly_fig(
            arr,
            self.avg_tid.isChecked(),
            self.is3d.isChecked(),
            color_settings=cs,
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )
        self._multicolor_win.update_one(
            base, fig,
            reset_view=reset_view,
            is3d=self.is3d.isChecked(),
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )

    def _multicolor_rebuild_if_open(self):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        bases = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]
        self._multicolor_win.rebuild(bases)

    def _dbscan_rebuild_if_open(self):
        if self._dbscan_win is None or not self._dbscan_win.isVisible():
            return
        bases = self._dbscan_available_bases()
        self._dbscan_win.rebuild(bases)

    def _dbscan_available_bases(self):
        # Match multicolor loading behavior: include bases that currently resolve
        # to an array through _get_multicolor_arr_for_base/_get_arr_for_base.
        ordered_all = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]

        if getattr(self, "_multicolor_crop_bounds", None):
            self._recompute_multicolor_crop_cache()

        available = []
        for b in ordered_all:
            arr = self._get_multicolor_arr_for_base(b)
            if arr is None:
                arr = self._get_arr_for_base(b)
            if arr is not None:
                available.append(b)

        if available:
            return available

        return ordered_all

    def update_plotly_fig(self, fig, reset_view=False):
        if (not self.is3d.isChecked()) and fig is not None:
            add_scalebar_2d(fig, length_nm=float(getattr(self, "_scalebar_nm", 100.0)))
        self.plot_view.update_fig(
            fig,
            reset_view=reset_view,
            is3d=self.is3d.isChecked(),
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )
    # ---------------- UI actions ----------------
    def browse_folder(self):
        old = self.data_edit.text().strip()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select MINFLUX data folder")
        if path:
            path = os.path.normpath(path)
            if os.path.normpath(old) != path:
                self.reset_session()   # <-- key line

            self.data_edit.setText(path)
            self.set_default_output()
            self.refresh_files()

    def set_default_output(self):
        p = self.data_edit.text().strip()
        if p:
            self.out_edit.setText(p + "_filtered")

    def closeEvent(self, event):
        self._closing = True

        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
                self._current_worker.wait(1000)
        except Exception:
            pass

        try:
            if self.plot_view is not None:
                self.plot_view.shutdown()
        except Exception:
            pass

        try:
            if self._multicolor_win is not None:
                self._multicolor_win.shutdown()
                self._multicolor_win.close()
        except Exception:
            pass

        try:
            if self._binning_win is not None:
                self._binning_win.shutdown()
                self._binning_win.close()
        except Exception:
            pass

        try:
            if self._dbscan_win is not None:
                self._dbscan_win.shutdown()
                self._dbscan_win.close()
        except Exception:
            pass

        event.accept()
        QtWidgets.QApplication.quit()
        
    def _on_worker_finished(self, worker):
        """Clean up worker reference when it finishes."""
        if self._current_worker is worker:
            self._current_worker = None    

    def show_plotly(self, html: str):
        if self._plotly_tmp is None:
            fd, path = tempfile.mkstemp(suffix=".html")
            os.close(fd)
            self._plotly_tmp = path

        with open(self._plotly_tmp, "w", encoding="utf-8") as f:
            f.write(html)

        self.web.load(QUrl.fromLocalFile(self._plotly_tmp))

    def refresh_files(self):
        data_path = self.data_edit.text().strip()
        if not data_path or not os.path.isdir(data_path):
            self._all_files = []
        else:
            files = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
            files.sort()
            self._all_files = files

        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for f in self._all_files:
            self.file_combo.addItem(os.path.relpath(f, data_path))
        self.file_combo.blockSignals(False)
        self._multicolor_rebuild_if_open()
        self._dbscan_rebuild_if_open()

        self._current_index = 0 if self._all_files else -1
        if self._all_files:
            self.file_combo.setCurrentIndex(0)
        bases = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]
        self.color_panel.rebuild(bases, self._color_settings_by_base)
        self._ensure_default_mbm_paths()

    def _update_output_for_base(self, base: str, arr_efo, n_after: int = None):
        """
        Update the Output table for the given base, using the currently stored ctx.
        Call this after applying EFO (and/or alignment) so the table matches what you're viewing.
        """
        ctx = self._ctx_by_base.get(base) or self._current_ctx
        if ctx is None:
            return

        if n_after is None:
            n_after = int(len(arr_efo)) if arr_efo is not None else 0

        # localization precision preview
        lp = preview_localization_precision(arr_efo)
        lp = tuple(float(f"{v:.2f}") for v in lp) if lp is not None else None

        # ratio loc per trace
        if arr_efo is None or len(arr_efo) == 0:
            ut = np.array([])
            ratio_loc_per_trace = 0.0
        else:
            ut = np.unique(arr_efo["tid"])
            ratio_loc_per_trace = float(len(arr_efo) / len(ut)) if len(ut) else 0.0

        last_it = int(ctx.get("last_iteration_loc", max(1, ctx.get("total_loc", 1))))

        items = [
            ("File", base),
            ("Total imaging time (min)", f"{ctx['total_tim']/60:.2f}"),
            ("Last iteration localizations", str(last_it)),
            ("After trace filtering", str(ctx.get("after_trace", "—"))),
            ("After EFO and CFR filtering", str(n_after)),
            ("Total traces", str(ut.size if arr_efo is not None else "—")),
            ("Localization precision (x, y, z)", str(lp) if lp is not None else "—"),
            ("Ratio loc per trace", f"{ratio_loc_per_trace:.4f}"),
            ("% remaining", f"{(n_after / last_it * 100):.2f}%"),
            ("Note", "In-memory preview. Use Save all to write files."),
        ]
        self.set_output(items)

    def on_file_changed(self, idx):
        if getattr(self, "_closing", False):
            return
        if idx < 0 or idx >= len(self._all_files):
            return

        self._current_index = idx
        f = self._all_files[idx]
        base = os.path.splitext(os.path.basename(f))[0]

        # if already loaded -> display from memory
        if base in self._ctx_by_base:
            ctx = self._ctx_by_base[base]
            self.on_need_efo(ctx)
            return

        # else: load it
        if self.run_btn.isEnabled() is False:
            self.start_worker_for_current()

    def _apply_filter_ranges_to_memory(self, base, efo_xmin, efo_xmax, cfr_xmin, cfr_xmax, *, update_view=True, reset_view=False):
        arr = self._raw_arr_by_base.get(base)
        if arr is None:
            return

        mask_efo = (arr["efo"] >= efo_xmin) & (arr["efo"] <= efo_xmax)
        mask_cfr = (arr["cfr"] >= cfr_xmin) & (arr["cfr"] <= cfr_xmax)
        mask = mask_efo & mask_cfr
        arr_filt = arr[mask]

        if len(arr_filt) == 0:
            QtWidgets.QMessageBox.warning(self, "Filter", "No localizations left, adjust filter")
            return

        self._efo_range_by_base[base] = (float(efo_xmin), float(efo_xmax))
        self._cfr_range_by_base[base] = (float(cfr_xmin), float(cfr_xmax))
        self._filtered_by_base[base] = arr_filt

        # default plotted data is the filtered data (or aligned filtered if enabled)
        if self._mbm_enabled and base in self._aligned_arr_by_base:
            self._arr_by_base[base] = self._aligned_arr_by_base[base]
        else:
            self._arr_by_base[base] = arr_filt

        if update_view:
            self.redraw_scatter(reset_view=reset_view)
            self._refresh_multicolor_contents(reset_view=False)
            self._update_output_for_base(base, arr_filt, int(np.count_nonzero(mask)))
            if self._dbscan_win is not None and self._dbscan_win.isVisible():
                self._dbscan_win.refresh_current_plot()

    def run_start(self):
        data_path = self.data_edit.text().strip()
        save_folder = self.out_edit.text().strip()
        if not data_path or not os.path.isdir(data_path):
            QtWidgets.QMessageBox.critical(self, "Error", "Please select a valid MINFLUX data folder.")
            return
        if not save_folder:
            QtWidgets.QMessageBox.critical(self, "Error", "Please set an output folder.")
            return

        if not self._all_files:
            self.refresh_files()
        if not self._all_files:
            QtWidgets.QMessageBox.critical(self, "Error", "No .npy files found.")
            return
        os.makedirs(save_folder, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.start_worker_for_current()

    def start_worker_for_current(self):
        if getattr(self, "_closing", False):
            return

        if self._current_index < 0 or self._current_index >= len(self._all_files):
            self.statusBar().showMessage("No more files.")
            self.run_btn.setEnabled(True)
            return

        params = dict(
            min_trace_len=int(self.min_trace.value()),
            z_corr=float(self.zcorr.value()),
            scale=float(self.scale.value()),
            save_folder=self.out_edit.text().strip(),
            bin_size=float(self.bin_size.value()),
        )
        f = self._all_files[self._current_index]

        self.apply_btn.setEnabled(False)

        w = FileWorker(f, params, parent=self)

        self._workers.add(w)
        
        # Clean up references properly
        w.finished.connect(lambda: self._workers.discard(w))
        w.finished.connect(lambda: self._on_worker_finished(w))  # ADD THIS LINE
        w.finished.connect(w.deleteLater)

        w.need_efo.connect(self.on_need_efo)
        w.status.connect(self.statusBar().showMessage)
        w.done_one.connect(self.on_done_one)

        self._current_worker = w
        w.start()

    # ---------------- EFO + CFR selection ----------------
    def on_need_efo(self, ctx):
        """
        Called by FileWorker when a file is loaded and trace-filtered and the GUI should
        display EFO/CFR histograms + initial scatter preview.

        This version also:
        - caches the current dataset into _raw_arr_by_base (for MBM align enable/disable)
        - updates _arr_by_base to either raw or aligned depending on mbm_align toggle
        - refreshes main plot + multicolor plots
        """
        # ---- basic state ----
        self._plot_arr = ctx["arr"]          # start from trace-filtered data
        self._plot_is_filtered = False
        self._current_ctx = ctx

        base = ctx.get("base")
        if not base:
            return

        arr = ctx.get("arr")
        if arr is None:
            return

        efo_vals = np.asarray(ctx["efo_vals"])
        cfr_vals = np.asarray(ctx["cfr_vals"])
        bin_size = float(self.bin_size.value())

        # Ensure caches exist
        if not hasattr(self, "_raw_arr_by_base") or self._raw_arr_by_base is None:
            self._raw_arr_by_base = {}
        if not hasattr(self, "_arr_by_base") or self._arr_by_base is None:
            self._arr_by_base = {}

        # ---- cache as "raw" (trace-filtered) for this base ----
        self._raw_arr_by_base[base] = arr
        self._arr_by_base[base] = arr

        # ---- EFO histogram ----
        self.ax_efo.clear()
        if len(efo_vals) > 0:
            bins = np.arange(efo_vals.min(), efo_vals.max() + bin_size, bin_size)
            self.ax_efo.hist(
                efo_vals,
                bins=bins,
                edgecolor=HIST_THEME["hist_edge"],
                color=HIST_THEME["hist_face"],
            )

            efo_med = float(np.median(efo_vals))
            self.ax_efo.axvline(efo_med, color="red", linewidth=2)
            self.ax_efo.text(
                efo_med,
                self.ax_efo.get_ylim()[1] * 0.95,
                f"median={efo_med:.2f}",
                color="red",
                ha="left",
                va="top",
            )
        #self.ax_efo.set_xlabel("EFO")
        self.ax_efo.set_ylabel("Count")
        self.ax_efo.set_title(base)

        apply_hist_theme(self.fig_efo, self.ax_efo)
        self.canvas_efo.draw()

        if len(efo_vals) > 0:
            if base in self._efo_range_by_base:
                self._span_efo_xmin, self._span_efo_xmax = self._efo_range_by_base[base]
            else:
                self._span_efo_xmin = float(efo_vals.min())
                self._span_efo_xmax = float(efo_vals.max())
        else:
            self._span_efo_xmin = None
            self._span_efo_xmax = None

        def onselect_efo(xmin, xmax):
            self._span_efo_xmin, self._span_efo_xmax = float(xmin), float(xmax)
            self.efo_range_lbl.setText(f"Selected EFO range: {self._span_efo_xmin:.2f} ... {self._span_efo_xmax:.2f}")

        if self._span_efo is not None:
            self._span_efo.disconnect_events()

        if self._span_efo_xmin is not None and self._span_efo_xmax is not None:
            self._span_efo = SpanSelector(
                self.ax_efo, onselect_efo, direction="horizontal", useblit=True,
                props=dict(
                    facecolor=HIST_THEME["span_face"],
                    alpha=HIST_THEME["span_alpha"],
                    edgecolor=HIST_THEME["span_edge"],
                    linewidth=HIST_THEME["span_lw"],
                ),
                interactive=True, drag_from_anywhere=True
            )
            self._span_efo.extents = (self._span_efo_xmin, self._span_efo_xmax)
            onselect_efo(self._span_efo_xmin, self._span_efo_xmax)
        else:
            self._span_efo = None
            self.efo_range_lbl.setText("Selected EFO range: —")

        # ---- CFR histogram ----
        self.ax_cfr.clear()
        if len(cfr_vals) > 0:
            self.ax_cfr.hist(
                cfr_vals,
                bins=int(self.cfr_bin_count.value()),
                edgecolor=HIST_THEME["hist_edge"],
                color=HIST_THEME["hist_face"],
            )

            cfr_med = float(np.median(cfr_vals))
            self.ax_cfr.axvline(cfr_med, color="red", linewidth=2)
            self.ax_cfr.text(
                cfr_med,
                self.ax_cfr.get_ylim()[1] * 0.95,
                f"median={cfr_med:.2f}",
                color="red",
                ha="left",
                va="top",
            )
        #self.ax_cfr.set_xlabel("CFR")
        self.ax_cfr.set_ylabel("Count")
        self.ax_cfr.set_title(base)

        apply_hist_theme(self.fig_cfr, self.ax_cfr)
        self.canvas_cfr.draw()

        if len(cfr_vals) > 0:
            if base in self._cfr_range_by_base:
                self._span_cfr_xmin, self._span_cfr_xmax = self._cfr_range_by_base[base]
            else:
                self._span_cfr_xmin = float(cfr_vals.min())
                self._span_cfr_xmax = float(cfr_vals.max())
        else:
            self._span_cfr_xmin = None
            self._span_cfr_xmax = None

        def onselect_cfr(xmin, xmax):
            self._span_cfr_xmin, self._span_cfr_xmax = float(xmin), float(xmax)
            self.cfr_range_lbl.setText(f"Selected CFR range: {self._span_cfr_xmin:.2f} ... {self._span_cfr_xmax:.2f}")

        if self._span_cfr is not None:
            self._span_cfr.disconnect_events()

        if self._span_cfr_xmin is not None and self._span_cfr_xmax is not None:
            self._span_cfr = SpanSelector(
                self.ax_cfr, onselect_cfr, direction="horizontal", useblit=True,
                props=dict(
                    facecolor=HIST_THEME["span_face"],
                    alpha=HIST_THEME["span_alpha"],
                    edgecolor=HIST_THEME["span_edge"],
                    linewidth=HIST_THEME["span_lw"],
                ),
                interactive=True, drag_from_anywhere=True
            )
            self._span_cfr.extents = (self._span_cfr_xmin, self._span_cfr_xmax)
            onselect_cfr(self._span_cfr_xmin, self._span_cfr_xmax)
        else:
            self._span_cfr = None
            self.cfr_range_lbl.setText("Selected CFR range: —")

        self.apply_btn.setEnabled(True)
        self.reset_btn.setEnabled(
            self._span_efo_xmin is not None and self._span_efo_xmax is not None
            and self._span_cfr_xmin is not None and self._span_cfr_xmax is not None
        )

        # ---- cache ctx for later "Save all" ----
        self._ctx_by_base[base] = ctx

        # ---- apply existing ranges if present; otherwise default each to full range ----
        if len(efo_vals) > 0 and len(cfr_vals) > 0:
            if base in self._efo_range_by_base:
                efo_xmin, efo_xmax = self._efo_range_by_base[base]
            else:
                efo_xmin, efo_xmax = float(efo_vals.min()), float(efo_vals.max())
                self._efo_range_by_base[base] = (efo_xmin, efo_xmax)

            if base in self._cfr_range_by_base:
                cfr_xmin, cfr_xmax = self._cfr_range_by_base[base]
            else:
                cfr_xmin, cfr_xmax = float(cfr_vals.min()), float(cfr_vals.max())
                self._cfr_range_by_base[base] = (cfr_xmin, cfr_xmax)

            self._apply_filter_ranges_to_memory(
                base,
                efo_xmin,
                efo_xmax,
                cfr_xmin,
                cfr_xmax,
                update_view=True,
                reset_view=True,
            )

        # ---- enable navigation + save ----
        self.back_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)

        # ---- update plotting arrays depending on MBM align toggle ----
        mbm_enabled = bool(getattr(self, "_mbm_enabled", False))
        if mbm_enabled and getattr(self, "_mbm_source_base", None):
            # will update _arr_by_base for all datasets + redraw plots
            self._apply_mbm_alignment()
        else:
            self._arr_by_base[base] = self._filtered_by_base.get(base, arr)

        

    def redraw_scatter(self, reset_view=False):
        if self._current_ctx is None:
            return

        base = self._current_ctx.get("base")
        if not base:
            return

        # Always plot whatever is currently active for this base (raw or aligned)
        arr_to_plot = self._arr_by_base.get(base)
        if arr_to_plot is None:
            # fallback to ctx
            arr_to_plot = self._current_ctx.get("arr")
            if arr_to_plot is None:
                return

        # keep state consistent
        self._plot_arr = arr_to_plot

        cs = self._color_settings_by_base.get(
            base,
            {
                "mode": DEFAULT_MODE,
                "solid": DEFAULT_SOLID,
                "lut": DEFAULT_LUT,
                "alpha": DEFAULT_ALPHA,
                "size": DEFAULT_SIZE_2D,
            },
        )

        fig = make_plotly_fig(
            arr_to_plot,
            self.avg_tid.isChecked(),
            self.is3d.isChecked(),
            color_settings=cs,
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )
        self.update_plotly_fig(fig, reset_view=reset_view)

    def apply_preview(self):
        if self._current_ctx is None:
            return
        base = self._current_ctx.get("base")
        if (
            not base
            or self._span_efo_xmin is None or self._span_efo_xmax is None
            or self._span_cfr_xmin is None or self._span_cfr_xmax is None
        ):
            return
        self._apply_filter_ranges_to_memory(
            base,
            self._span_efo_xmin,
            self._span_efo_xmax,
            self._span_cfr_xmin,
            self._span_cfr_xmax,
            update_view=True,
            reset_view=False,
        )

        # if alignment is enabled, recompute it on the *filtered* datasets
        if self._mbm_enabled:
            self._apply_mbm_alignment()
        if self._binning_win is not None and self._binning_win.isVisible():
            self._binning_win.refresh_plot()
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

    def reset_filter(self):
        if self._current_ctx is None:
            return

        base = self._current_ctx.get("base")
        if not base:
            return

        arr = self._raw_arr_by_base.get(base)
        if arr is None or len(arr) == 0:
            return

        efo_vals = np.asarray(arr["efo"])
        cfr_vals = np.asarray(arr["cfr"])
        if efo_vals.size == 0 or cfr_vals.size == 0:
            return

        efo_xmin = float(np.min(efo_vals))
        efo_xmax = float(np.max(efo_vals))
        cfr_xmin = float(np.min(cfr_vals))
        cfr_xmax = float(np.max(cfr_vals))
        self._span_efo_xmin, self._span_efo_xmax = efo_xmin, efo_xmax
        self._span_cfr_xmin, self._span_cfr_xmax = cfr_xmin, cfr_xmax

        if self._span_efo is not None:
            self._span_efo.extents = (efo_xmin, efo_xmax)
        if self._span_cfr is not None:
            self._span_cfr.extents = (cfr_xmin, cfr_xmax)

        self.efo_range_lbl.setText(f"Selected EFO range: {efo_xmin:.2f} ... {efo_xmax:.2f}")
        self.cfr_range_lbl.setText(f"Selected CFR range: {cfr_xmin:.2f} ... {cfr_xmax:.2f}")
        self.canvas_efo.draw_idle()
        self.canvas_cfr.draw_idle()

        # Reuse the same update path as Apply filter so all plots and outputs refresh.
        self._apply_filter_ranges_to_memory(
            base,
            efo_xmin,
            efo_xmax,
            cfr_xmin,
            cfr_xmax,
            update_view=True,
            reset_view=False,
        )

        if self._mbm_enabled:
            self._apply_mbm_alignment()
        if self._binning_win is not None and self._binning_win.isVisible():
            self._binning_win.refresh_plot()
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

    def on_mbm_align_toggled(self, state):
        enabled = (state == Qt.Checked)
        if enabled:
            if not getattr(self, "_mbm_source_base", None):
                self.statusBar().showMessage("MBM align: no source selected")
                return
            if not self._mbm_dir_by_base:
                self.statusBar().showMessage("MBM align: MBM folders not set (open Parameters and press OK)")
                return
            self._apply_mbm_alignment()
        else:
            self._remove_mbm_alignment()

    def save_all(self):
        save_folder = self.out_edit.text().strip()
        if not save_folder:
            QtWidgets.QMessageBox.critical(self, "Error", "Please set an output folder.")
            return
        os.makedirs(save_folder, exist_ok=True)

        for base, arr_filt in self._filtered_by_base.items():
            if arr_filt is None:
                continue

            # what you actually save (aligned if enabled)
            arr_to_save = self._aligned_arr_by_base.get(base, arr_filt) if self._mbm_enabled else arr_filt
            if arr_to_save is None:
                continue

            # --- save main localization CSV (UNCHANGED; no extra columns) ---
            df = np_to_df(arr_to_save)
            save_path = os.path.join(save_folder, f"{base}.csv")
            save_to_csv(df, save_path)

            # --- build per-tid stats DataFrame (as before) ---
            dims = [c.replace("loc_", "") for c in df.columns if c.startswith("loc_")]
            avg_df = pd.DataFrame()

            if len(df) > 0 and dims:
                for dim in dims:
                    avg_df[f"loc_{dim}_mean"] = df.groupby("tid")[f"loc_{dim}"].mean()
                    avg_df[f"loc_{dim}_std"] = df.groupby("tid")[f"loc_{dim}"].std()
                avg_df["n"] = df.groupby("tid")[f"loc_{dims[0]}"].count()
                avg_df["tim_tot"] = df.groupby("tid")["tim"].sum()

            # --- summary metrics: write into base_stats.csv ONLY (row 0 only) ---
            n_after = int(len(arr_to_save))

            ctx = self._ctx_by_base.get(base) or self._current_ctx
            last_it = 0
            if ctx and ctx.get("base") == base:
                try:
                    last_it = int(ctx.get("last_iteration_loc", 0))
                except Exception:
                    last_it = 0
            if last_it <= 0:
                last_it = max(n_after, 1)

            # localization precision (per dimension)
            lp = preview_localization_precision(arr_to_save)  # (x,y,z) or (x,y) or None

            # ratio loc per trace
            if n_after > 0:
                ut = np.unique(arr_to_save["tid"])
                ratio_loc_per_trace = float(n_after / len(ut)) if len(ut) else 0.0
            else:
                ratio_loc_per_trace = 0.0

            percent_remaining = float(n_after / last_it * 100.0) if last_it else float("nan")

            # Ensure at least one row exists so we can fill row 0
            if avg_df is None or len(avg_df) == 0:
                avg_df = pd.DataFrame(index=[0])

            # Add columns and fill only first row
            dim_names = ["x", "y", "z"]
            if lp is not None:
                for i, v in enumerate(lp):
                    col = f"localization_precision_{dim_names[i] if i < len(dim_names) else i}"
                    avg_df[col] = np.nan
                    avg_df.iloc[0, avg_df.columns.get_loc(col)] = float(v) if np.isfinite(v) else np.nan

            avg_df["ratio_loc_per_trace"] = np.nan
            avg_df.iloc[0, avg_df.columns.get_loc("ratio_loc_per_trace")] = float(ratio_loc_per_trace)

            avg_df["percent_remaining"] = np.nan
            avg_df.iloc[0, avg_df.columns.get_loc("percent_remaining")] = float(percent_remaining)

            avg_save_path = os.path.join(save_folder, f"{base}_stats.csv")
            avg_df.to_csv(avg_save_path, index=True)

        self.statusBar().showMessage(f"Saved {len(self._filtered_by_base)} datasets.")

    def _apply_mbm_alignment(self):
        self.statusBar().showMessage(
            f"MBM aligning to source={self._mbm_source_base}, have dirs={len(self._mbm_dir_by_base)}, "
            f"have filtered={len(self._filtered_by_base)}"
        )
        self._mbm_enabled = True

        src = self._mbm_source_base
        if not src:
            return

        # align the EFO-filtered arrays (what will be saved)
        src_arr = self._filtered_by_base.get(src)
        if src_arr is None:
            src_arr = self._arr_by_base.get(src)
        if src_arr is None:
            return

        mbm_src = self._mbm_dir_by_base.get(src)
        if not mbm_src or not os.path.isdir(mbm_src):
            self.statusBar().showMessage(f"MBM: source folder missing for {src}")
            return

        self._aligned_arr_by_base.clear()
        self._T_by_base.clear()

        # source stays unchanged
        self._aligned_arr_by_base[src] = src_arr
        self._T_by_base[src] = np.eye(4)

        for base, arr in self._filtered_by_base.items():
            if base == src:
                continue
            if arr is None or len(arr) == 0:
                self._aligned_arr_by_base[base] = arr
                continue

            mbm_mov = self._mbm_dir_by_base.get(base)
            if not mbm_mov or not os.path.isdir(mbm_mov):
                # keep unaligned if no mbm folder
                self._aligned_arr_by_base[base] = arr
                continue

            try:
                T, common, diagnostics = compute_mbm_transform(
                    mbm_src,
                    mbm_mov,
                    k=4,
                    scale=float(self.scale.value()),
                    z_corr=float(self.zcorr.value()),
                    return_diagnostics=True,
                )
                self._T_by_base[base] = T
                self._aligned_arr_by_base[base] = apply_transform_to_arr(arr, T)
                self._save_mbm_alignment_bead_plots(src, base, diagnostics, T)
            except Exception as e:
                self.statusBar().showMessage(f"MBM align failed for {base}: {e}")
                self._aligned_arr_by_base[base] = arr

        # drive plotting from aligned (aligned if present else original)
        for base, arr in self._filtered_by_base.items():
            self._arr_by_base[base] = self._aligned_arr_by_base.get(base, arr)

        self.redraw_scatter(reset_view=False)
        self._refresh_multicolor_contents(reset_view=False)
        if self._binning_win is not None and self._binning_win.isVisible():
            self._binning_win.refresh_plot()
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

    def _save_mbm_alignment_bead_plots(self, src_base: str, mov_base: str, diagnostics: dict, T=None):
        if not isinstance(diagnostics, dict):
            return

        out_root = self.out_edit.text().strip() if hasattr(self, "out_edit") else ""
        if not out_root:
            return

        ref = np.asarray(diagnostics.get("ref_common_scaled", diagnostics.get("ref_common", [])), dtype=float)
        mov = np.asarray(diagnostics.get("mov_common_scaled", diagnostics.get("mov_common", [])), dtype=float)
        keep = np.asarray(diagnostics.get("keep_mask", []), dtype=bool)

        if ref.ndim != 2 or mov.ndim != 2 or ref.shape[1] < 3 or mov.shape[1] < 3:
            return
        if len(ref) == 0 or len(mov) == 0 or len(ref) != len(mov):
            return
        if keep.shape[0] != len(ref):
            keep = np.ones(len(ref), dtype=bool)

        os.makedirs(out_root, exist_ok=True)
        diag_dir = os.path.join(out_root, "mbm_alignment_plots")
        os.makedirs(diag_dir, exist_ok=True)

        def _safe_name(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(s))

        src_name = _safe_name(src_base)
        mov_name = _safe_name(mov_base)
        stem = f"mbm_beads_{mov_name}_to_{src_name}"

        c_in = "#ff8c00"
        c_out = "#9e9e9e"

        fig2d = Figure(figsize=(8.0, 6.0), dpi=150)
        ax2d = fig2d.add_subplot(111)
        ax2d.scatter(ref[~keep, 0], ref[~keep, 1], c=c_out, s=26, marker="o", alpha=0.75, label=f"{src_base} filtered out")
        ax2d.scatter(ref[keep, 0], ref[keep, 1], c=c_in, s=30, marker="o", alpha=0.9, label=f"{src_base} included")
        ax2d.scatter(mov[~keep, 0], mov[~keep, 1], c=c_out, s=26, marker="^", alpha=0.75, label=f"{mov_base} filtered out")
        ax2d.scatter(mov[keep, 0], mov[keep, 1], c=c_in, s=30, marker="^", alpha=0.9, label=f"{mov_base} included")
        ax2d.set_title(f"MBM beads (2D): {mov_base} -> {src_base}")
        ax2d.set_xlabel("x")
        ax2d.set_ylabel("y")
        ax2d.grid(True, alpha=0.25)
        ax2d.legend(loc="best", fontsize=8)
        fig2d.tight_layout()

        fig3d = Figure(figsize=(8.4, 6.4), dpi=150)
        ax3d = fig3d.add_subplot(111, projection="3d")
        ref_used = ref[keep]
        mov_used = mov[keep]
        if isinstance(T, np.ndarray) and T.shape == (4, 4) and len(mov_used) > 0:
            mov_registered = apply_T(mov_used, T)
        else:
            mov_registered = mov_used

        ax3d.scatter(ref_used[:, 0], ref_used[:, 1], ref_used[:, 2], c=c_in, s=34, marker="o", alpha=0.9, label="reference")
        ax3d.scatter(mov_registered[:, 0], mov_registered[:, 1], mov_registered[:, 2], c="#1f77b4", s=34, marker="^", alpha=0.9, label="registered")
        if len(ref_used) > 0:
            dx = float(np.ptp(ref_used[:, 0]))
            dy = float(np.ptp(ref_used[:, 1]))
            dz = float(np.ptp(ref_used[:, 2]))
            ax3d.set_box_aspect((max(dx, 1e-12), max(dy, 1e-12), max(dz, 1e-12)))
        ax3d.set_title(f"MBM bead registration (3D): {mov_base} -> {src_base}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.legend(loc="best", fontsize=8)
        fig3d.tight_layout()

        out_2d = os.path.join(diag_dir, f"{stem}_2d.png")
        out_3d = os.path.join(diag_dir, f"{stem}_3d.png")
        fig2d.savefig(out_2d, dpi=200)
        fig3d.savefig(out_3d, dpi=200)

        self.statusBar().showMessage(f"MBM bead plots saved: {os.path.basename(out_2d)}, {os.path.basename(out_3d)}")

    def go_next(self):
        if not self._all_files:
            return
        self.file_combo.setCurrentIndex((self.file_combo.currentIndex() + 1) % len(self._all_files))

    def go_back(self):
        if not self._all_files:
            return
        self.file_combo.setCurrentIndex((self.file_combo.currentIndex() - 1) % len(self._all_files))

    def _remove_mbm_alignment(self):
        for base, arr in self._raw_arr_by_base.items():
            self._arr_by_base[base] = arr
        self._mbm_enabled = False
        self._aligned_arr_by_base.clear()
        self._T_by_base.clear()
        self.redraw_scatter(reset_view=False)
        self._refresh_multicolor_contents(reset_view=False)

        # NEW: refresh binning window too
        if self._binning_win is not None and self._binning_win.isVisible():
            self._binning_win.refresh_plot(keep_view=True)
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

    # ---------------- output / done ----------------
    def set_output(self, items):
        self.out_table.setRowCount(0)
        for r, (k, v) in enumerate(items):
            self.out_table.insertRow(r)
            self.out_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.out_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))

    def on_done_one(self, result):
        if getattr(self, "_closing", False):
            return
        if not result:
            return

        ctx = result.get("ctx")
        if not ctx:
            return

        base = ctx.get("base")
        if not base:
            return

        # store ctx; on_need_efo already updates UI
        self._ctx_by_base[base] = ctx

def handle_download_requested(download):
    suggested = download.suggestedFileName() or "plot.png"
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Save Plot",
        suggested,
        "PNG Image (*.png);;All Files (*)"
    )
    if not path:
        download.cancel()
        return

    if not os.path.splitext(path)[1]:
        path += ".png"

    download.setDownloadDirectory(os.path.dirname(path))
    download.setDownloadFileName(os.path.basename(path))
    download.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    apply_gui_theme(app)

    from PySide6.QtWebEngineCore import QWebEngineProfile
    profile = QWebEngineProfile.defaultProfile()
    profile.downloadRequested.connect(handle_download_requested)

    w = MainWindow()
    w.show()

    QtWidgets.QApplication.processEvents()  # ensure window is realized

    # pick the screen where the window currently is
    screen = w.screen()
    if screen is None:
        screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
    if screen is None:
        screen = QtWidgets.QApplication.primaryScreen()

    if screen is not None:
        g = screen.availableGeometry()
        if g.width() < 1500 or g.height() < 950:
            w.showMaximized()
        else:
            w.resize(1500, 950)
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()