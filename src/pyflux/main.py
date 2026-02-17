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

def match_and_filter_beads(beads_ref, beads_mov):
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

def compute_mbm_transform(mbm_ref_dir, mbm_mov_dir, k=4, scale=1.0):
    pts_ref = load_mbm_points(mbm_ref_dir)
    pts_mov = load_mbm_points(mbm_mov_dir)

    beads_ref = bead_initial_positions(pts_ref, k=k)
    beads_mov = bead_initial_positions(pts_mov, k=k)

    ref_pts, mov_pts, common = match_and_filter_beads(beads_ref, beads_mov)

    # IMPORTANT: match units to MINFLUX loc units
    ref_pts = ref_pts * scale
    mov_pts = mov_pts * scale

    _, T_total = icp(mov_pts, ref_pts, max_iterations=50, tolerance=0.5e-9 * scale)
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

        # Equal aspect ratio if non-degenerate
        dx = float(np.nanmax(x) - np.nanmin(x)) if len(x) else 0.0
        dy = float(np.nanmax(y) - np.nanmin(y)) if len(y) else 0.0
        if dx > 0 and dy > 0:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

    apply_plot_theme(fig, is3d=is3d)
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

        tr = make_trace_for_arr(
            arr,
            avg_tid=avg_tid,
            is3d=is3d,
            color_settings=cs,
            name=base,
            show_colorbar=True,  # NOTE: can cause overlapping bars if many traces have a colorbar
        )
        if tr is not None:
            traces.append(tr)

    if not traces:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        apply_plot_theme(fig, is3d=is3d)
        return fig

    fig = go.Figure(traces)

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
    return fig

def pointcloud_to_image(x_nm, y_nm, pixel_size_nm=5.0, padding_nm=0.0):
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
        cb_title = "log"#10(count+1)"
        zmin = 0
    else:
        Z = H.astype(float, copy=False)
        cb_title = "count"
        zmin = 0

    fig = go.Figure(
        data=go.Heatmap(
            z=Z.tolist(),                 # JSON-safe
            x=x_centers.tolist(),
            y=y_centers.tolist(),
            colorscale=lut,
            colorbar=dict(title=cb_title),
            zmin=zmin,
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
        if show_colorbar:
            marker["colorbar"] = dict(title="z")

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
            marker["colorbar"] = dict(title="end-to-end")

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
    fig.layout.shapes = tuple(s for s in (fig.layout.shapes or []) if s.get("name") != "scalebar")
    fig.layout.annotations = tuple(a for a in (fig.layout.annotations or []) if a.get("name") != "scalebar_label")

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
        colorscale.append([float(x), f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"])
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

        form.addWidget(QtWidgets.QLabel("Min trace length:"), 0, 0)
        form.addWidget(self.min_trace, 0, 1)
        form.addWidget(QtWidgets.QLabel("Z correction factor:"), 0, 2)
        form.addWidget(self.zcorr, 0, 3)

        form.addWidget(QtWidgets.QLabel("Scale factor:"), 1, 0)
        form.addWidget(self.scale, 1, 1)
        form.addWidget(QtWidgets.QLabel("EFO histogram bin size:"), 1, 2)
        form.addWidget(self.bin_size, 1, 3)

        self.scalebar_nm = QtWidgets.QDoubleSpinBox()
        self.scalebar_nm.setDecimals(1)
        self.scalebar_nm.setRange(1, 1e9)
        self.scalebar_nm.setSingleStep(10.0)

        form.addWidget(QtWidgets.QLabel("Scale bar size (nm):"), 2, 0)
        form.addWidget(self.scalebar_nm, 2, 1)
        form.addWidget(QtWidgets.QLabel(""), 2, 2)
        form.addWidget(QtWidgets.QLabel(""), 2, 3)

        note = QtWidgets.QLabel("Data is usually recorded in meters, set scale factor to 1e9 to convert to nanometers.")
        note.setWordWrap(True)
        # optional: make it look like a hint
        note.setStyleSheet("color: #B0B3B8;")  # or remove if you don't want styling

        # row 2, start at column 0, span 1 row x 4 columns
        form.addWidget(note, 3, 0, 1, 4)

        self.mbm_table = QtWidgets.QTableWidget(0, 3)
        self.mbm_table.setHorizontalHeaderLabels(["File", "MBM folder", ""])
        self.mbm_table.verticalHeader().setVisible(False)
        self.mbm_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.mbm_table.horizontalHeader().setStretchLastSection(False)
        self.mbm_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.mbm_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.mbm_table.setColumnWidth(2, 120)
        self.mbm_table.setMinimumHeight(180)   # adjust (e.g. 300–450)

        root.addWidget(QtWidgets.QLabel("MBM folders (per file):"))
        root.addWidget(self.mbm_table)

        # buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        root.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def set_values(self, *, min_trace_len, z_corr, scale, bin_size, scalebar_nm):
        self.min_trace.setValue(int(min_trace_len))
        self.zcorr.setValue(float(z_corr))
        self.scale.setValue(float(scale))
        self.bin_size.setValue(float(bin_size))
        self.scalebar_nm.setValue(float(scalebar_nm))

    def values(self):
        return dict(
            min_trace_len=int(self.min_trace.value()),
            z_corr=float(self.zcorr.value()),
            scale=float(self.scale.value()),
            bin_size=float(self.bin_size.value()),
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
    Table: File | XX (single selection) | LUT | Pixel size (nm)
    Shows heatmap for the selected file, using the SAVED array:
      - if MBM enabled -> aligned saved array if available
      - else -> saved array
    """
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Binning")
        self.resize(1100, 700)

        self._mw = main_window
        self._widgets_by_base = {}
        self._selected_base = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # top bar: table + plot
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, 2)

        # --- left: table ---
        left = QtWidgets.QVBoxLayout()
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

        self.table.setMinimumWidth(420)
        self.table.setMaximumWidth(520)
        left.addWidget(self.table, 1)

        # --- controls under table (global) ---
        ctrl = QtWidgets.QHBoxLayout()
        left.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("Scale:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["linear", "log10(count+1)"])
        self.scale_combo.setCurrentText("linear")  # DEFAULT = linear
        self.scale_combo.currentTextChanged.connect(lambda _: self.refresh_plot(keep_view=True))
        ctrl.addWidget(self.scale_combo)

        ctrl.addSpacing(12)

        ctrl.addWidget(QtWidgets.QLabel("Pixel size (nm):"))
        self.px_spin = QtWidgets.QDoubleSpinBox()
        self.px_spin.setDecimals(1)
        self.px_spin.setRange(0.1, 1e6)
        self.px_spin.setSingleStep(1.0)
        self.px_spin.setValue(5.0)
        self.px_spin.valueChanged.connect(lambda _: self.refresh_plot(keep_view=True))
        ctrl.addWidget(self.px_spin)

        ctrl.addStretch(1)

        # --- right: plot ---
        self.view = PlotlyView()
        top.addWidget(self.view, 3)

        # bottom buttons
        btnrow = QtWidgets.QHBoxLayout()
        root.addLayout(btnrow)
        btnrow.addStretch(1)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_plot)
        btnrow.addWidget(self.refresh_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btnrow.addWidget(self.close_btn)

    def rebuild(self, base_names):
        self.table.setRowCount(0)
        self._widgets_by_base.clear()

        for r, base in enumerate(base_names):
            self.table.insertRow(r)

            item = QtWidgets.QTableWidgetItem(base)
            self.table.setItem(r, 0, item)

            # XX checkbox (single selection)
            xx = QtWidgets.QCheckBox()
            xx.toggled.connect(lambda checked, b=base: self._on_xx_toggled(b, checked))
            xx_cell = QtWidgets.QWidget()
            xx_lay = QtWidgets.QHBoxLayout(xx_cell)
            xx_lay.setContentsMargins(0, 0, 0, 0)
            xx_lay.setAlignment(Qt.AlignCenter)
            xx_lay.addWidget(xx)
            self.table.setCellWidget(r, 1, xx_cell)

            # LUT combo
            lut = QtWidgets.QComboBox()
            lut.addItems(LUT_CHOICES)
            lut.setCurrentText(DEFAULT_LUT_BIN)
            lut.currentTextChanged.connect(lambda _, b=base: self._on_settings_changed(b))
            self.table.setCellWidget(r, 2, lut)

            self._widgets_by_base[base] = dict(xx=xx, lut=lut)

        self.table.resizeRowsToContents()

        # optionally auto-select first file if any
        if base_names and self._selected_base is None:
            first = base_names[0]
            self._select_base(first, do_refresh=True)

    def _on_settings_changed(self, base: str):
        if base == self._selected_base:
            self.refresh_plot(keep_view=True)

    def _select_base(self, base: str, do_refresh: bool):
        # enforce single selection
        for b, w in self._widgets_by_base.items():
            chk = w["xx"]
            chk.blockSignals(True)
            chk.setChecked(b == base)
            chk.blockSignals(False)

        self._selected_base = base
        if do_refresh:
            self.refresh_plot(keep_view=False)  # autorange for new file

    def _on_xx_toggled(self, base: str, checked: bool):
        if checked:
            self._select_base(base, do_refresh=True)
        else:
            if self._selected_base == base:
                self._selected_base = None
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No file selected", x=0.5, y=0.5, showarrow=False)])
                apply_plot_theme(fig, is3d=False)
                self.view.update_fig(fig, reset_view=True, is3d=False)

    def _on_zmax_checked(self, state: int):
            on = (state == Qt.Checked)
            self.zmax_spin.setEnabled(on)

            if on:
                # try to set a reasonable default based on current selection
                base, lut, px = self._get_selected_settings()
                if base:
                    arr = self._mw._filtered_by_base.get(base)
                    if getattr(self._mw, "_mbm_enabled", False):
                        arr = self._mw._aligned_arr_by_base.get(base, arr)
                    if arr is not None and len(arr):
                        loc = np.asarray(arr["loc"], dtype=float)
                        x = loc[:, 0]; y = loc[:, 1]
                        m = np.isfinite(x) & np.isfinite(y)
                        x = x[m]; y = y[m]
                        if len(x):
                            H, _ = pointcloud_to_image(x, y, pixel_size_nm=float(px), padding_nm=0.0)
                            if self.scale_combo.currentText() == "log":
                                Z = np.log10(H + 1.0)
                            else:
                                Z = H.astype(float, copy=False)
                            if self.minmax_chk.isChecked():
                                # normalized => zmax should be 1
                                self.zmax_spin.setValue(1.0)
                            else:
                                vmax = float(np.nanmax(Z)) if np.size(Z) else 1.0
                                self.zmax_spin.setValue(vmax)

            self.refresh_plot()

    def _get_selected_settings(self):
        base = self._selected_base
        if not base:
            return None, None
        w = self._widgets_by_base.get(base)
        if not w:
            return base, DEFAULT_LUT_BIN
        return base, w["lut"].currentText()

    def refresh_plot(self, keep_view: bool = True):
        base, lut = self._get_selected_settings()
        if not base:
            return

        # saved array (EFO-filtered), aligned if enabled
        arr = self._mw._filtered_by_base.get(base)
        if arr is None:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(
                text=f"No saved (EFO-filtered) data for: {base}<br>Apply EFO first.",
                x=0.5, y=0.5, showarrow=False
            )])
            apply_plot_theme(fig, is3d=False)
            self.view.update_fig(fig, reset_view=True, is3d=False)
            return

        if getattr(self._mw, "_mbm_enabled", False):
            arr = self._mw._aligned_arr_by_base.get(base, arr)

        # global settings
        pixel_size_nm = float(self.px_spin.value()) if hasattr(self, "px_spin") else 5.0
        scale_mode = self.scale_combo.currentText().strip().lower() if hasattr(self, "scale_combo") else "linear"

        fig = make_plotly_heatmap_from_arr(
            arr,
            pixel_size_nm=pixel_size_nm,
            lut=lut,
            title=None,
            scale_mode=scale_mode,
        )

        # keep current zoom if requested
        self.view.update_fig(
            fig,
            reset_view=not keep_view,
            is3d=False,
            scalebar_nm=getattr(self._mw, "_scalebar_nm", 100.0),
        )

class PlotSyncBridge(QtCore.QObject):
    viewChanged = Signal(str, object)  # (source_id, payload dict)

    @QtCore.Slot(str, "QVariant")
    def relayView(self, source_id, payload):
        # payload is a dict like {"mode":"2d","xRange":[...],"yRange":[...]} or {"mode":"3d","camera":{...}}
        self.viewChanged.emit(source_id, payload)


class MultiColorWindow(QtWidgets.QMainWindow):
    def __init__(self, on_view_changed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multicolor")
        self.resize(1200, 700)
        self._on_view_changed = on_view_changed

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        # --- stacked widget with plots (create ONCE) ---
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)

        # --- export button overlaid bottom-right ---
        self._export_btn = QtWidgets.QPushButton("Export (SVG/PDF)", central)
        self._export_btn.clicked.connect(self._export_current)
        self._export_btn.setFixedHeight(24)
        self._export_btn.setMaximumWidth(140)
        self._export_btn.raise_()

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
        mv.addWidget(self.merged_view, 1)

        self.stack.addWidget(self.merged_page)

        # state
        self._views = {}    # base -> PlotlyView
        self._mode = "grid" # or "merged"

    def set_mode(self, mode: str):
        mode = "merged" if mode == "merged" else "grid"
        self._mode = mode
        self.stack.setCurrentIndex(1 if mode == "merged" else 0)

    def _export_current(self):
        if self.mode() == "merged":
            self.merged_view.export_vector()
        else:
            # export the first grid view (or you can choose)
            if self._views:
                next(iter(self._views.values())).export_vector()

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
        view.update_fig(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def update_all(self, figs_by_base, reset_view=False, is3d=False, scalebar_nm=100.0):
        for base, fig in figs_by_base.items():
            self.update_one(base, fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_export_btn") and self._export_btn is not None:
            m = 12  # margin from edges
            btn = self._export_btn
            btn.adjustSize()
            x = self.centralWidget().width() - btn.width() - m
            y = self.centralWidget().height() - btn.height() - m
            btn.move(x, y)

    def update_merged(self, fig, reset_view=False, is3d=False, scalebar_nm=100.0):
        self.merged_view.update_fig(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

class PlotlyView(QtWidgets.QWidget):
    """Widget hosting Plotly in a QWebEngineView + emits view changes."""
    viewChanged = Signal(str, object)  # (source_id, payload)

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
        if (!xa || !ya || !xa.range || !ya.range) return;

        const xMin = xa.range[0], xMax = xa.range[1];
        const yMin = ya.range[0], yMax = ya.range[1];

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

    def export_vector(self):
        if self._last_fig is None:
            QtWidgets.QMessageBox.information(self, "Export", "No figure to export yet.")
            return

        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "plot.svg",
            "SVG (*.svg);;PDF (*.pdf)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext not in (".svg", ".pdf"):
            # infer from filter if user omitted extension
            if "PDF" in filt:
                path += ".pdf"
            else:
                path += ".svg"
            ext = os.path.splitext(path)[1].lower()

        fmt = ext.lstrip(".")  # "svg" or "pdf"

        try:
            pio.write_image(self._last_fig, path, format=fmt, scale=1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not export {fmt.upper()}.\n\n"
                f"Make sure kaleido is installed.\n\nError:\n{e}"
            )
            return

        QtWidgets.QMessageBox.information(self, "Export", f"Saved:\n{path}")

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

        const config = {{ scrollZoom: true, displaylogo: false, responsive: true }};

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

        MFX_Data = MFX_Data.copy()
        MFX_Data["loc"][:, -1] *= p["z_corr"]
        MFX_Data["loc"] *= p["scale"]

        last_itr = int(np.max(MFX_Data["itr"]))
        MFX_Data_vld_fnl = MFX_Data[MFX_Data["itr"] == last_itr]
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

        max_itr = np.max(MFX_Data_vld_fnl_filt["itr"])
        efo_vals = MFX_Data_vld_fnl_filt["efo"][MFX_Data_vld_fnl_filt["itr"] == max_itr]
        self._check_cancel()
        if len(efo_vals) == 0:
            self.status.emit(f"Skipping {base}: no EFO values.")
            return None

        ctx = dict(
            base=base,
            arr=MFX_Data_vld_fnl_filt,
            efo_vals=efo_vals,
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

        self._ctx_by_base = {}          # base -> ctx (from worker)
        self._efo_range_by_base = {}    # base -> (xmin, xmax) selected
        self._filtered_by_base = {}     # base -> array after EFO filter (what will be saved)

        # widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self._binning_win = None

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

        # --- Binning button (standalone, above multicolor section) ---
        VIEWER_W = 150
        BTN_H = 34

        def tune_btn(b, w):
            b.setFixedHeight(BTN_H)
            b.setFixedWidth(w)

        self.binning_btn = QtWidgets.QPushButton("Binning")
        self.binning_btn.clicked.connect(self.open_binning_window)
        tune_btn(self.binning_btn, VIEWER_W)
        right_controls.addWidget(self.binning_btn, 0, Qt.AlignHCenter)

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
        hist_box = QtWidgets.QGroupBox("EFO selection")
        bottom.addWidget(hist_box, 1)
        hv = QtWidgets.QVBoxLayout(hist_box)

        self.fig = Figure(figsize=(5, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        apply_hist_theme(self.fig, self.ax)
        self.canvas.draw()
        hv.addWidget(self.canvas)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.nav_toolbar.setIconSize(QtCore.QSize(16, 16))   # try 12, 14, 16
        self.nav_toolbar.setFixedHeight(28)                  # try 24–30
        hv.addWidget(self.nav_toolbar)

        ctrl = QtWidgets.QHBoxLayout()
        hv.addLayout(ctrl)
        self.range_lbl = QtWidgets.QLabel("Selected range: —")
        ctrl.addWidget(self.range_lbl, 1)

        self.apply_btn = QtWidgets.QPushButton("Apply EFO")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_preview)
        ctrl.addWidget(self.apply_btn)

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

        self._span = None
        self._span_xmin = None
        self._span_xmax = None

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
        self._filtered_by_base.clear()

        self._raw_arr_by_base.clear()
        self._arr_by_base.clear()
        self._aligned_arr_by_base.clear()
        self._T_by_base.clear()

        self._mbm_enabled = False
        self._mbm_source_base = None

        # reset UI controls
        self.apply_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)

        self.range_lbl.setText("Selected range: —")
        self.set_output([])

        # clear histogram
        try:
            self.ax.clear()
            apply_hist_theme(self.fig, self.ax)
            self.canvas.draw()
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
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )

        dlg.set_mbm_rows(self._all_files)   # <-- required
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._mbm_dir_by_base = dlg.mbm_map()
            vals = dlg.values()
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
            self._multicolor_win = MultiColorWindow(self.on_any_plot_view_changed, self)

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

        is3d = self.is3d.isChecked()
        avg = self.avg_tid.isChecked()
        sb = float(getattr(self, "_scalebar_nm", 100.0))
        if self._multicolor_win.mode() == "merged":
            # Build merged from whatever is in _arr_by_base (trace-filtered or EFO-filtered)
            # preserve insertion order from _all_files:
            arr_by_base = {}
            for f in self._all_files:
                base = os.path.splitext(os.path.basename(f))[0]
                arr = self._get_arr_for_base(base)
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
                arr = self._get_arr_for_base(base)
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

    def on_any_plot_view_changed(self, source_id, payload):
        if isinstance(payload, dict) and payload.get("mode") == "3d" and "camera" in payload:
            payload = dict(payload)
            payload["camera"] = _camera_slim(payload["camera"])

        # main plot
        if hasattr(self, "plot_view") and self.plot_view._id != source_id:
            self.plot_view.apply_view(payload)

        # multicolor
        if self._multicolor_win is not None and self._multicolor_win.isVisible():
            # merged view
            if self._multicolor_win.merged_view._id != source_id:
                self._multicolor_win.merged_view.apply_view(payload)

            # grid views
            for base, view in self._multicolor_win._views.items():
                if view._id == source_id:
                    continue
                view.apply_view(payload)

    def _multicolor_update_base(self, base, reset_view=False):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        arr = self._arr_by_base.get(base)
        if arr is None:
            return
        cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})
        fig = make_plotly_fig(arr, self.avg_tid.isChecked(), self.is3d.isChecked(), color_settings=cs)
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

    def update_plotly_fig(self, fig, reset_view=False):
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

        # cancel worker if running
        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
                self._current_worker.wait(1000)  # wait up to 1s
        except Exception:
            pass

        # close multicolor window if open
        try:
            if self._multicolor_win is not None:
                self._multicolor_win.close()
        except Exception:
            pass
        try:
            if getattr(self, "_binning_win", None) is not None:
                self._binning_win.close()
        except Exception:
            pass

        event.accept()
        QtCore.QTimer.singleShot(1000, lambda: os._exit(0))
        
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
            ("After EFO filtering", str(n_after)),
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

            if base in self._efo_range_by_base:
                xmin, xmax = self._efo_range_by_base[base]
                self._apply_efo_range_to_memory(base, xmin, xmax, update_view=True, reset_view=True)
            return

        # else: load it
        if self.run_btn.isEnabled() is False:
            self.start_worker_for_current()

    def _apply_efo_range_to_memory(self, base, xmin, xmax, *, update_view=True, reset_view=False):
        arr = self._raw_arr_by_base.get(base)
        if arr is None:
            return

        mask = (arr["efo"] >= xmin) & (arr["efo"] <= xmax)
        arr_efo = arr[mask]

        self._efo_range_by_base[base] = (float(xmin), float(xmax))
        self._filtered_by_base[base] = arr_efo

        # default plotted data is the filtered data (or aligned filtered if enabled)
        if self._mbm_enabled and base in self._aligned_arr_by_base:
            self._arr_by_base[base] = self._aligned_arr_by_base[base]
        else:
            self._arr_by_base[base] = arr_efo

        if update_view:
            self.redraw_scatter(reset_view=reset_view)
            self._refresh_multicolor_contents(reset_view=False)
            self._update_output_for_base(base, arr_efo, int(np.count_nonzero(mask)))

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

    # ---------------- EFO selection ----------------
    def on_need_efo(self, ctx):
        """
        Called by FileWorker when a file is loaded and trace-filtered and the GUI should
        display the EFO histogram + initial scatter preview.

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
        bin_size = float(self.bin_size.value())

        # Ensure caches exist
        if not hasattr(self, "_raw_arr_by_base") or self._raw_arr_by_base is None:
            self._raw_arr_by_base = {}
        if not hasattr(self, "_arr_by_base") or self._arr_by_base is None:
            self._arr_by_base = {}

        # ---- cache as "raw" (trace-filtered) for this base ----
        self._raw_arr_by_base[base] = arr
        self._arr_by_base[base] = arr

        # ---- histogram ----
        self.ax.clear()
        if len(efo_vals) > 0:
            bins = np.arange(efo_vals.min(), efo_vals.max() + bin_size, bin_size)
            self.ax.hist(
                efo_vals,
                bins=bins,
                edgecolor=HIST_THEME["hist_edge"],
                color=HIST_THEME["hist_face"],
            )

        self.ax.set_xlabel("EFO")
        self.ax.set_ylabel("Count")
        self.ax.set_title(base)

        apply_hist_theme(self.fig, self.ax)
        self.canvas.draw()

        if len(efo_vals) > 0:
            if base in self._efo_range_by_base:
                self._span_xmin, self._span_xmax = self._efo_range_by_base[base]
            else:
                self._span_xmin = float(efo_vals.min())
                self._span_xmax = float(efo_vals.max())
        else:
            self._span_xmin = None
            self._span_xmax = None


        def onselect(xmin, xmax):
            self._span_xmin, self._span_xmax = float(xmin), float(xmax)
            self.range_lbl.setText(f"Selected range: {self._span_xmin:.2f} ... {self._span_xmax:.2f}")

        if self._span is not None:
            self._span.disconnect_events()

        if self._span_xmin is not None and self._span_xmax is not None:
            self._span = SpanSelector(
                self.ax, onselect, direction="horizontal", useblit=True,
                props=dict(
                    facecolor=HIST_THEME["span_face"],
                    alpha=HIST_THEME["span_alpha"],
                    edgecolor=HIST_THEME["span_edge"],
                    linewidth=HIST_THEME["span_lw"],
                ),
                interactive=True, drag_from_anywhere=True
            )
            self._span.extents = (self._span_xmin, self._span_xmax)
            onselect(self._span_xmin, self._span_xmax)
        else:
            self._span = None
            self.range_lbl.setText("Selected range: —")

        self.apply_btn.setEnabled(True)

        # ---- cache ctx for later "Save all" ----
        self._ctx_by_base[base] = ctx

        # ---- apply existing range if present; otherwise default to full range (once) ----
        if len(efo_vals) > 0:
            if base in self._efo_range_by_base:
                xmin, xmax = self._efo_range_by_base[base]
            else:
                xmin, xmax = float(efo_vals.min()), float(efo_vals.max())
                self._efo_range_by_base[base] = (xmin, xmax)

            # apply (this updates _filtered_by_base + plots)
            self._apply_efo_range_to_memory(
                base,
                xmin,
                xmax,
                update_view=True,
                reset_view=True
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
            # just show raw (trace-filtered)
            self._arr_by_base[base] = arr

            # initial scatter - reset view for new file
            self.redraw_scatter(reset_view=True)

            # update multicolor for this base and any open multicolor view
            self._multicolor_update_base(base, reset_view=True)
            self._refresh_multicolor_contents(reset_view=False)

        

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
        if not base or self._span_xmin is None or self._span_xmax is None:
            return
        self._apply_efo_range_to_memory(base, self._span_xmin, self._span_xmax, update_view=True, reset_view=False)

        # if alignment is enabled, recompute it on the *filtered* datasets
        if self._mbm_enabled:
            self._apply_mbm_alignment()
        if self._binning_win is not None and self._binning_win.isVisible():
            self._binning_win.refresh_plot()

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
                T, common = compute_mbm_transform(
                    mbm_src, mbm_mov, k=4, scale=float(self.scale.value())
                )
                self._T_by_base[base] = T
                self._aligned_arr_by_base[base] = apply_transform_to_arr(arr, T)
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
    main()