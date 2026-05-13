import hashlib

import matplotlib
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


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


def make_plotly_fig(arr, avg_tid: bool, is3d: bool, color_settings: dict = None, scalebar_nm=100.0):
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


def make_plotly_fig_merged(arr_by_base: dict, avg_tid: bool, is3d: bool, color_settings_by_base: dict, scalebar_nm=100):
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

    xmin -= padding_nm
    xmax += padding_nm
    ymin -= padding_nm
    ymax += padding_nm

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
    scale_mode: str = "linear",  # "linear" (default) or "log"
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
            z=Z.tolist(),  # JSON-safe
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
            opacity=1.0,  # alpha already baked into rgba(); keep opacity=1 to avoid double-multiplying
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
            customdata = z.astype(float).tolist()  # 1D list
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


def custom_LUT(colors=["#000000", "#ff0000"], bins=2**8):
    """
    colors: list of colors that define the color scale
    bins: range
    """

    cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=bins
    )
    xs = np.linspace(0, 1, bins)
    rgba = cm(xs)  # (n,4) floats in [0,1]
    colorscale = []
    for x, (r, g, b, a) in zip(xs, rgba):
        rr = int(round(r * 255))
        gg = int(round(g * 255))
        bb = int(round(b * 255))
        colorscale.append([float(x), f"#{rr:02x}{gg:02x}{bb:02x}"])
    return colorscale


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

LUT_CHOICES = [
    "turbo", "viridis", "cividis", "inferno", "magma", "plasma",
    "electric", "hot", "hsv", "jet", "rainbow", "twilight", "icefire",
    "piyg", "brbg", "rdbu", "brwnyl", "reds", "rdpu", "ylgn", "ylorbr",
    "thermal", "gray", "ice", "algae", "speed", "temps",
]  # Plotly colorscale names
LUT_CHOICES = sorted(LUT_CHOICES + list(CUSTOM_LUTS.keys()))

DEFAULT_SOLID = "darkgreen"
DEFAULT_LUT = "twilight"
DEFAULT_LUT_BIN = "hot"
DEFAULT_MODE = "depth"

# -------------------- Plot theme --------------------

PLOT_THEME = {
    # overall
    "template": "none",  # "none" to rely on your explicit colors; or "plotly_dark"/"plotly_white"
    "font_family": "Arial",
    "font_size": 12,
    "font_color": "#E6E6E6",

    # backgrounds
    "paper_bg": "#202122",  # outside plotting area
    "plot_bg": "#202122",  # 2D plotting area
    "scene_bg": "#202122",  # 3D scene background

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
    "legend_bg": "rgba(0,0,0,0)",  # transparent
    "legend_border": "#666666",

    # margins
    "margin": dict(l=0, r=0, t=20, b=0),
}

DEFAULT_SCALEBAR_LENGTH_NM = 100.0
SCALEBAR_LENGTH_NM = DEFAULT_SCALEBAR_LENGTH_NM
SCALEBAR_MARGIN_FRACTION = 0.05  # distance from lower-left as fraction of current view range
SCALEBAR_LINE_WIDTH = 4
SCALEBAR_COLOR = "#E6E6E6"

PLOTLY_HTML_BG = PLOT_THEME["paper_bg"]  # or PLOT_THEME["plot_bg"]

# -------------------- Matplotlib theme (histogram) --------------------

HIST_THEME = {
    "fig_bg": "#202122",  # matches PLOT_THEME["paper_bg"]
    "ax_bg": "#202122",  # matches PLOT_THEME["plot_bg"]
    "text": "#E6E6E6",
    "grid": "#2F2F2F",
    "spines": "#8A8A8A",
    "ticks": "#CFCFCF",

    # histogram style
    "hist_face": "#b8b8ff",
    "hist_edge": "#b8b8ff",

    # selection span style (SpanSelector)
    "span_face": "#f8f7ff",
    "span_alpha": 0.15,
    "span_edge": "black",
    "span_lw": 2,
}
