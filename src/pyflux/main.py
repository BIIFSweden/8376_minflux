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

# -------------------- helpers --------------------
def save_to_csv(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)



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


def make_plotly_fig(arr, avg_tid: bool, is3d: bool, color_settings: dict = None):
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


def make_plotly_fig_merged(arr_by_base: dict, avg_tid: bool, is3d: bool, color_settings_by_base: dict):
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
                y=1.02,
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
            customdata = z.reshape(-1, 1)
            hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{customdata[0]:.3f}<br>%{text}<extra></extra>"
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

# -------------------- Qt (GUI) theme --------------------

GUI_THEME = {
    "bg": "#202122",          # main window background
    "panel_bg": "#25272A",    # groupboxes / panels
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
DEFAULT_SIZE_2D = 6
DEFAULT_SIZE_3D = 6

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

LUT_CHOICES = ["turbo", "viridis", "cividis", "inferno", "magma", "plasma", 
               "electric", "hot", "hsv", "jet", "rainbow", "twilight", "icefire", 
               "piyg", "brbg", "rdbu", "brwnyl", "reds", "rdpu", "ylgn", "ylorbr", 
               "thermal","gray", "ice", "algae", "speed", "temps",
               ]  # Plotly colorscale names
LUT_CHOICES = sorted(LUT_CHOICES)

DEFAULT_SOLID = "darkgreen"
DEFAULT_LUT = "twilight"
DEFAULT_MODE = "depth"

# -------------------- Plot theme --------------------

PLOT_THEME = {
    # overall
    "template": "none",                 # "none" to rely on your explicit colors; or "plotly_dark"/"plotly_white"
    "font_family": "Arial",
    "font_size": 12,
    "font_color": "#E6E6E6",

    # backgrounds
    "paper_bg": "#1E1E1E",              # outside plotting area
    "plot_bg":  "#1E1E1E",              # 2D plotting area
    "scene_bg": "#1E1E1E",              # 3D scene background

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

PLOTLY_HTML_BG = PLOT_THEME["paper_bg"]   # or PLOT_THEME["plot_bg"]

# -------------------- Matplotlib theme (histogram) --------------------

HIST_THEME = {
    "fig_bg":   "#1E1E1E",   # matches PLOT_THEME["paper_bg"]
    "ax_bg":    "#1E1E1E",   # matches PLOT_THEME["plot_bg"]
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
    Embedded color settings:
      base : [mode combo] [palette combo] alpha: [0..1] size: [int]

    mode:
      - "solid"                 -> palette is SOLID_COLOR_CHOICES
      - "end-to-end"   -> palette is LUT_CHOICES
      - "depth"                 -> palette is LUT_CHOICES

    Emits changed(base, settings_dict).
    settings_dict keys:
      - mode: str
      - solid: str
      - lut: str
      - alpha: float
      - size: int
    """
    changed = Signal(str, object)  # (base, settings dict)

    def __init__(self, title="Color settings", parent=None):
        super().__init__(title, parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        self.inner = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.inner)
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.form.setFormAlignment(Qt.AlignTop)
        self.form.setVerticalSpacing(8)
        self.scroll.setWidget(self.inner)

        # base -> (mode_combo, palette_combo, alpha_spin, size_spin)
        self._widgets_by_base = {}

        # show ~3 rows before scrolling; adjust to taste
        self.setMinimumHeight(160)
        self.setMaximumHeight(220)

    def _make_row_widget(self, base: str, settings: dict):
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItems(MODE_CHOICES)

        palette_combo = QtWidgets.QComboBox()

        alpha_lbl = QtWidgets.QLabel("\u03B1:")  # Greek letter alpha
        alpha_spin = QtWidgets.QDoubleSpinBox()
        alpha_spin.setRange(0.0, 1.0)
        alpha_spin.setDecimals(2)
        alpha_spin.setSingleStep(0.05)

        size_lbl = QtWidgets.QLabel("size:")
        size_spin = QtWidgets.QSpinBox()
        size_spin.setRange(1, 50)  # adjust if you want larger
        size_spin.setSingleStep(1)

        # ---- apply initial settings ----
        s = settings or {}
        mode = s.get("mode", DEFAULT_MODE)
        if mode not in MODE_CHOICES:
            mode = DEFAULT_MODE

        solid = s.get("solid", DEFAULT_SOLID)
        if solid not in SOLID_COLOR_CHOICES:
            solid = DEFAULT_SOLID

        lut = s.get("lut", DEFAULT_LUT)
        if lut not in LUT_CHOICES:
            lut = DEFAULT_LUT

        alpha = float(s.get("alpha", DEFAULT_ALPHA))
        alpha = max(0.0, min(1.0, alpha))

        size = int(s.get("size", DEFAULT_SIZE_2D))
        size = max(1, min(50, size))

        mode_combo.setCurrentText(mode)
        alpha_spin.setValue(alpha)
        size_spin.setValue(size)

        def fill_palette_for_mode(m: str):
            palette_combo.blockSignals(True)
            palette_combo.clear()

            if m == MODE_SOLID:
                palette_combo.setEnabled(True)
                palette_combo.addItems(SOLID_COLOR_CHOICES)
                palette_combo.setCurrentText(solid if solid in SOLID_COLOR_CHOICES else DEFAULT_SOLID)

            elif m == MODE_TID:
                palette_combo.setEnabled(False)
                palette_combo.addItem("—")

            else:  # MODE_DEPTH or MODE_E2E
                palette_combo.setEnabled(True)
                palette_combo.addItems(LUT_CHOICES)
                palette_combo.setCurrentText(lut if lut in LUT_CHOICES else DEFAULT_LUT)

            palette_combo.blockSignals(False)

        fill_palette_for_mode(mode)

        def emit_changed():
            m = mode_combo.currentText()
            a = float(alpha_spin.value())
            sz = int(size_spin.value())

            if m == MODE_SOLID:
                payload = {
                    "mode": MODE_SOLID,
                    "solid": palette_combo.currentText(),
                    "lut": DEFAULT_LUT,
                    "alpha": a,
                    "size": sz,
                }
            elif m == MODE_DEPTH:
                payload = {
                    "mode": MODE_DEPTH,
                    "solid": DEFAULT_SOLID,
                    "lut": palette_combo.currentText(),
                    "alpha": a,
                    "size": sz,
                }
            elif m == MODE_TID:
                payload = {
                    "mode": MODE_TID,
                    "solid": DEFAULT_SOLID,
                    "lut": DEFAULT_LUT,   # ignored
                    "alpha": a,
                    "size": sz,
                }
            else:  # MODE_E2E
                payload = {
                    "mode": MODE_E2E,
                    "solid": DEFAULT_SOLID,
                    "lut": palette_combo.currentText(),
                    "alpha": a,
                    "size": sz,
                }

            self.changed.emit(base, payload)

        def on_mode_changed(m: str):
            fill_palette_for_mode(m)
            emit_changed()

        # ---- wire signals ----
        mode_combo.currentTextChanged.connect(on_mode_changed)
        palette_combo.currentTextChanged.connect(lambda _: emit_changed())
        alpha_spin.valueChanged.connect(lambda _: emit_changed())
        size_spin.valueChanged.connect(lambda _: emit_changed())

        # ---- layout row ----
        h.addWidget(mode_combo, 1)
        h.addWidget(palette_combo, 2)
        h.addSpacing(4)
        h.addWidget(alpha_lbl)
        h.addWidget(alpha_spin)
        h.addSpacing(4)
        h.addWidget(size_lbl)
        h.addWidget(size_spin)

        self._widgets_by_base[base] = (mode_combo, palette_combo, alpha_spin, size_spin)
        return row

    def rebuild(self, base_names, current_settings_by_base: dict):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets_by_base.clear()

        for base in base_names:
            settings = (current_settings_by_base or {}).get(base, None)
            row_widget = self._make_row_widget(base, settings)
            self.form.addRow(f"{base}:", row_widget)

    def set_settings(self, base: str, settings: dict):
        pair = self._widgets_by_base.get(base)
        if not pair:
            return

        mode_combo, palette_combo, alpha_spin, size_spin = pair
        s = settings or {}

        # normalize
        mode = s.get("mode", DEFAULT_MODE)
        if mode not in MODE_CHOICES:
            mode = DEFAULT_MODE

        solid = s.get("solid", DEFAULT_SOLID)
        if solid not in SOLID_COLOR_CHOICES:
            solid = DEFAULT_SOLID

        lut = s.get("lut", DEFAULT_LUT)
        if lut not in LUT_CHOICES:
            lut = DEFAULT_LUT

        alpha = float(s.get("alpha", DEFAULT_ALPHA))
        alpha = max(0.0, min(1.0, alpha))

        size = int(s.get("size", DEFAULT_SIZE_2D))
        size = max(1, min(50, size))

        # set mode first (this controls palette content)
        mode_combo.blockSignals(True)
        mode_combo.setCurrentText(mode)
        mode_combo.blockSignals(False)

        # set palette according to mode (solid vs LUT vs disabled for tid)
        palette_combo.blockSignals(True)
        palette_combo.clear()

        if mode == "solid":
            palette_combo.setEnabled(True)
            palette_combo.addItems(SOLID_COLOR_CHOICES)
            palette_combo.setCurrentText(solid if solid in SOLID_COLOR_CHOICES else DEFAULT_SOLID)

        elif mode == "tid":
            palette_combo.setEnabled(False)
            palette_combo.addItem("—")

        else:
            # "depth" or "end-to-end"
            palette_combo.setEnabled(True)
            palette_combo.addItems(LUT_CHOICES)
            palette_combo.setCurrentText(lut if lut in LUT_CHOICES else DEFAULT_LUT)

        palette_combo.blockSignals(False)

        # alpha/size
        alpha_spin.blockSignals(True)
        alpha_spin.setValue(alpha)
        alpha_spin.blockSignals(False)

        size_spin.blockSignals(True)
        size_spin.setValue(size)
        size_spin.blockSignals(False)


class ParametersDialog(QtWidgets.QDialog):
    """Dialog to edit analysis parameters, bound to MainWindow spinboxes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.setModal(True)
        self.resize(520, 180)

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

        note = QtWidgets.QLabel("Data is usually recorded in meters, set scale factor to 1e9 to convert to nanometers.")
        note.setWordWrap(True)
        # optional: make it look like a hint
        note.setStyleSheet("color: #B0B3B8;")  # or remove if you don't want styling

        # row 2, start at column 0, span 1 row x 4 columns
        form.addWidget(note, 2, 0, 1, 4)

        # buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        root.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def set_values(self, *, min_trace_len, z_corr, scale, bin_size):
        self.min_trace.setValue(int(min_trace_len))
        self.zcorr.setValue(float(z_corr))
        self.scale.setValue(float(scale))
        self.bin_size.setValue(float(bin_size))

    def values(self):
        return dict(
            min_trace_len=int(self.min_trace.value()),
            z_corr=float(self.zcorr.value()),
            scale=float(self.scale.value()),
            bin_size=float(self.bin_size.value()),
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

        # Stacked: page 0 = grid, page 1 = merged
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)

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
        self._views = {}   # base -> PlotlyView
        self._mode = "grid"  # or "merged"

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
            vbox.addWidget(view, 1)

            self._views[base] = view
            self.grid.addWidget(box, r, c)

            c += 1
            if c >= cols:
                c = 0
                r += 1

    def update_one(self, base, fig, reset_view=False, is3d=False):
        view = self._views.get(base)
        if view is None:
            return
        view.update_fig(fig, reset_view=reset_view, is3d=is3d)

    def update_all(self, figs_by_base, reset_view=False, is3d=False):
        for base, fig in figs_by_base.items():
            self.update_one(base, fig, reset_view=reset_view, is3d=is3d)

    def update_merged(self, fig, reset_view=False, is3d=False):
        self.merged_view.update_fig(fig, reset_view=reset_view, is3d=is3d)

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


        self.web.page().setBackgroundColor(QColor(PLOTLY_HTML_BG))
        self.ensure_page()

    def _bootstrap_html(self):
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
      if (window._plotSync.ignore) return;

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

    def update_fig(self, fig, reset_view=False, is3d=False):
        self.ensure_page()
        if not self._plotly_ready:
            self._pending_fig = (fig, reset_view, is3d)
            return

        fig_json = pio.to_json(fig, validate=False)
        source_id = self._id

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
        self._check_cancel()
        # wait for range or cancel
        # in FileWorker._process_one_file, replace the "wait for range or cancel" block with:

        self._mutex.lock()
        try:
            while (not self._cancelled) and (self._chosen_range is None):
                if self.isInterruptionRequested():
                    self._cancelled = True
                    break
                # timed wait so we can re-check interruption regularly
                self._wait.wait(self._mutex, 100)  # 100 ms
            chosen = self._chosen_range
            cancelled = self._cancelled
        finally:
            self._mutex.unlock()

        if cancelled or chosen is None:
            self.status.emit(f"Cancelled/Skipped: {base}")
            return None
        xmin, xmax = chosen

        MFX_filtered = MFX_Data_vld_fnl_filt[
            (MFX_Data_vld_fnl_filt["efo"] >= xmin) & (MFX_Data_vld_fnl_filt["efo"] <= xmax)
        ]
        df = np_to_df(MFX_filtered)

        dims = [c.replace("loc_", "") for c in df.columns if c.startswith("loc_")]
        avg_df = pd.DataFrame()
        if len(df) > 0 and dims:
            for dim in dims:
                avg_df[f"loc_{dim}_mean"] = df.groupby("tid")[f"loc_{dim}"].mean()
                avg_df[f"loc_{dim}_std"] = df.groupby("tid")[f"loc_{dim}"].std()
            avg_df["n"] = df.groupby("tid")[f"loc_{dims[0]}"].count()
            avg_df["tim_tot"] = df.groupby("tid")["tim"].sum()

        loc_prec = None
        if len(avg_df) > 0 and dims:
            meds = []
            for dim in dims:
                s = avg_df[f"loc_{dim}_std"].dropna()
                meds.append(float(s.median()) if len(s) else float("nan"))
            loc_prec = tuple(float(f"{v:.2f}") for v in meds)
            ratio_loc_per_trace = avg_df["n"].sum() / len(avg_df)
            avg_df['ratio_loc_per_trace'] = ""
            if len(avg_df) > 0:
                avg_df.iloc[0, avg_df.columns.get_loc("ratio_loc_per_trace")] = f"{ratio_loc_per_trace:.6f}"

        # add loc prec to avg_df
        if loc_prec is not None:
            for i, dim in enumerate(dims):
                avg_df[f"loc_{dim}_prec"] = ""
                if len(avg_df) > 0:
                    avg_df.iloc[0, avg_df.columns.get_loc(f"loc_{dim}_prec")] = f"{loc_prec[i]:.4f}"
        
        save_path = os.path.join(p["save_folder"], f"{base}.csv")
        avg_save_path = os.path.join(p["save_folder"], f"{base}_stats.csv")
        save_to_csv(df, save_path)
        avg_df.to_csv(avg_save_path, index=True)

        items = [
            ("File", base),
            ("Total imaging time (min)", f"{total_tim/60:.2f}"),
            #("Total raw localizations", str(total_loc)),
            ("Last iteration localizations", str(last_iteration_loc)),
            ("After trace filtering", str(after_trace)),
            ("After EFO filtering", str(len(df))),
            ("Localization precision (x, y, z)", str(loc_prec) if loc_prec is not None else "—"),
            ("Ratio loc per trace", f"{ratio_loc_per_trace:.4f}"),
            ("% remaining", f"{(len(df)/last_iteration_loc*100):.2f}%"),
            ("Saved filtered CSV", save_path),
            ("Saved stats CSV", avg_save_path),
        ]
        self.status.emit(f"Saved: {base}")
        return dict(display_name=base, items=items, ctx=ctx, chosen=chosen)

# -------------------- main window --------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MINFLUX Filter GUI (PyQt + Matplotlib + Plotly)")
        self.resize(1200, 780)
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
        # widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

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

        self.multi_btn = QtWidgets.QPushButton("multicolor")
        self.multi_btn.clicked.connect(self.open_multicolor)
        # make button smaller
        self.multi_btn.setMaximumWidth(110)
        self.multi_btn.setMinimumHeight(22)
        right_controls.addWidget(self.multi_btn)

        self.merged_chk = QtWidgets.QCheckBox("merged")
        self.merged_chk.setChecked(False)
        self.merged_chk.stateChanged.connect(self.on_merged_toggled)
        right_controls.addWidget(self.merged_chk)

        self.avg_tid = QtWidgets.QCheckBox("avg loc (tid)")
        self.avg_tid.stateChanged.connect(lambda _: self._refresh_all_plots_same_data())
        right_controls.addWidget(self.avg_tid)

        self.is3d = QtWidgets.QCheckBox("3D")
        self.is3d.stateChanged.connect(lambda _: self._refresh_all_plots_same_data())
        right_controls.addWidget(self.is3d)

        right_controls.addStretch(1)

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
        
        self.continue_btn = QtWidgets.QPushButton("Save && Continue")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(self.continue_file)
        ctrl.addWidget(self.continue_btn)

        self._span = None
        self._span_xmin = None
        self._span_xmax = None

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

    def _on_plotly_load_finished(self, ok: bool):
        self._plotly_ready = bool(ok)
        if ok and self._pending_fig is not None:
            fig = self._pending_fig
            self._pending_fig = None
            self.update_plotly_fig(fig)


    def on_color_settings_changed(self, base: str, settings: dict):
        if not isinstance(settings, dict):
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
        dlg = ParametersDialog(self)
        dlg.set_values(
            min_trace_len=self.min_trace.value(),
            z_corr=self.zcorr.value(),
            scale=self.scale.value(),
            bin_size=self.bin_size.value(),
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()
            self.min_trace.setValue(vals["min_trace_len"])
            self.zcorr.setValue(vals["z_corr"])
            self.scale.setValue(vals["scale"])
            self.bin_size.setValue(vals["bin_size"])

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
                color_settings_by_base=self._color_settings_by_base
            )
            self._multicolor_win.update_merged(fig, reset_view=reset_view, is3d=is3d)
        else:
            figs = {}
            for f in self._all_files:
                base = os.path.splitext(os.path.basename(f))[0]
                cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})
                arr = self._get_arr_for_base(base)
                if arr is None:
                    continue
                figs[base] = make_plotly_fig(arr, avg, is3d, color_settings=cs)

            self._multicolor_win.update_all(figs, reset_view=reset_view, is3d=is3d)

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
        self._multicolor_win.update_one(base, fig, reset_view=reset_view, is3d=self.is3d.isChecked())

    def _multicolor_rebuild_if_open(self):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        bases = [os.path.splitext(os.path.basename(f))[0] for f in self._all_files]
        self._multicolor_win.rebuild(bases)

    def update_plotly_fig(self, fig, reset_view=False):
        self.plot_view.update_fig(fig, reset_view=reset_view, is3d=self.is3d.isChecked())

    # ---------------- UI actions ----------------
    def browse_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select MINFLUX data folder")
        if path:
            self.data_edit.setText(path)
            self.set_default_output()
            self.refresh_files()

    def set_default_output(self):
        p = self.data_edit.text().strip()
        if p:
            self.out_edit.setText(p + "_filtered")

    def closeEvent(self, event):
        print("closeEvent called")
        self._closing = True

        # stop/cancel any running workers - with safe check
        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
        except RuntimeError:
            # C++ object already deleted
            self._current_worker = None

        # give Qt/WebEngine a moment to shut down cleanly, then hard-exit
        event.ignore()
        QtCore.QTimer.singleShot(1500, lambda: os._exit(0))
        
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


    def on_file_changed(self, idx):
        if getattr(self, "_closing", False):
            return
        if idx < 0 or idx >= len(self._all_files):
            return

        # Safe check - worker might be deleted
        try:
            worker_running = (
                self._current_worker is not None 
                and self._current_worker.isRunning()
            )
        except RuntimeError:
            # C++ object deleted
            self._current_worker = None
            worker_running = False

        if worker_running:
            self._current_worker.cancel()
            # Wait for current worker to finish before switching
            # Optionally queue the switch for after cancellation completes
            return
        
        self._current_index = idx

        if self.run_btn.isEnabled() is False:
            # if session running, start loading this file
            self.start_worker_for_current()

    def run_start(self):
        data_path = self.data_edit.text().strip()
        save_folder = self.out_edit.text().strip()
        if not data_path or not os.path.isdir(data_path):
            QtWidgets.QMessageBox.critical(self, "Error", "Please select a valid MINFLUX data folder.")
            return
        if not save_folder:
            QtWidgets.QMessageBox.critical(self, "Error", "Please set an output folder.")
            return
        os.makedirs(save_folder, exist_ok=True)

        if not self._all_files:
            self.refresh_files()
        if not self._all_files:
            QtWidgets.QMessageBox.critical(self, "Error", "No .npy files found.")
            return

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
        self.continue_btn.setEnabled(False)

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
        self._plot_arr = ctx["arr"]          # start from trace-filtered data
        self._plot_is_filtered = False
        self._current_ctx = ctx
        efo_vals = np.asarray(ctx["efo_vals"])
        base = ctx["base"]
        bin_size = float(self.bin_size.value())

        # histogram
        self.ax.clear()
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

        self._span_xmin = float(efo_vals.min())
        self._span_xmax = float(efo_vals.max())

        def onselect(xmin, xmax):
            self._span_xmin, self._span_xmax = float(xmin), float(xmax)
            self.range_lbl.setText(f"Selected range: {self._span_xmin:.2f} ... {self._span_xmax:.2f}")

        if self._span is not None:
            self._span.disconnect_events()

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

        self.apply_btn.setEnabled(True)
        self.continue_btn.setEnabled(False)

        # initial scatter - reset view for new file
        self.redraw_scatter(reset_view=True)

        base = ctx["base"]
        self._arr_by_base[base] = ctx["arr"]  # trace-filtered (initial)
        self._multicolor_update_base(base, reset_view=True)
        self._refresh_multicolor_contents(reset_view=False)

    def redraw_scatter(self, reset_view=False):
        if self._current_ctx is None:
            return
        if self._plot_arr is None:
            self._plot_arr = self._current_ctx["arr"]
            self._plot_is_filtered = False
        base = self._current_ctx["base"] if self._current_ctx else None
        cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})
        fig = make_plotly_fig(self._plot_arr, self.avg_tid.isChecked(), self.is3d.isChecked(), color_settings=cs)
        self.update_plotly_fig(fig, reset_view=reset_view)

    def apply_preview(self):
        if self._current_ctx is None:
            return
        arr = self._current_ctx["arr"]
        xmin, xmax = self._span_xmin, self._span_xmax
        mask = (arr["efo"] >= xmin) & (arr["efo"] <= xmax)
        arr_efo = arr[mask]

        base = self._current_ctx["base"] if self._current_ctx else None
        cs = self._color_settings_by_base.get(base, {"mode": DEFAULT_MODE, "solid": DEFAULT_SOLID, "lut": DEFAULT_LUT, "alpha": DEFAULT_ALPHA, "size": DEFAULT_SIZE_2D})

        self._arr_by_base[base] = arr_efo  # now filtered
        self._multicolor_update_base(base, reset_view=False)
        self._refresh_multicolor_contents(reset_view=False)
        # store as current plot data
        self._plot_arr = arr_efo
        self._plot_is_filtered = True

        # update scatter
        fig = make_plotly_fig(self._plot_arr, self.avg_tid.isChecked(), self.is3d.isChecked(), color_settings=cs)
        self.update_plotly_fig(fig)

        # preview localization precision
        lp = preview_localization_precision(arr_efo)
        lp = tuple(float(f"{v:.2f}") for v in lp) if lp is not None else None

        # ratio loc per trace
        if len(arr_efo) == 0:
            ratio_loc_per_trace = 0.0
        else:
            unique_tids, inv_idx, locs_per_tid = np.unique(
                arr_efo["tid"], return_inverse=True, return_counts=True
            )
            ratio_loc_per_trace = np.sum(locs_per_tid) / len(unique_tids)
        items = [
            ("File", self._current_ctx["base"]),
            ("Total imaging time (min)", f"{self._current_ctx['total_tim']/60:.2f}"),
            #("Total raw localizations", str(self._current_ctx["total_loc"])),
            ("Last iteration localizations", str(self._current_ctx["last_iteration_loc"])),
            ("After trace filtering", str(self._current_ctx["after_trace"])),
            ("After EFO filtering", str(int(np.count_nonzero(mask)))),
            ("Localization precision (x, y, z)", str(lp) if lp is not None else "—"),
            ("Ratio loc per trace", f"{ratio_loc_per_trace:.4f}"),
            ("% remaining", f"{(int(np.count_nonzero(mask)) / self._current_ctx['last_iteration_loc'] * 100):.2f}%"),
            ("Note", "Preview only. Click Continue to save + advance."),
        ]
        self.set_output(items)
        self.continue_btn.setEnabled(True)

    def continue_file(self):
        if self._current_worker is None:
            return
        if self._span_xmin is None or self._span_xmax is None:
            return
        self.apply_btn.setEnabled(False)
        self.continue_btn.setEnabled(False)
        self._current_worker.set_range_and_continue(self._span_xmin, self._span_xmax)

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
        if result is not None:
            self.set_output(result["items"])

        # advance to next file
        self._current_index += 1
        
        # Loop back to first file when reaching the end
        if self._current_index >= len(self._all_files):
            self._current_index = 0
            self.statusBar().showMessage("Reached end of list. Looping back to first file.")

        # Update combo box and label
        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(self._current_index)
        self.file_combo.blockSignals(False)
        
        self.start_worker_for_current()

def main():
    app = QtWidgets.QApplication(sys.argv)

    apply_gui_theme(app)   # <<< add this line

    from PySide6.QtWebEngineCore import QWebEngineProfile
    QWebEngineProfile.defaultProfile()

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
