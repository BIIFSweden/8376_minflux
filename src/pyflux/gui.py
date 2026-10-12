import glob
import json
import os
import sys
import tempfile
import time
from collections import Counter
from functools import partial

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from .core import (
    MinfluxLayoutError,
    apply_T,
    apply_transform_to_arr,
    compute_mbm_transform,
    dbscan_numpy,
    load_minflux_npy,
    np_to_df,
    preview_localization_precision,
    save_to_csv,
)
from .plotting import (
    add_scalebar_2d,
    apply_hist_theme,
    apply_plot_theme,
    CUSTOM_LUTS,
    DEFAULT_ALPHA,
    DEFAULT_LUT,
    DEFAULT_LUT_BIN,
    DEFAULT_MODE,
    DEFAULT_PROPERTY,
    DEFAULT_PROPERTY_CHOICES,
    DEFAULT_SIZE_2D,
    DEFAULT_SOLID,
    hist_edges,
    HIST_THEME,
    LUT_CHOICES,
    LUT_VALUE_MODES,
    make_plotly_fig,
    make_plotly_fig_merged,
    make_plotly_gaussian_from_arr,
    make_plotly_heatmap_from_arr,
    make_plotly_overlay_heatmap_from_two_arrs,
    MODE_CHOICES,
    MODE_DEPTH,
    MODE_E2E,
    MODE_PROPERTY,
    MODE_SOLID,
    MODE_TID,
    parse_limits,
    PLOTLY_HTML_BG,
    property_columns,
    render_gaussians_xy,
    SCALEBAR_COLOR,
    SCALEBAR_LENGTH_NM,
    SCALEBAR_LINE_WIDTH,
    SCALEBAR_MARGIN_FRACTION,
    scatter_points_and_color,
    SOLID_COLOR_CHOICES,
    SOLID_COLOR_MAP,
    tid_to_color,
    _clip_hist_for_overlay,
    _clip_to_scale_max,
    _lut_rgb_image,
    _resolve_overlay_colorscale,
    _xy_from_arr,
)
from .napari_storm_view import (
    DEFAULT_CONTRAST,
    NapariStormView,
    build_render_spec,
    build_render_specs,
    color_coded_supported,
    default_style,
    napari_storm_available,
    rendering_styles,
)

#: The two renderers pyflux can draw with.  napari-storm is optional: when
#: it (and napari) can be imported the localizations are drawn on an embedded
#: napari canvas; otherwise in Plotly scatter plots, as pyflux always did.
#: Decided once, when the main window is built; see
#: `napari_storm_view.napari_storm_available`.
BACKEND_NAPARI_STORM = "napari-storm"
BACKEND_PLOTLY = "plotly"


def rendering_backend():
    """Which renderer this process draws with."""
    return BACKEND_NAPARI_STORM if napari_storm_available() else BACKEND_PLOTLY


#: The contrast slider: its upper window limit runs from CONTRAST_MIN to
#: CONTRAST_MAX overlapping localizations on a log scale, so one slider spans
#: a single localization and a thousand with the same feel at either end.
CONTRAST_MIN = 0.05
CONTRAST_MAX = 1000.0
CONTRAST_SLIDER_STEPS = 1000


def contrast_from_slider(position):
    """The upper window limit at slider *position* (0..CONTRAST_SLIDER_STEPS)."""
    t = max(0.0, min(1.0, float(position) / CONTRAST_SLIDER_STEPS))
    return float(CONTRAST_MIN * (CONTRAST_MAX / CONTRAST_MIN) ** t)


def contrast_slider_position(upper):
    """The slider position for an upper window limit (the inverse)."""
    upper = max(CONTRAST_MIN, min(CONTRAST_MAX, float(upper)))
    t = np.log(upper / CONTRAST_MIN) / np.log(CONTRAST_MAX / CONTRAST_MIN)
    return int(round(t * CONTRAST_SLIDER_STEPS))


matplotlib.use("QtAgg")


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


# -------------------- Qt (GUI) theme --------------------

GUI_THEME = {
    "bg": "#202122",  # main window background
    "panel_bg": "#202122",  # groupboxes / panels
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


# -------------------- GUI components --------------------

class ColorSettingsPanel(QtWidgets.QGroupBox):
    """
    Table-based color settings:
      Columns: File | Mode | LUT/Solid | Property | Limits | α | Size (nm)
      One row per file base.

    Emits changed(base, settings_dict).
    settings_dict keys:
      - mode: str  ("solid"|"depth"|"tid"|"end-to-end"|"property")
      - solid: str
      - lut: str
      - property: str  -- the column coloured by in "property" mode
      - limits: (low, high) or None  -- the colour range of that column;
                None is automatic (1st to 99th percentile)
      - alpha: float
      - size: int   -- point radius, or Gaussian sigma, in nm (it was the
                       marker size in px)
    """
    changed = Signal(str, object)  # (base, settings dict)

    COL_FILE = 0
    COL_MODE = 1
    COL_PALETTE = 2
    COL_PROPERTY = 3
    COL_LIMITS = 4
    COL_ALPHA = 5
    COL_SIZE = 6
    COL_MBM_SOURCE = 7

    def __init__(self, title="Color settings", parent=None):
        super().__init__(title, parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.table = QtWidgets.QTableWidget(0, 8, self)
        self.table.setHorizontalHeaderLabels(
            ["File", "Mode", "LUT / Solid", "Property", "Limits", "α", "Size nm", "mbm source"]
        )
        self._alpha_enabled = True  # α only means something for Gaussians
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setAlternatingRowColors(False)

        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(False)

        # Eight columns have to share the panel: the file name gets what is
        # left after the controls have what they need to be read.
        hdr.setSectionResizeMode(self.COL_FILE, QtWidgets.QHeaderView.Stretch)
        hdr.setMinimumSectionSize(48)
        for col, width in (
            (self.COL_MODE, 92),
            (self.COL_PALETTE, 86),
            (self.COL_PROPERTY, 78),
            (self.COL_LIMITS, 84),
            (self.COL_ALPHA, 58),
            (self.COL_SIZE, 58),
            (self.COL_MBM_SOURCE, 78),
        ):
            hdr.setSectionResizeMode(col, QtWidgets.QHeaderView.Interactive)
            self.table.setColumnWidth(col, width)

        # make it scroll when many rows
        self.table.setMinimumHeight(170)
        self.table.setMaximumHeight(260)

        outer.addWidget(self.table, 1)

        # base -> dict of widgets
        self._row_by_base = {}  # base -> row index
        self._widgets_by_base = {}  # base -> dict(mode=..., palette=..., property=..., limits=..., alpha=..., size=...)
        # base -> the columns its file has, once it is loaded
        self._properties_by_base = {}

    def _emit_row(self, base: str):
        w = self._widgets_by_base.get(base)
        if not w:
            return

        mode = w["mode"].currentText()
        alpha = float(w["alpha"].value())
        size = int(w["size"].value())
        prop = w["property"].currentText() or DEFAULT_PROPERTY
        limits = parse_limits(w["limits"].text())

        # palette meaning depends on mode
        if mode == MODE_SOLID:
            payload = {
                "mode": MODE_SOLID,
                "solid": w["palette"].currentText(),
                "lut": DEFAULT_LUT,
            }
        elif mode == MODE_TID:
            payload = {
                "mode": MODE_TID,
                "solid": DEFAULT_SOLID,
                "lut": DEFAULT_LUT,
            }
        else:  # depth, end-to-end, property: a LUT over a value
            payload = {
                "mode": mode,
                "solid": DEFAULT_SOLID,
                "lut": w["palette"].currentText(),
            }
        payload.update(property=prop, limits=limits, alpha=alpha, size=size)

        self.changed.emit(base, payload)

    def set_properties(self, base: str, names):
        """Offer the columns of *base*'s file, now that it is loaded.

        The row is built before the file is read, from a stock list of MINFLUX
        columns; the file's own columns replace it here, keeping the chosen one
        when the file has it.
        """
        names = [str(n) for n in (names or [])]
        self._properties_by_base[base] = names
        w = self._widgets_by_base.get(base)
        if not w or not names:
            return
        combo = w["property"]
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        combo.setCurrentText(current if current in names else (DEFAULT_PROPERTY if DEFAULT_PROPERTY in names else names[0]))
        combo.blockSignals(False)
        if combo.currentText() != current and w["mode"].currentText() == MODE_PROPERTY:
            self._emit_row(base)

    def set_alpha_enabled(self, enabled: bool):
        """Grey α out while it has no effect: points are drawn opaque."""
        self._alpha_enabled = bool(enabled)
        for widgets in self._widgets_by_base.values():
            widgets["alpha"].setEnabled(self._alpha_enabled)

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

        else:  # "depth", "end-to-end" or "property"
            palette.setEnabled(True)
            palette.addItems(LUT_CHOICES)
            palette.setCurrentText(lut_default if lut_default in LUT_CHOICES else DEFAULT_LUT)

        palette.blockSignals(False)
        # the column and its range only mean something when colouring by a property
        w["property"].setEnabled(mode == MODE_PROPERTY)
        w["limits"].setEnabled(mode == MODE_PROPERTY)

    def rebuild(self, base_names, current_settings_by_base: dict, mbm_source_base=None):
        """Repopulate the table. `mbm_source_base` is the single authority for the
        MBM source column — exactly one row can be checked, so the flag is derived
        from MainWindow._mbm_source_base rather than stored per base in
        current_settings_by_base (nothing ever wrote it there, so every rebuild
        used to silently uncheck the source while _mbm_source_base stayed set).
        """
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

            prop = s.get("property")
            limits = s.get("limits")

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

            # --- Property combo: the column coloured by in property mode ---
            property_combo = QtWidgets.QComboBox()
            choices = list(self._properties_by_base.get(base) or DEFAULT_PROPERTY_CHOICES)
            if prop is None:
                prop = DEFAULT_PROPERTY if DEFAULT_PROPERTY in choices else choices[0]
            elif prop not in choices:
                choices.append(prop)  # chosen earlier; keep it
            property_combo.addItems(choices)
            property_combo.setCurrentText(prop)
            property_combo.setToolTip(
                "The column of the file to colour by (mode 'property'):\n"
                "EFO, CFR, photon counts, time, ... The file's own columns\n"
                "are listed once it is loaded."
            )

            # --- Limits: the range of that column the LUT spans ---
            limits_edit = QtWidgets.QLineEdit()
            limits_edit.setPlaceholderText("auto")
            if limits is not None:
                limits_edit.setText(", ".join("" if v is None else f"{float(v):g}" for v in limits))
            limits_edit.setToolTip(
                "Colour range of the property, as 'low, high'.\n"
                "Empty or 'auto': the 1st to 99th percentile of the values.\n"
                "Leave one end empty to keep it automatic, e.g. ', 0.8'."
            )

            # --- Alpha spin ---
            alpha_spin = QtWidgets.QDoubleSpinBox()
            alpha_spin.setRange(0.0, 1.0)
            alpha_spin.setDecimals(2)
            alpha_spin.setSingleStep(0.05)
            alpha_spin.setValue(alpha)
            alpha_spin.setEnabled(self._alpha_enabled)
            alpha_spin.setToolTip(
                "Opacity of Gaussians. Points are opaque: overlapping points hide\n"
                "each other by depth instead of blending."
            )

            # --- Size spin ---
            size_spin = QtWidgets.QSpinBox()
            size_spin.setRange(1, 50)
            size_spin.setValue(size)
            size_spin.setToolTip(
                "Point radius, or Gaussian width (sigma), in nm.\n"
                "Points are never drawn smaller than 5 screen pixels."
            )

            # --- mbm source ---
            src_chk = QtWidgets.QCheckBox()
            src_chk.setChecked(mbm_source_base is not None and base == mbm_source_base)
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
                property=property_combo,
                limits=limits_edit,
                alpha=alpha_spin,
                size=size_spin,
                mbm_source=src_chk,  # self._widgets_by_base[base]["mbm_source"] = src_chk # Different??
            )

            # initial palette fill based on mode
            self._fill_palette_for_mode(base, mode, solid_default=solid, lut_default=lut)

            # connect signals (avoid lambda capturing row; capture base)
            mode_combo.currentTextChanged.connect(lambda m, b=base: self._on_mode_changed(b, m))
            palette_combo.currentTextChanged.connect(lambda _, b=base: self._emit_row(b))
            property_combo.currentTextChanged.connect(lambda _, b=base: self._emit_row(b))
            limits_edit.editingFinished.connect(lambda b=base: self._emit_row(b))
            alpha_spin.valueChanged.connect(lambda _, b=base: self._emit_row(b))
            size_spin.valueChanged.connect(lambda _, b=base: self._emit_row(b))

            self.table.setCellWidget(row, self.COL_MODE, mode_combo)
            self.table.setCellWidget(row, self.COL_PALETTE, palette_combo)
            self.table.setCellWidget(row, self.COL_PROPERTY, property_combo)
            self.table.setCellWidget(row, self.COL_LIMITS, limits_edit)
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
        # stateChanged delivers a plain int and Qt.CheckState is not an IntEnum in
        # PySide6 6.9, so `state != Qt.Checked` is always True. Read the widget.
        chk_self = (self._widgets_by_base.get(base) or {}).get("mbm_source")
        checked = chk_self.isChecked() if chk_self is not None else int(state) == Qt.Checked.value

        if not checked:
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

    def set_mbm_rows(self, file_paths, current=None, bases=None):
        """One row per file. `current` maps base -> already-chosen MBM folder; a base
        missing from it (or mapped to "") falls back to the <dir of .npy>/grd/mbm
        default. Without it a custom path would be overwritten by the default every
        time the dialog is opened, since mbm_map() is written back on OK.

        `bases` are the dataset keys parallel to `file_paths` (MainWindow.all_bases());
        deriving them here from the basename would reintroduce the collision between
        two identically named .npy files in different subfolders."""
        current = current or {}
        self.mbm_table.setRowCount(0)
        for r, fp in enumerate(file_paths):
            base = bases[r] if bases else os.path.splitext(os.path.basename(fp))[0]
            default_mbm = os.path.join(os.path.dirname(fp), "grd", "mbm")

            self.mbm_table.insertRow(r)
            self.mbm_table.setItem(r, 0, QtWidgets.QTableWidgetItem(base))

            le = QtWidgets.QLineEdit(current.get(base) or default_mbm)
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
        self.eps_spin.setValue(80.0)
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
            self._show_fig(fig)

    def _selected_arr(self, base: str):
        # Keep behavior aligned with multicolor viewer: refresh crop cache first,
        # then resolve through the same accessor. Must cover per-base crops too —
        # _get_multicolor_arr_for_base serves them whether or not a global crop exists.
        if self._mw._has_any_crop():
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
        if "tid" in (arr.dtype.names or ()):
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

    def _show_fig(self, fig):
        """Render a figure into this window's view via a full page load.

        This window deliberately bypasses PlotlyView's Plotly.react path to avoid
        the sync/scalebar side effects that were hiding markers here. setHtml
        replaces the bootstrap DOM (#plot, #container, window._plotSync) and
        ensure_page() will not rebuild it, since _plotly_loaded was set when
        PlotlyView was constructed — so *every* figure in this window must go
        through here. A stray view.update_fig() would inject JS into a page with no
        #plot, hit its `return "error: container missing"` guard and do nothing,
        with no error anywhere: the previous plot would simply stay on screen.

        Cost of the full reload is pan/zoom state, which this window does not try
        to preserve. include_plotlyjs="cdn" is now the window's only CDN reference,
        so vendoring plotly.min.js means changing this one call.
        """
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

    def _plot_tracks(self, tracks_df: pd.DataFrame, labels=None, title: str = None):
        fig = go.Figure()
        is3d = self._is_3d_mode()
        if tracks_df is None or len(tracks_df) == 0:
            fig.update_layout(annotations=[dict(text="No avg loc (tid) data", x=0.5, y=0.5, showarrow=False)])
            apply_plot_theme(fig, is3d=is3d)
            self._show_fig(fig)
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
                    dragmode="turntable",
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

        self._show_fig(fig)

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
            self._show_fig(fig)
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
        # `dtype.names` is None for a plain (non-structured) array, so membership has
        # to be tested against a tuple. The fallbacks must also all be length n or the
        # DataFrame below raises on mismatched column lengths.
        names = arr.dtype.names or ()
        n = len(arr)
        tid = np.asarray(arr["tid"]).astype(int) if "tid" in names else np.full(n, -1, dtype=int)
        loc = np.asarray(arr["loc"], dtype=float) if "loc" in names else np.full((n, 3), np.nan)
        if loc.ndim != 2 or len(loc) != n:
            loc = np.full((n, 3), np.nan)

        x = loc[:, 0] if loc.shape[1] >= 1 else np.full(n, np.nan)
        y = loc[:, 1] if loc.shape[1] >= 2 else np.full(n, np.nan)
        z = loc[:, 2] if loc.shape[1] >= 3 else np.full(n, np.nan)

        if "tim" in names:
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
    """Every dataset side by side (grid) or in one canvas (merged).

    Drawn by whichever renderer the main window uses (*backend*).  With
    napari-storm each grid cell is its own embedded napari viewer and the merged
    page is one viewer holding a layer per dataset; Crop keeps what a 2-D panel
    currently shows.  With Plotly each cell is a scatter plot in a web view and
    Crop takes the last box selection.  Panels with Synchronize ticked share
    pan, zoom and 3-D rotation either way.

    `update_one`, `update_all` and `update_merged` take what the backend draws:
    a `RenderSpec` per base for napari-storm, a Plotly figure otherwise.
    """

    def __init__(self, on_view_changed, on_crop_requested=None, on_reset_crop_requested=None, parent=None,
                 on_ndisplay_changed=None, backend=BACKEND_NAPARI_STORM):
        super().__init__(parent)
        self.setWindowTitle("Multicolor")
        self.resize(1200, 700)
        self._backend = backend
        self._on_view_changed = on_view_changed
        self._on_crop_requested = on_crop_requested
        self._on_reset_crop_requested = on_reset_crop_requested
        self._on_ndisplay_changed = on_ndisplay_changed
        self._last_active_source_id = None  # the panel the pointer touched last
        self._last_selection_payload = None  # Plotly: the last box selection
        self._last_selection_source_id = None
        self._contrast = DEFAULT_CONTRAST
        self._specs_by_base = {}  # napari-storm: what each grid panel shows
        self._merged_specs = {}  # napari-storm: what the merged viewer shows

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        # --- stacked widget with plots (create ONCE) ---
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)

        # --- crop buttons overlaid bottom-right ---
        self._crop_btn = QtWidgets.QPushButton("Crop", central)
        if self.uses_napari_storm:
            self._crop_btn.setToolTip(
                "Keep only the localizations inside the visible area of a 2-D panel:\n"
                "pan and zoom to the region, then press Crop.\n"
                "In grid mode the panel the pointer touched last is used."
            )
        else:
            self._crop_btn.setToolTip("Keep only the localizations inside the last box selection.")
        self._crop_btn.clicked.connect(self._crop_current_selection)
        self._crop_btn.setFixedHeight(24)
        self._crop_btn.setMaximumWidth(90)
        self._crop_btn.raise_()

        self._reset_crop_btn = QtWidgets.QPushButton("Reset Crop", central)
        self._reset_crop_btn.clicked.connect(self._reset_crop)
        self._reset_crop_btn.setFixedHeight(24)
        self._reset_crop_btn.setMaximumWidth(110)
        self._reset_crop_btn.raise_()

        self._axis_titles = {}  # Plotly: 3-D axis titles, learned from each pushed figure
        if self.uses_napari_storm:
            self._grid_on = False
            self._grid_chk = QtWidgets.QCheckBox("Axes", central)
            self._grid_chk.setToolTip("Show the coordinate axes in every panel.")
        else:
            self._grid_on = True
            self._grid_chk = QtWidgets.QCheckBox("Grid", central)
            self._grid_chk.setToolTip(
                "Show the plot grid.\n"
                "In 3-D this hides the whole coordinate system (grid, ticks, labels, axes)."
            )
        self._grid_chk.setChecked(self._grid_on)
        self._grid_chk.toggled.connect(self._on_grid_toggled)
        self._grid_chk.setFixedHeight(24)
        self._grid_chk.raise_()

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

        self.merged_view = self._make_view("Multicolor (merged)", self._on_view_changed)
        mv.addWidget(self.merged_view, 1)

        self.stack.addWidget(self.merged_page)

        # state
        self._views = {}  # base -> NapariStormView
        self._mode = "grid"  # or "merged"

        # --- per-sample view synchronization (grid mode only) ---
        self._sync_by_base = {}      # base -> bool, survives rebuild()
        self._sync_boxes = {}        # base -> QCheckBox
        self._last_view_by_base = {}  # base -> last camera payload, to seed a newly ticked view
        # base -> (deadline, payload just pushed). A view we pushed a camera into
        # may echo that same change straight back; the view's own guard catches
        # the common case and this is the backstop that stops an echo becoming a
        # loop. Matching on the payload as well as the time window matters:
        # suppressing a base outright would swallow a real gesture made right
        # after it was synced.
        self._sync_suppress = {}

    @property
    def uses_napari_storm(self):
        return self._backend == BACKEND_NAPARI_STORM

    def _make_view(self, title, on_view_changed):
        """A viewer widget of the backend's kind, wired to this window."""
        if self.uses_napari_storm:
            view = NapariStormView(title=title)
            view.viewChanged.connect(on_view_changed)
            view.activated.connect(self._on_view_activated)
            view.ndisplayChanged.connect(self._relay_ndisplay)
        else:
            view = PlotlyView()
            view.viewChanged.connect(on_view_changed)
            view.selectionChanged.connect(self._on_plot_selection)
        return view

    def set_mode(self, mode: str):
        mode = "merged" if mode == "merged" else "grid"
        self._mode = mode
        self.stack.setCurrentIndex(1 if mode == "merged" else 0)

    def mode(self):
        return self._mode

    def rebuild(self, base_names):
        base_names = list(base_names)

        # A napari viewer is expensive to build and its camera is worth keeping
        # across a refresh, so panels of samples this folder still has are
        # reused; only the ones that went away are torn down.  A Plotly page
        # is rebuilt with its figure anyway, so those panels are not kept.
        keep = {b: v for b, v in self._views.items() if b in base_names} if self.uses_napari_storm else {}
        for base, view in list(self._views.items()):
            if base not in keep:
                view.shutdown()
                view.setParent(None)
                view.deleteLater()
        for view in keep.values():
            view.setParent(None)  # out of the group boxes about to be deleted
        self._specs_by_base = {b: v for b, v in self._specs_by_base.items() if b in keep}

        # clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._views = {}
        self._sync_boxes.clear()
        # keep the ticks the user set, but only for samples this folder still has
        self._sync_by_base = {b: v for b, v in self._sync_by_base.items() if b in base_names}
        self._last_view_by_base = {b: v for b, v in self._last_view_by_base.items() if b in base_names}
        self._sync_suppress.clear()

        n = max(1, len(base_names))
        cols = 2 if n <= 4 else 3
        r = c = 0

        for base in base_names:
            box = QtWidgets.QGroupBox(base)
            vbox = QtWidgets.QVBoxLayout(box)
            vbox.setContentsMargins(4, 8, 4, 4)

            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            sync_chk = QtWidgets.QCheckBox("Synchronize")
            sync_chk.setToolTip(
                "Pan, zoom and 3-D rotation done here are applied to every other\n"
                "sample with Synchronize ticked. Cropping a synchronized sample\n"
                "crops all of them."
            )
            sync_chk.setChecked(bool(self._sync_by_base.get(base, False)))
            sync_chk.toggled.connect(partial(self._on_sync_toggled, base))
            header.addWidget(sync_chk)
            header.addStretch(1)
            vbox.addLayout(header)

            view = keep.get(base)
            if view is None:
                view = self._make_view(f"Multicolor: {base}", self._on_grid_view_changed)
            # set before the page exists: a Plotly view re-asserts it into the new page
            view.set_live_sync(self._sync_by_base.get(base, False))
            if self.uses_napari_storm:
                view.set_grid_visible(self._grid_on)
            vbox.addWidget(view, 1)

            self._views[base] = view
            self._sync_boxes[base] = sync_chk
            self.grid.addWidget(box, r, c)

            c += 1
            if c >= cols:
                c = 0
                r += 1

    # ---------------- grid / axes visibility ----------------

    def _apply_grid_to_fig(self, fig, is3d: bool, grid_on: bool):
        """Plotly: bake the Grid setting into a figure before it is pushed.

        Applied at the two places figures reach a view, so a redraw (filter, crop,
        colour change, mode switch) cannot quietly bring the grid back. When the
        grid is on the figure is left untouched rather than forced to `True`, so
        "reinstate" restores exactly what `apply_plot_theme` produced.
        """
        if fig is None or not hasattr(fig, "update_layout"):
            return fig
        if is3d:
            # Read off the *untouched* figure, which always carries the themed
            # titles whichever way the checkbox is set — the live page cannot be
            # asked for them once they have been blanked below, and hard-coding
            # "X"/"Y"/"Z" here would silently drift from plotting.py.
            try:
                sc = fig.layout.scene
                seen = {a: getattr(sc, a).title.text for a in ("xaxis", "yaxis", "zaxis")}
                if all(seen.values()):
                    self._axis_titles = seen
            except Exception:
                pass
        if grid_on:
            return fig
        if is3d:
            # component by component rather than `visible`, and matching
            # PlotlyView.set_grid_visible exactly — see the note there on why
            # hiding a scene axis outright is a one-way door for the titles
            off = dict(showgrid=False, zeroline=False, showticklabels=False,
                       showline=False, ticks="", title=dict(text=""))
            fig.update_layout(scene=dict(xaxis=off, yaxis=off, zaxis=off))
        else:
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
        return fig

    def _on_grid_toggled(self, checked: bool):
        self._grid_on = bool(checked)
        # relayout the live pages for instant feedback instead of re-rendering every
        # figure; for Plotly the figure-side bake above keeps it stuck on the next redraw
        for view in list(self._views.values()) + [self.merged_view]:
            if view is not None:
                try:
                    if self.uses_napari_storm:
                        view.set_grid_visible(self._grid_on)
                    else:
                        view.set_grid_visible(self._grid_on, self._axis_titles)
                except Exception:
                    pass

    def update_one(self, base, item, reset_view=False, is3d=False, scalebar_nm=100.0):
        """Draw *item* in the panel of *base*.

        *item* is a RenderSpec (or None for nothing) with napari-storm, a
        Plotly figure otherwise.
        """
        view = self._views.get(base)
        if view is None:
            return
        if reset_view:
            # the remembered camera no longer describes what is on screen, and
            # after a 2D/3D switch it is the wrong kind of payload entirely
            self._last_view_by_base.pop(base, None)
            self._sync_suppress.pop(base, None)
        if self.uses_napari_storm:
            self._specs_by_base[base] = item
            view.set_scalebar_nm(scalebar_nm)
            view.show_datasets({base: item}, is3d=is3d, reset_view=reset_view)
        else:
            self._apply_grid_to_fig(item, is3d, self._grid_on)
            if (not is3d) and item is not None:
                add_scalebar_2d(item, length_nm=float(scalebar_nm))
            view.update_fig(item, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def update_all(self, items_by_base, reset_view=False, is3d=False, scalebar_nm=100.0):
        for base, item in items_by_base.items():
            self.update_one(base, item, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def set_contrast(self, contrast):
        """napari-storm: re-window every open dataset to *contrast* (see
        `NapariStormView.set_contrast`) without redrawing it."""
        self._contrast = tuple(contrast)
        if not self.uses_napari_storm:
            return
        for base, view in self._views.items():
            spec = self._specs_by_base.get(base)
            if spec is not None:
                view.set_contrast(self._contrast, {base: spec})
        if self._merged_specs:
            self.merged_view.set_contrast(self._contrast, self._merged_specs)

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

            chk = getattr(self, "_grid_chk", None)
            if chk is not None:
                chk.adjustSize()
                # sits to the left of Crop, baseline-aligned with the two buttons
                chk.move(x_crop - chk.width() - gap,
                         y + (btn_crop.height() - chk.height()) // 2)

    def _source_id_to_base(self, source_id: str):
        for base, view in self._views.items():
            if view._id == source_id:
                return base
        return None

    # ---------------- per-sample view synchronization ----------------

    SYNC_ECHO_SUPPRESS_S = 0.15

    def synced_bases(self):
        """Bases whose Synchronize box is ticked, in grid order."""
        return [b for b in self._views if self._sync_by_base.get(b, False)]

    def _on_sync_toggled(self, base: str, checked: bool):
        self._sync_by_base[base] = bool(checked)
        # only a ticked panel streams during-drag events, so an unused feature costs
        # nothing and the per-frame traffic stops the moment the box is cleared
        view = self._views.get(base)
        if view is not None:
            view.set_live_sync(bool(checked))
        if not checked:
            return
        # adopt whatever the group is currently looking at, so ticking the box does
        # not leave this sample on a different region until the next mouse move
        for other in self.synced_bases():
            if other == base:
                continue
            payload = self._last_view_by_base.get(other)
            if isinstance(payload, dict):
                self._push_view(base, payload)
                break

    @staticmethod
    def _same_view(a, b):
        """Structural comparison of two relayout payloads, tolerant of float drift."""
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(MultiColorWindow._same_view(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(MultiColorWindow._same_view(x, y) for x, y in zip(a, b))
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
        return a == b

    @staticmethod
    def _strip_live(payload):
        """A payload minus the transport-only `live` marker.

        `live` records how a frame was produced (mid-gesture vs gesture end), not
        what the view is, and only mid-gesture frames carry it. Echo suppression
        compares key sets, so leaving it in means a pushed live frame can never
        match the echo it provokes — the echo is then re-broadcast as if it were a
        fresh gesture and bounces around the synchronized group.
        """
        if isinstance(payload, dict) and "live" in payload:
            return {k: v for k, v in payload.items() if k != "live"}
        return payload

    def _push_view(self, base: str, payload: dict):
        view = self._views.get(base)
        if view is None:
            return
        self._sync_suppress[base] = (time.monotonic() + self.SYNC_ECHO_SUPPRESS_S,
                                     self._strip_live(payload))
        try:
            view.apply_view(payload)
        except Exception:
            pass

    def _on_grid_view_changed(self, source_id, payload):
        """Relay one grid view's pan/zoom/camera to the other synchronized ones."""
        base = self._source_id_to_base(source_id)
        if base is not None and isinstance(payload, dict):
            deadline, pushed = self._sync_suppress.get(base, (0.0, None))
            if deadline > time.monotonic() and self._same_view(self._strip_live(payload), pushed):
                # the echo of what we just pushed here, not a gesture the user made
                return
            self._last_view_by_base[base] = payload
            if self._mode != "merged" and self._sync_by_base.get(base, False):
                for other in self.synced_bases():
                    if other == base:
                        continue
                    self._last_view_by_base[other] = payload
                    self._push_view(other, payload)
        if self._on_view_changed is not None:
            self._on_view_changed(source_id, payload)

    def _on_view_activated(self, source_id):
        self._last_active_source_id = source_id

    def _on_plot_selection(self, _source_id, payload):
        """Plotly: remember the last box selection, which Crop then uses."""
        if not isinstance(payload, dict):
            return
        if payload.get("mode") != "2d":
            return
        xr = payload.get("xRange")
        yr = payload.get("yRange")
        if not (isinstance(xr, (list, tuple)) and isinstance(yr, (list, tuple)) and len(xr) == 2 and len(yr) == 2):
            return
        self._last_selection_payload = payload
        self._last_selection_source_id = _source_id

    def _relay_ndisplay(self, ndisplay):
        if self._on_ndisplay_changed is not None:
            self._on_ndisplay_changed(int(ndisplay))

    def _crop_current_selection(self):
        """Crop to the visible area of the merged view, or of the panel touched last.

        With Plotly, to the last box selection instead.
        """
        if self._on_crop_requested is None:
            return
        if not self.uses_napari_storm:
            self._crop_plotly_selection()
            return
        if self._mode == "merged":
            view, base = self.merged_view, None
        else:
            base = self._source_id_to_base(self._last_active_source_id)
            if base is None and self._views:
                base = next(iter(self._views))
            view = self._views.get(base)
        rect = view.visible_rect_2d() if view is not None else None
        if rect is None:
            QtWidgets.QMessageBox.information(
                self, "Crop",
                "Crop keeps what a 2-D panel is showing: switch 3D off, pan and zoom "
                "to the region, then press Crop.",
            )
            return
        xmin, xmax, ymin, ymax = rect
        payload = {
            "mode": "2d",
            "xRange": [float(xmin), float(xmax)],
            "yRange": [float(ymin), float(ymax)],
        }
        if base is not None:
            payload["base"] = base
            if self._sync_by_base.get(base, False):
                # a synchronized sample crops its whole group, not just itself
                payload["bases"] = self.synced_bases()
        self._on_crop_requested(payload)

    def _crop_plotly_selection(self):
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
        if self._mode != "merged":
            src_id = self._last_selection_source_id
            base = self._source_id_to_base(src_id) if src_id else None
            if base is not None:
                payload["base"] = base
                if self._sync_by_base.get(base, False):
                    # a synchronized sample crops its whole group, not just itself
                    payload["bases"] = self.synced_bases()
        self._on_crop_requested(payload)

    def _reset_crop(self):
        if self._on_reset_crop_requested is None:
            return
        self._on_reset_crop_requested()

    def grid_sync_enabled(self) -> bool:
        return False

    def update_merged(self, item, reset_view=False, is3d=False, scalebar_nm=100.0):
        """Draw *item* in the one merged viewer: every dataset's RenderSpec by
        base with napari-storm, one Plotly figure otherwise."""
        if self.uses_napari_storm:
            self._merged_specs = dict(item)
            self.merged_view.set_scalebar_nm(scalebar_nm)
            self.merged_view.show_datasets(item, is3d=is3d, reset_view=reset_view)
        else:
            self._apply_grid_to_fig(item, is3d, self._grid_on)
            self.merged_view.update_fig(item, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

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
        self._live_sync = False  # during-drag broadcast; see set_live_sync

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

    # Never connected to any signal; the real handler is the module-level
    # handle_download_requested wired up in main.py.
    # def _on_download_requested(self, download):
    #     suggested = download.suggestedFileName() or "plot.png"
    #     path, _ = QtWidgets.QFileDialog.getSaveFileName(
    #         self, "Save Plot", suggested, "PNG Image (*.png);;All Files (*)")
    #     if not path:
    #         download.cancel()
    #         return
    #     if not os.path.splitext(path)[1]:
    #         path += ".png"
    #     download.setDownloadDirectory(os.path.dirname(path))
    #     download.setDownloadFileName(os.path.basename(path))
    #     download.accept()

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
    window._plotSync = {{ ready:false, bridge:null, sourceId:null, ignore:false, _scalebarUpdating:false, scalebarNm: 100.0,
                          liveSync:false, _applying:false, _queued:null, gestureActive:false,
                          _settleRaf:null, _settleCommit:false, _lastCamSent:null,
                          pointerDown:false }};

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

    // This runs after EVERY Plotly.react, not just a DOM rebuild. Only
    // 'plotly_relayout' used to be cleared, so one live-sync handler was left
    // behind per redraw — each with its own _pending/_timer/_lastSent, all
    // broadcasting, older ones bypassing the throttle with a stale _lastSent and
    // sending snapshots taken at different instants. Clear both.
    if (gd.removeAllListeners) {{
        gd.removeAllListeners('plotly_relayout');
        gd.removeAllListeners('plotly_relayouting');
    }}
    // A 2-D/3-D switch rebuilds the node but keeps _plotSync, so a settle watcher
    // left running from the old scene would commit a camera to a detached div.
    if (window._plotSync._settleRaf) {{
        cancelAnimationFrame(window._plotSync._settleRaf);
        window._plotSync._settleRaf = null;
    }}

    let _lastSent = 0;
    let _timer = null;
    let _pending = null;

    // A 3-D gesture is only ever started by an input event. gl-plot3d keeps
    // interpolating the camera for a few frames after the button is released, so
    // plotly_relayouting carries on firing AFTER the gesture-end plotly_relayout —
    // and whatever arrives last wins on the receiving side, leaving the other
    // panels on the settled camera while this one's _fullLayout is frozen at the
    // mouse-up value. They then disagree on the next full redraw. Gating live
    // frames on a real gesture drops those trailing frames exactly, with no timer.
    // Flag lives on _plotSync, and the DOM listeners are attached once per node:
    // this function re-runs on every redraw, and addEventListener has no
    // removeAllListeners equivalent, so a per-call closure would accumulate.
    // A 2-D/3-D switch rebuilds the node, which drops the marker with it.
    if (!gd.__gestureHooked) {{
        gd.__gestureHooked = true;
        ['mousedown', 'wheel', 'touchstart'].forEach(function(evt) {{
        gd.addEventListener(evt, function() {{
            window._plotSync.gestureActive = true;
        }}, true);
        }});
        gd.addEventListener('mousedown', function() {{
        window._plotSync.pointerDown = true;
        }}, true);
    }}
    // Whether a button is held is tracked separately from gestureActive, because
    // the relayout handler clears gestureActive on EVERY relayout including a
    // mid-drag one — which let the settle watcher run against a gesture still in
    // progress, commit a scene.camera on top of it and cancel the user's rotation
    // outright (observed as source_moved=False while the targets had followed).
    // The button can be released outside the plot div, so this goes on the window,
    // once per page rather than once per node.
    if (!window.__plotSyncPointerHooked) {{
        window.__plotSyncPointerHooked = true;
        ['mouseup', 'touchend', 'touchcancel'].forEach(function(evt) {{
        window.addEventListener(evt, function() {{
            window._plotSync.pointerDown = false;
        }}, true);
        }});
    }}

    function buildPayload(gd) {{
        const is3d = !!(gd._fullLayout && gd._fullLayout.scene);
        const payload = {{ mode: is3d ? "3d" : "2d" }};

        if (is3d) {{
        const sc = gd._fullLayout.scene;
        let cam = null;
        // Always the RENDERED camera, never _fullLayout.scene.camera. plotly does
        // not commit the stored one during a gesture (measured: 2 distinct values
        // across a drag versus 26 from getCamera()), and — the part that was missed
        // first time round — it has still not committed it when the gesture-end
        // plotly_relayout fires either. Building that frame from the stored value
        // broadcast the PRE-GESTURE camera, so every target visibly snapped back to
        // where the gesture started and was only corrected ~100 ms later by the
        // settle broadcast below. Same {{up, center, eye, projection}} shape
        // apply_view feeds to scene.camera. Stored value kept only as the fallback
        // for a page with no WebGL, or before gl-plot3d has come up.
        try {{
            const s = sc && sc._scene;
            if (s && s.glplot && s.getCamera) cam = s.getCamera();
        }} catch (e) {{}}
        if (!cam) cam = sc && sc.camera;
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

        // Only `ignore` is re-checked here. _scalebarUpdating is deliberately not:
        // the handler already rejected the scalebar's own relayout on the way in,
        // and on the immediate path this flag is still true from the update that
        // this very gesture just kicked off.
        if (window._plotSync.ignore) {{
        _pending = null;
        return;
        }}

        const payload = _pending;
        _pending = null;

        // Remember the camera we last put on the wire, so the settle watcher can
        // tell "the ease carried on past this" from "nothing has changed".
        if (payload.camera) {{
        try {{
            const c = payload.camera;
            window._plotSync._lastCamSent = [c.eye.x, c.eye.y, c.eye.z,
                                             c.center.x, c.center.y, c.center.z,
                                             c.up.x, c.up.y, c.up.z];
        }} catch (e) {{}}
        }}

        if (window._plotSync.ready && window._plotSync.bridge) {{
        window._plotSync.bridge.relayView(window._plotSync.sourceId || "unknown", payload);
        }}
    }}

    // gl-plot3d EASES the camera toward its target over several frames. A scroll
    // zoom emits one plotly_relayout per notch, each carrying the camera as it
    // stands at that instant — mid-ease — and then emits nothing whatsoever once
    // the wheel stops, so the settled camera is never broadcast at all and every
    // synchronized panel is left one animation tail behind. Measured over six
    // notches: the source ends at eye-distance 0.889 and the targets at 1.033, a
    // visibly shallower zoom that reads as the other panels snapping back to a
    // previous view. A drag escapes this only because its gesture-end relayout
    // arrives after the ease has already finished (residual 2e-4). So watch the
    // RENDERED camera until it stops moving, then broadcast that.
    function scheduleSettleBroadcast(gd) {{
        const sc = gd._fullLayout && gd._fullLayout.scene;
        const s = sc && sc._scene;
        if (!s || !s.glplot || !s.getCamera) return;

        if (window._plotSync._settleRaf) {{
        cancelAnimationFrame(window._plotSync._settleRaf);
        window._plotSync._settleRaf = null;
        }}

        const flat = (c) => [c.eye.x, c.eye.y, c.eye.z,
                             c.center.x, c.center.y, c.center.z,
                             c.up.x, c.up.y, c.up.z];
        let prev = null, stillFor = 0;
        const giveUpAt = Date.now() + 1500;

        const step = function() {{
        window._plotSync._settleRaf = null;
        // a gesture owns the camera now, and will schedule its own watcher when it
        // ends. pointerDown covers the case gestureActive cannot: a mid-drag
        // relayout clears gestureActive while the button is still held.
        if (window._plotSync.gestureActive || window._plotSync.pointerDown) return;
        let cam = null;
        try {{ cam = s.getCamera(); }} catch (e) {{ return; }}
        if (!cam || !cam.eye) return;
        const cur = flat(cam);
        const moved = !prev || cur.some((v, i) => Math.abs(v - prev[i]) > 1e-7);
        prev = cur;
        stillFor = moved ? 0 : stillFor + 1;
        // two identical frames, or a cap so this can never spin indefinitely
        if (stillFor < 2 && Date.now() < giveUpAt) {{
            window._plotSync._settleRaf = requestAnimationFrame(step);
            return;
        }}

        // Broadcasting and committing are decided SEPARATELY. Tying them to one
        // "has anything changed" test meant that whenever the ease had already
        // finished by mouse-up — so the gesture-end frame was already correct and
        // needed no follow-up — the commit was skipped along with it, and the
        // source kept plotly's stale stored camera while the targets moved on
        // (seen as source_moved=False in the loop harness). Between them these two
        // tests are also what stops the commit feeding itself: after it lands,
        // stored == cur == _lastCamSent and the next watcher finds nothing to do.
        const off = (a) => !a || a.some((v, i) => Math.abs(v - cur[i]) > 1e-6);
        const last = window._plotSync._lastCamSent;
        const storedCam = sc && sc.camera;
        const needBroadcast = off(last);
        const needCommit = off(storedCam && flat(storedCam));
        if (!needBroadcast && !needCommit) return;

        // Out through the same throttled channel as every other frame, dropping
        // anything still queued behind the throttle. The gesture-end frame is
        // usually sitting in that 40 ms timer while this watcher finishes in ~32 ms;
        // broadcasting around it let the older frame land last and leave the targets
        // a fraction behind the source (measured: 1e-3 off, 6/8 drags in the loop
        // harness, reproducibly 8/8 with this watcher disabled).
        if (needBroadcast) {{
            window._plotSync._lastCamSent = cur;
            if (_timer) {{ clearTimeout(_timer); _timer = null; }}
            _pending = {{ mode: "3d", camera: cam }};
            _lastSent = Date.now();
            maybeSend();
        }}

        // Commit it to this page as well: plotly's stored camera is still the
        // mid-ease value, so without this the SOURCE is the one that jumps, on
        // its next Plotly.react, while the targets stay put. The page is already
        // rendering this camera, so the relayout is visually a no-op.
        //
        // Suppressed by a consume-once flag rather than by holding `ignore` over a
        // time window: a window swallows whatever real gesture happens to land
        // inside it, which cost a scroll zoom its final notch (measured: the last
        // wheel of six produced no broadcast at all).
        if (needCommit) {{
            window._plotSync._settleCommit = true;
            const clear = () => {{ window._plotSync._settleCommit = false; }};
            try {{
            const r = Plotly.relayout(gd, {{"scene.camera": cam}});
            if (r && r.then) {{ r.then(clear).catch(clear); }} else {{ clear(); }}
            }} catch (e) {{ clear(); }}
        }}
        }};
        window._plotSync._settleRaf = requestAnimationFrame(step);
    }}

    gd.on('plotly_relayout', function(e) {{
        try {{
        // Decide whether this relayout is the user's BEFORE touching the scalebar:
        // updateScaleBar2D sets _scalebarUpdating synchronously, so a guard read
        // after the call always sees our own update and 2D never broadcasts at all.
        // The settle commit is consumed here rather than timed out, so it can only
        // ever swallow its own single event.
        const settleEcho = window._plotSync._settleCommit;
        if (settleEcho) window._plotSync._settleCommit = false;
        const suppressed = window._plotSync.ignore || window._plotSync._scalebarUpdating || settleEcho;

        // ALWAYS update scalebar for user pan/zoom (skip only if we are updating it ourselves)
        if (!window._plotSync._scalebarUpdating) {{
            updateScaleBar2D(gd);
        }}

        // broadcasting part
        // the gesture is over either way, even if this relayout is one of ours
        window._plotSync.gestureActive = false;
        if (suppressed) return;

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

        // 3-D only, and a no-op if the camera is already at rest
        scheduleSettleBroadcast(gd);
        }} catch(err) {{
        console.log("relayout hook error", err);
        }}
    }});

    // A 2-D pan/zoom emits plotly_relayout only ONCE, at mouse-up, while a 3-D
    // camera drag emits it continuously — which is why 2-D synchronization used to
    // land in one jump at the end of the gesture and 3-D followed live. The live
    // stream for 2-D is plotly_relayouting (38 events for the same drag that
    // produced 1 relayout). Gated on liveSync so only panels with Synchronize
    // ticked pay for it: an unticked panel and the main viewer emit nothing extra.
    gd.on('plotly_relayouting', function(e) {{
        try {{
        if (!window._plotSync.liveSync || window._plotSync.ignore) return;
        if (!window._plotSync.gestureActive) return;  // post-gesture settling, not the user
        // no updateScaleBar2D here on purpose: that is a second relayout per frame,
        // and the gesture-end plotly_relayout above places the bar correctly anyway
        _pending = buildPayload(gd);
        _pending.live = true;

        const now = Date.now();
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
        console.log("relayouting hook error", err);
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
            fig, reset_view, is3d, scalebar_nm = self._pending_fig
            self._pending_fig = None
            self.update_fig(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=scalebar_nm)

    def update_fig(self, fig, reset_view=False, is3d=False, scalebar_nm=None):
        self.ensure_page()
        if not self._plotly_ready:
            # scalebar_nm must be carried too: replaying without it in
            # _on_load_finished would silently fall back to the 100 nm default
            self._pending_fig = (fig, reset_view, is3d, scalebar_nm)
            return
        self._last_fig = fig
        fig_json = pio.to_json(fig, validate=False)
        source_id = self._id
        sb = 100.0 if scalebar_nm is None else float(scalebar_nm)
        live_sync = "true" if getattr(self, "_live_sync", False) else "false"
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
        // re-asserted here because a page reload rebuilds _plotSync with liveSync:false
        window._plotSync.liveSync = {live_sync};
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
        """Apply external view state to this plot (ranges/camera) without re-broadcast.

        Used by MultiColorWindow's per-sample "Synchronize" boxes. `ignore` is held
        for the whole `Plotly.relayout` — clearing it on a fixed timer instead would
        let the applied relayout echo back out and bounce between synced views.
        """
        self.ensure_page()
        if not self._plotly_ready:
            return

        payload_json = json.dumps(payload)
        js = f"""
(function() {{
  const gd = document.getElementById('plot');
  if (!gd || !gd._fullLayout) return "no gd";

  const p = {payload_json};
  // the source's mode can be stale (someone toggled 2D/3D since it was captured),
  // and relayouting xaxis.range on a scene — or scene.camera on a 2-D plot — is junk
  const is3d = !!gd._fullLayout.scene;
  let update = null;
  if (p.mode === "2d" && !is3d && p.xRange && p.yRange) {{
    update = {{
      "xaxis.range": p.xRange,
      "yaxis.range": p.yRange,
      "xaxis.autorange": false,
      "yaxis.autorange": false
    }};
  }} else if (p.mode === "3d" && is3d && p.camera) {{
    update = {{ "scene.camera": p.camera }};
  }}
  if (!update) return "mode mismatch";

  // Coalesce instead of queueing. During a live drag these arrive faster than a
  // big scattergl panel can redraw; without this the backlog grows for as long as
  // the gesture lasts and the panel keeps animating after the mouse stops. Holding
  // only the newest pending update makes the cost self-limiting: a fast machine
  // follows every frame, a slow one drops intermediate frames and still lands on
  // the exact final view, because the last update always gets applied.
  if (window._plotSync._applying) {{
    window._plotSync._queued = update;
    return "coalesced";
  }}

  const run = (u) => {{
    window._plotSync._applying = true;
    window._plotSync.ignore = true;
    const done = () => {{
      const next = window._plotSync._queued;
      window._plotSync._queued = null;
      if (next) {{ run(next); return; }}
      window._plotSync._applying = false;
      setTimeout(() => {{ window._plotSync.ignore = false; }}, 30);
    }};
    try {{
      const r = Plotly.relayout(gd, u);
      if (r && r.then) {{ r.then(done).catch(done); }} else {{ done(); }}
    }} catch (e) {{
      window._plotSync._queued = null;
      window._plotSync._applying = false;
      setTimeout(() => {{ window._plotSync.ignore = false; }}, 30);
    }}
  }};
  run(update);
  return "applied";
}})();
"""
        self.web.page().runJavaScript(js)

    def set_grid_visible(self, on: bool, axis_titles=None):
        """Show/hide the grid on the live page without rebuilding the figure.

        2-D toggles the grid and zero lines. 3-D switches off each part of the
        coordinate system separately and blanks the axis titles, which together
        render identically to hiding the axes — deliberately *not* `visible`,
        even though that is the obvious one-liner: plotly's AxesOptions.merge
        sets `labelEnable=false` for an invisible scene axis and never sets it
        back, so one off/on cycle would cost the X/Y/Z titles for the lifetime
        of the scene. Every attribute used here is re-derived on the way back.

        `axis_titles` is the {axis: text} the themed figure carries, since the
        page cannot know it once the titles have been blanked.
        """
        self.ensure_page()
        if not self._plotly_ready:
            return
        flag = "true" if on else "false"
        titles = json.dumps(axis_titles or {})
        js = f"""
(function() {{
  const gd = document.getElementById('plot');
  if (!gd || !gd._fullLayout) return "no gd";
  const on = {flag};
  const titles = {titles};
  const is3d = !!gd._fullLayout.scene;
  let u;
  if (is3d) {{
    u = {{}};
    ['xaxis', 'yaxis', 'zaxis'].forEach(function(a) {{
      const p = 'scene.' + a + '.';
      // null deletes the key, so plotly re-derives it from the themed figure
      u[p + 'showgrid']       = on ? null : false;
      u[p + 'zeroline']       = on ? null : false;
      u[p + 'showticklabels'] = on ? null : false;
      u[p + 'showline']       = on ? null : false;
      u[p + 'ticks']          = on ? null : '';
      if (!on) {{
        u[p + 'title.text'] = '';
      }} else if (titles[a] !== undefined) {{
        u[p + 'title.text'] = titles[a];
      }}
    }});
  }} else {{
    u = {{ "xaxis.showgrid": on, "yaxis.showgrid": on,
          "xaxis.zeroline": on, "yaxis.zeroline": on }};
  }}
  // not a view change: suppress the broadcast so a synchronized group does not
  // treat a grid toggle as a pan
  window._plotSync.ignore = true;
  const release = () => setTimeout(() => {{ window._plotSync.ignore = false; }}, 30);
  try {{
    const r = Plotly.relayout(gd, u);
    if (r && r.then) {{ r.then(release).catch(release); }} else {{ release(); }}
  }} catch (e) {{
    release();
    return "error: " + e;
  }}
  return "applied";
}})();
"""
        self.web.page().runJavaScript(js)

    def set_live_sync(self, on: bool):
        """Turn the during-drag (plotly_relayouting) broadcast on or off for this view.

        Off by default and re-asserted by update_fig, so a panel without Synchronize
        ticked — and the main viewer, which never calls this — emits nothing extra.
        """
        self._live_sync = bool(on)
        self.ensure_page()
        if not self._plotly_ready:
            return
        flag = "true" if self._live_sync else "false"
        self.web.page().runJavaScript(
            f"(function(){{ if (window._plotSync) window._plotSync.liveSync = {flag}; }})();"
        )

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
def last_valid_cfr_iteration(itr, cfr, default):
    """The last iteration at which the CFR was measured: the last whose values vary.

    *itr* and *cfr* are the columns of a flat (one row per iteration) MINFLUX
    array.  Iterations without a measurement carry a constant (0, -0.0) or NaN;
    NaN is ignored, so an iteration measured for most localizations and skipped
    for some still counts.  *default* when no iteration varies.
    """
    itr = np.asarray(itr)
    cfr = np.asarray(cfr, dtype=float)
    for it in np.sort(np.unique(itr))[::-1]:
        values = cfr[itr == it]
        values = values[np.isfinite(values)]
        if len(values) > 1 and np.std(values) > 0.0:
            return int(it)
    return int(default)


class FileWorker(QtCore.QThread):
    need_efo = Signal(object)  # ctx dict -> GUI should show histogram + scatter and let user choose
    status = Signal(str)
    done_one = Signal(object)  # result dict (final stats row items), None if skipped

    # Every field _process_one_file reads off the loaded array. Checked up front so
    # a non-MINFLUX or truncated .npy is reported as a skipped file instead of
    # surfacing a raw KeyError/IndexError through run()'s except handler.
    REQUIRED_FIELDS = ("tim", "vld", "loc", "itr", "tid", "efo", "cfr")

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

    def _load_interruptible(self, file, chunk=100_000):
        """`np.load` split into cancellable pieces, in either MINFLUX layout.

        A plain `np.load` of a large file is one uninterruptible C call with no
        `_check_cancel()` inside it, so `MainWindow.closeEvent`'s cancel + wait
        times out and Qt destroys a still-running QThread — a hard crash on quit.
        Memory-mapping and copying row blocks gives cancellation somewhere to fire.

        Returns ``(records, info)``; a legacy-layout file (iterations nested
        under `itr`) comes back converted to the flat layout. See
        `core.load_minflux_npy`.
        """
        return load_minflux_npy(file, chunk=chunk, check_cancel=self._check_cancel)

    def _process_one_file(self):
        p = self.params
        file = self.file_path
        # the key is assigned by MainWindow (see _unique_bases); the stem is only a
        # fallback for a caller that did not pass one
        base = p.get("base") or os.path.splitext(os.path.basename(file))[0]
        self.status.emit(f"Loading: {base}")

        try:
            MFX_Data, load_info = self._load_interruptible(file)
        except MinfluxLayoutError as e:
            self.status.emit(f"Skipping {base}: not MINFLUX data — {e}.")
            return None
        if self.isInterruptionRequested():
            return None
        self._check_cancel()
        legacy = load_info.get("layout") == "legacy"
        if legacy:
            self.status.emit(
                f"Loading: {base} (legacy layout, {load_info['n_valid']} valid of "
                f"{load_info['n_localizations']} localizations)"
            )

        # Validate before touching any field. `dtype.names` is None for a plain
        # (non-structured) array, so testing membership against it directly would
        # raise TypeError instead of reporting an unusable file.
        names = MFX_Data.dtype.names or ()
        missing = [f for f in self.REQUIRED_FIELDS if f not in names]
        if missing:
            what = "not a structured array" if not names else f"missing field(s): {', '.join(missing)}"
            self.status.emit(f"Skipping {base}: not MINFLUX data — {what}.")
            return None
        if len(MFX_Data) == 0:
            # a converted legacy file holds valid localizations only
            what = "no valid localizations" if legacy and load_info["n_localizations"] else "file contains no localizations"
            self.status.emit(f"Skipping {base}: {what}.")
            return None
        # loc is indexed as [:, -1] and [:, 1] below
        loc_shape = MFX_Data["loc"].shape
        if len(loc_shape) != 2 or loc_shape[1] < 2:
            self.status.emit(f"Skipping {base}: 'loc' has shape {loc_shape}, expected (N, 2) or (N, 3).")
            return None

        # A converted legacy file has lost its invalid localizations, and with them
        # possibly the first and last time stamps; the loader kept the full span.
        total_tim = load_info.get("total_tim", MFX_Data["tim"][-1] - MFX_Data["tim"][0])
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

        # The CFR is measured at one iteration of the sequence, not at the last:
        # a 3-D sequence measures it at the second-to-last, and the final
        # iteration's `cfr` is a constant (0 or -0.0). Take the last iteration
        # whose CFR varies, whatever the sequence. NaN is ignored: a legacy
        # file writes NaN for iterations a re-localization skipped, and a NaN
        # standard deviation would have thrown that iteration out with them.
        last_cfr_itr = last_valid_cfr_iteration(MFX_Data["itr"], MFX_Data["cfr"], default=last_itr)

        max_itr = np.max(MFX_Data_vld_fnl_filt["itr"])
        efo_vals = MFX_Data_vld_fnl_filt["efo"][MFX_Data_vld_fnl_filt["itr"] == max_itr]

        # get CFR for the final localizations from the last valid CFR iteration.
        # Both frames hold one row per localization of a trace, so the join key is
        # (tid, occurrence within tid): row k of a tid at last_cfr_itr belongs to the
        # same localization cycle as row k of that tid at last_itr. Joining on tid
        # alone would be a per-tid cartesian product (n rows in -> n**2 rows out).
        final_tids = pd.DataFrame({"tid": MFX_Data_vld_fnl_filt["tid"]})
        final_tids["occ"] = final_tids.groupby("tid").cumcount()
        cfr_src = MFX_Data[MFX_Data["itr"] == last_cfr_itr]
        cfr_df = pd.DataFrame({"tid": cfr_src["tid"], "cfr": cfr_src["cfr"]})
        cfr_df["occ"] = cfr_df.groupby("tid").cumcount()

        merged = final_tids.merge(cfr_df, on=["tid", "occ"], how="left")

        # overwrite cfr in the returned array so the histogram and the span filter
        # both see the values from last_cfr_itr instead of the constant final ones
        cfr_arr = merged["cfr"].to_numpy(dtype=float)
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
    # Workers that outlived closeEvent's wait budget. Held at class scope so neither
    # Qt nor Python destroys a thread that is still running (see closeEvent).
    _orphaned_workers = set()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyflux")
        self.resize(1500, 950)
        # napari-storm when it is installed, Plotly otherwise (see rendering_backend)
        self._backend = rendering_backend()
        #: window on the summed image of an additive napari-storm style
        self._contrast = DEFAULT_CONTRAST
        # state
        self._all_files = []
        self._current_index = 0
        self._current_ctx = None
        self._current_worker = None
        self._last_span = (None, None)
        self._current_is_3d = False  # Track current plot mode
        self._arr_by_base = {}  # base -> currently active array for plotting (trace-filtered or EFO-filtered)
        self._base_by_file = {}  # file_path -> base; the authority for every _by_base key
        self._multicolor_win = None
        self._multicolor_crop_bounds = None  # (xmin, xmax, ymin, ymax) for merged-mode crop
        self._multicolor_crop_bounds_by_base = {}  # base -> (xmin, xmax, ymin, ymax) for per-sample crop in grid mode
        self._multicolor_cropped_by_base = {}  # base -> cropped array used only in multicolor viewer

        self._ctx_by_base = {}  # base -> ctx (from worker)
        self._efo_range_by_base = {}  # base -> (xmin, xmax) selected
        self._cfr_range_by_base = {}  # base -> (xmin, xmax) selected
        self._filtered_by_base = {}  # base -> array after EFO+CFR filtering (what will be saved)

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

        self._plot_arr = None  # array currently shown in scatter (may be EFO-filtered)
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
        color_and_controls.addWidget(self.color_panel, 5)  # change here the width of the color settings panel

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

        # What every localization is drawn as: napari-storm's rendering styles,
        # points by default.  A napari-storm without its palette (2.1.0 and
        # earlier) draws Gaussians only, and the menu is locked; without
        # napari-storm at all pyflux draws Plotly markers, and the menu is
        # locked on those.
        style_row = QtWidgets.QHBoxLayout()
        style_row.setContentsMargins(0, 0, 0, 0)
        style_row.addWidget(QtWidgets.QLabel("Render as"), 0)
        self.render_style = QtWidgets.QComboBox()
        self._styles = {style.name: style for style in rendering_styles()} if self.uses_napari_storm else {}
        for style in self._styles.values():
            self.render_style.addItem(style.label, style.name)
            self.render_style.setItemData(
                self.render_style.count() - 1, style.description, Qt.ToolTipRole
            )
            if style.name == "gaussian" and len(self._styles) > 1:
                self.render_style.insertSeparator(self.render_style.count())
        if not self.uses_napari_storm:
            self.render_style.addItem("Markers (Plotly)", "plotly")
            self.render_style.setEnabled(False)
            self.render_style.setToolTip(
                "napari-storm is not installed, so localizations are drawn as\n"
                "Plotly markers. Install napari-storm (pip install -r\n"
                "requirements-napari-storm.txt) for points, Gaussians and the\n"
                "other rendering styles."
            )
        elif len(self._styles) > 1:
            self.render_style.setCurrentIndex(self.render_style.findData(default_style()))
            self.render_style.setToolTip(
                "What every localization is drawn as. Gaussian (scientific) sums\n"
                "them like a super-resolution image; the rest are napari-storm's\n"
                "alternative visualisations."
            )
        else:
            self.render_style.setCurrentIndex(self.render_style.findData(default_style()))
            self.render_style.setEnabled(False)
            self.render_style.setToolTip(
                "Other styles need napari-storm with its footprint palette\n"
                "(newer than 2.1.0); this installation draws Gaussians."
            )
        self.render_style.currentIndexChanged.connect(lambda _: self._on_render_style_changed())
        style_row.addWidget(self.render_style, 1)
        right_controls.addLayout(style_row)

        # Contrast of the summed image: how many overlapping localizations
        # reach the full colour.  Only an additive napari-storm style (the
        # Gaussian, the sprites) sums anything; markers are opaque and Plotly
        # has no summed image, so the control is greyed out for those.
        contrast_row = QtWidgets.QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.addWidget(QtWidgets.QLabel("Contrast"), 0)
        self.contrast_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, CONTRAST_SLIDER_STEPS)
        self.contrast_slider.setValue(contrast_slider_position(self._contrast[1]))
        self.contrast_slider.setToolTip(
            "Brightness window of the summed image: the number of overlapping\n"
            "localizations drawn at the full colour. Lower is brighter.\n"
            "Applies to the Gaussian and the glowing sprites; points are opaque."
        )
        self.contrast_slider.valueChanged.connect(self._on_contrast_slider_moved)
        contrast_row.addWidget(self.contrast_slider, 1)
        self.contrast_max = QtWidgets.QDoubleSpinBox()
        self.contrast_max.setRange(CONTRAST_MIN, CONTRAST_MAX)
        self.contrast_max.setDecimals(2)
        self.contrast_max.setValue(self._contrast[1])
        self.contrast_max.setToolTip(self.contrast_slider.toolTip())
        self.contrast_max.setFixedWidth(72)
        self.contrast_max.valueChanged.connect(self._on_contrast_max_edited)
        contrast_row.addWidget(self.contrast_max, 0)
        right_controls.addLayout(contrast_row)

        self.color_panel.set_alpha_enabled(not self._render_style_opaque())
        self._sync_contrast_enabled()

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
        mc_lay.setSpacing(0)  # tighter default spacing between items

        # divider "Multicolor"
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

        self._mbm_dir_by_base = {}  # base -> mbm folder
        self._mbm_source_base = None  # selected source base
        self._mbm_enabled = False

        self._raw_arr_by_base = {}  # base -> current array (EFO-filtered if applied)
        self._aligned_arr_by_base = {}  # base -> aligned array (same filtering), computed when enabled
        self._T_by_base = {}  # base -> 4x4 transform to source

        # rendering panel: the current dataset, drawn by napari-storm on an
        # embedded napari canvas, or by Plotly in a web view
        if self.uses_napari_storm:
            plot_box = QtWidgets.QGroupBox("Rendering (napari-storm)")
        else:
            plot_box = QtWidgets.QGroupBox("Scatter plot")
        bottom.addWidget(plot_box, 1)
        pv = QtWidgets.QVBoxLayout(plot_box)
        self._main_specs = {}  # napari-storm: what the main view shows
        if self.uses_napari_storm:
            self.plot_view = NapariStormView(title="pyflux")
            self.plot_view.viewChanged.connect(self.on_any_plot_view_changed)
            self.plot_view.ndisplayChanged.connect(self._on_view_ndisplay_changed)
        else:
            self.plot_view = PlotlyView()
            self.plot_view.viewChanged.connect(self.on_any_plot_view_changed)
        pv.addWidget(self.plot_view)
        # status
        self.statusBar().showMessage("Ready.")

    def open_binning_window(self):
            if self._binning_win is None:
                self._binning_win = BinningWindow(self, self)

            bases = self.all_bases()
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

    def reset_session(self):
        """Reset all analysis state so a new folder starts from scratch.

        Called from browse_folder when the folder actually changes, so *every*
        per-base dict must be cleared — including the user configuration that
        reload_all_data deliberately keeps. Base keys are only unique within one
        folder (see _unique_bases), so a surviving entry silently attaches to an
        unrelated dataset that happens to share a stem, and `_mbm_dir_by_base`
        would still point into the previous folder (`_ensure_mbm_defaults` only
        fills in keys that are missing or empty, so a stale path is never
        overwritten). refresh_files repopulates all of these right after.
        """
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
        self._multicolor_crop_bounds_by_base.clear()
        self._multicolor_cropped_by_base.clear()

        # per-base user configuration: meaningless in a new folder, and the MBM
        # paths would still resolve into the old one
        self._color_settings_by_base.clear()
        self._mbm_dir_by_base.clear()
        self._base_by_file.clear()

        self._mbm_enabled = False
        self._mbm_source_base = None

        # the span selectors are rebuilt from these, so stale bounds would draw a
        # selection over the new folder's histograms (the labels below already reset)
        self._span_efo_xmin = self._span_efo_xmax = None
        self._span_cfr_xmin = self._span_cfr_xmax = None

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
            if self.uses_napari_storm:
                self._main_specs = {}
                self.plot_view.clear()
            else:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
                apply_plot_theme(fig, is3d=False)
                self.plot_view.update_fig(fig, reset_view=True, is3d=False)
        except Exception:
            pass

        # re-enable Run
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Ready.")

    def reload_all_data(self, reason=""):
        """Drop every derived result and reload the .npy files from disk.

        min_trace_len / z_corr / scale are baked into the cached arrays by
        FileWorker, so changing one invalidates everything downstream: the EFO/CFR
        ranges were picked against the old values, and _apply_mbm_alignment reads
        scale/z_corr fresh for the bead transforms — leaving the old arrays in place
        would register beads and localizations in different units.

        Keeps the file selection, the MBM folders, the MBM source and the per-base
        colour settings; those are user configuration, not derived data. Only the
        current file is reloaded eagerly — the rest reload on selection, since
        clearing _ctx_by_base is what on_file_changed keys off.
        """
        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
                self._current_worker.wait(1000)
        except Exception:
            pass
        self._current_worker = None

        # alignment off
        self._mbm_enabled = False
        self._aligned_arr_by_base.clear()
        self._T_by_base.clear()

        # filtering off
        self._efo_range_by_base.clear()
        self._cfr_range_by_base.clear()
        self._filtered_by_base.clear()
        self._span_efo_xmin = self._span_efo_xmax = None
        self._span_cfr_xmin = self._span_cfr_xmax = None
        self.efo_range_lbl.setText("Selected EFO range: —")
        self.cfr_range_lbl.setText("Selected CFR range: —")

        # cached loads
        self._ctx_by_base.clear()
        self._raw_arr_by_base.clear()
        self._arr_by_base.clear()
        self._current_ctx = None
        self._plot_arr = None
        self._plot_is_filtered = False

        # crops were drawn in the old coordinates
        self._multicolor_crop_bounds = None
        self._multicolor_crop_bounds_by_base.clear()
        self._multicolor_cropped_by_base.clear()

        self.statusBar().showMessage(f"Reloading after {reason} change…" if reason else "Reloading…")
        self.start_worker_for_current()

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
            base = self.base_of(fp)
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

        prop = str(settings.get("property") or DEFAULT_PROPERTY)
        limits = settings.get("limits")
        if limits is not None:
            try:
                low, high = limits
                limits = (None if low is None else float(low), None if high is None else float(high))
            except (TypeError, ValueError):
                limits = None

        self._color_settings_by_base[base] = {
            "mode": mode,
            "solid": solid,
            "lut": lut,
            "alpha": alpha,
            "size": size,
            "property": prop,
            "limits": limits,
        }
        self._sync_contrast_enabled()

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

        dlg.set_mbm_rows(self._all_files, self._mbm_dir_by_base, self.all_bases())  # <-- required
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # update, not replace: the table only holds rows for _all_files, so a
            # wholesale assignment would drop every path when no folder is loaded
            self._mbm_dir_by_base.update(dlg.mbm_map())
            vals = dlg.values()

            # compare before writing the new values into the spinboxes: these three
            # are consumed by FileWorker at load time and baked into every cached
            # array, so a change means the data on screen is stale
            changed = [
                label
                for label, new, old in (
                    ("min trace length", int(vals["min_trace_len"]), int(self.min_trace.value())),
                    ("z correction", float(vals["z_corr"]), float(self.zcorr.value())),
                    ("scale", float(vals["scale"]), float(self.scale.value())),
                )
                if new != old
            ]

            self.cfr_bin_count.setValue(vals["cfr_bin_count"])
            self.min_trace.setValue(vals["min_trace_len"])
            self.zcorr.setValue(vals["z_corr"])
            self.scale.setValue(vals["scale"])
            self.bin_size.setValue(vals["bin_size"])
            self._scalebar_nm = float(vals["scalebar_nm"])

            if changed and self._ctx_by_base:
                self.reload_all_data(reason=" / ".join(changed))
                return

            self.redraw_scatter(reset_view=False)
            self._refresh_multicolor_contents(reset_view=False)
            if self._binning_win is not None and self._binning_win.isVisible():
                self._binning_win.refresh_plot(keep_view=True)

            # if a histogram is currently shown, refreshing it to new bin size can be helpful:
            if self._current_ctx is not None:
                self.on_need_efo(self._current_ctx)

    @property
    def uses_napari_storm(self):
        """True when this window draws with napari-storm rather than Plotly."""
        return self._backend == BACKEND_NAPARI_STORM

    def _render_footprint(self):
        """The chosen rendering style, a name from napari-storm's palette."""
        if not self.uses_napari_storm:
            return None
        return self.render_style.currentData() or "gaussian"

    def _render_style_opaque(self):
        style = self._styles.get(self._render_footprint())
        return bool(style is not None and style.opaque)

    def _contrast_applies(self):
        """True while the contrast window can change what is drawn.

        That takes an additive napari-storm style, and a dataset whose values
        are summed: every solid-coloured one, and the colour-coded ones only
        on a napari-storm that blends them (see
        `napari_storm_view.color_coded_supported`).
        """
        if not self.uses_napari_storm or self._render_style_opaque():
            return False
        if color_coded_supported():
            return True
        bases = self.all_bases() if hasattr(self, "file_combo") else list(self._color_settings_by_base)
        modes = {self._color_settings_by_base.get(b, {}).get("mode", DEFAULT_MODE) for b in bases}
        return MODE_SOLID in modes

    def _sync_contrast_enabled(self):
        on = self._contrast_applies()
        self.contrast_slider.setEnabled(on)
        self.contrast_max.setEnabled(on)

    def _on_render_style_changed(self):
        # α means something for the additive styles only; markers are opaque
        self.color_panel.set_alpha_enabled(not self._render_style_opaque())
        self._sync_contrast_enabled()
        self._refresh_all_plots_same_data()

    def _on_contrast_slider_moved(self, position):
        self._set_contrast_max(contrast_from_slider(position), from_slider=True)

    def _on_contrast_max_edited(self, value):
        self._set_contrast_max(float(value), from_slider=False)

    def _set_contrast_max(self, upper, *, from_slider):
        """Apply a new upper window limit to every open napari-storm view.

        The views are re-windowed in place -- no request is rebuilt -- so the
        slider can be dragged over a large dataset.
        """
        upper = max(CONTRAST_MIN, min(CONTRAST_MAX, float(upper)))
        self._contrast = (0.0, upper)
        if from_slider:
            self.contrast_max.blockSignals(True)
            self.contrast_max.setValue(upper)
            self.contrast_max.blockSignals(False)
        else:
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(contrast_slider_position(upper))
            self.contrast_slider.blockSignals(False)
        if not self.uses_napari_storm:
            return
        try:
            self.plot_view.set_contrast(self._contrast, self._main_specs)
        except Exception:
            pass
        if self._multicolor_win is not None:
            try:
                self._multicolor_win.set_contrast(self._contrast)
            except Exception:
                pass

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
                on_ndisplay_changed=self._on_view_ndisplay_changed,
                backend=self._backend,
            )
            self._multicolor_win.set_contrast(self._contrast)

        bases = self.all_bases()
        self._multicolor_win.rebuild(bases)

        mode = "merged" if getattr(self, "merged_chk", None) and self.merged_chk.isChecked() else "grid"
        self._multicolor_win.set_mode(mode)

        self._multicolor_win.show()
        self._multicolor_win.raise_()
        self._multicolor_win.activateWindow()

        # first draw once the window has its size, so reset_view frames the data
        QtCore.QTimer.singleShot(0, lambda: self._refresh_multicolor_contents(reset_view=True))
        if not self.uses_napari_storm:
            # and once more after the QWebEngineViews have finished bootstrapping
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

        # whatever is in _arr_by_base (trace-filtered or EFO-filtered), cropped
        # if a crop is active, in the order of _all_files
        arr_by_base = {}
        for f in self._all_files:
            base = self.base_of(f)
            arr = self._get_multicolor_arr_for_base(base)
            if arr is not None:
                arr_by_base[base] = arr

        if not self.uses_napari_storm:
            if self._multicolor_win.mode() == "merged":
                fig = make_plotly_fig_merged(
                    arr_by_base, avg_tid=avg, is3d=is3d,
                    color_settings_by_base=self._color_settings_by_base, scalebar_nm=sb,
                )
                self._multicolor_win.update_merged(fig, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)
            else:
                figs = {
                    base: make_plotly_fig(arr, avg, is3d, color_settings=self._color_settings_by_base.get(base), scalebar_nm=sb)
                    for base, arr in arr_by_base.items()
                }
                self._multicolor_win.update_all(figs, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)
            return

        if self._multicolor_win.mode() == "merged":
            # one depth scale for all channels, so the same z is the same colour
            specs = build_render_specs(
                arr_by_base, avg, is3d, self._color_settings_by_base,
                shared_depth_range=True, footprint=self._render_footprint(), contrast=self._contrast,
            )
            self._multicolor_win.update_merged(specs, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)
        else:
            specs = build_render_specs(
                arr_by_base, avg, is3d, self._color_settings_by_base,
                footprint=self._render_footprint(), contrast=self._contrast,
            )
            self._multicolor_win.update_all(specs, reset_view=reset_view, is3d=is3d, scalebar_nm=sb)

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

        bounds = (xmin, xmax, ymin, ymax)
        # "bases" is set when the box-selected sample has Synchronize ticked, so the
        # crop lands on the whole synchronized group; "base" alone is one sample;
        # neither means merged mode, i.e. the global crop
        crop_bases = payload.get("bases")
        if not crop_bases:
            one = payload.get("base")
            crop_bases = [one] if one is not None else []
        if crop_bases:
            for b in crop_bases:
                self._multicolor_crop_bounds_by_base[b] = bounds
        else:
            self._multicolor_crop_bounds = bounds
        self._recompute_multicolor_crop_cache()
        self._refresh_multicolor_contents(reset_view=True)
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()

        n = int(sum(len(v) for v in self._multicolor_cropped_by_base.values() if v is not None))
        label = ", ".join(crop_bases) if crop_bases else "all"
        self.statusBar().showMessage(f"Crop ({label}): x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}], kept {n} locs")

    def _on_multicolor_reset_crop_requested(self):
        if not self._has_any_crop():
            return
        self._multicolor_crop_bounds = None
        self._multicolor_crop_bounds_by_base.clear()
        self._multicolor_cropped_by_base.clear()
        self._refresh_multicolor_contents(reset_view=True)
        if self._dbscan_win is not None and self._dbscan_win.isVisible():
            self._dbscan_win.refresh_current_plot()
        self.statusBar().showMessage("Multicolor crop reset")

    def _has_any_crop(self):
        """True if any crop is active: the global one from a merged-mode box-select,
        or a per-base one from a grid-mode box-select.

        This is the condition `_get_multicolor_arr_for_base` tests before serving
        `_multicolor_cropped_by_base`, so anything that refreshes that cache must
        test the same thing. DBSCAN's two refresh sites used to check only the
        global crop and so served a stale cache for per-base crops.
        """
        return bool(
            getattr(self, "_multicolor_crop_bounds", None)
            or getattr(self, "_multicolor_crop_bounds_by_base", None)
        )

    def _recompute_multicolor_crop_cache(self):
        self._multicolor_cropped_by_base = {}
        global_bounds = getattr(self, "_multicolor_crop_bounds", None)
        per_base_bounds = getattr(self, "_multicolor_crop_bounds_by_base", {})
        if not global_bounds and not per_base_bounds:
            return

        for f in self._all_files:
            base = self.base_of(f)
            bounds = per_base_bounds.get(base, global_bounds)
            if not bounds:
                continue
            arr = self._get_arr_for_base(base)
            if arr is None:
                continue
            if len(arr) == 0:
                self._multicolor_cropped_by_base[base] = arr
                continue

            xmin, xmax, ymin, ymax = bounds
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
        if getattr(self, "_multicolor_crop_bounds", None) or base in getattr(self, "_multicolor_crop_bounds_by_base", {}):
            cropped = self._multicolor_cropped_by_base.get(base)
            if cropped is not None:
                return cropped
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
        cs = self._color_settings_by_base.get(base)
        if self.uses_napari_storm:
            item = build_render_spec(
                arr, self.avg_tid.isChecked(), self.is3d.isChecked(), cs, name=base,
                footprint=self._render_footprint(), contrast=self._contrast,
            )
        else:
            item = make_plotly_fig(
                arr, self.avg_tid.isChecked(), self.is3d.isChecked(), color_settings=cs,
                scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
            )
        self._multicolor_win.update_one(
            base, item,
            reset_view=reset_view,
            is3d=self.is3d.isChecked(),
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )

    def _multicolor_rebuild_if_open(self):
        if self._multicolor_win is None or not self._multicolor_win.isVisible():
            return
        bases = self.all_bases()
        self._multicolor_win.rebuild(bases)

    def _dbscan_rebuild_if_open(self):
        if self._dbscan_win is None or not self._dbscan_win.isVisible():
            return
        bases = self._dbscan_available_bases()
        self._dbscan_win.rebuild(bases)

    def _dbscan_available_bases(self):
        # Match multicolor loading behavior: include bases that currently resolve
        # to an array through _get_multicolor_arr_for_base/_get_arr_for_base.
        ordered_all = self.all_bases()

        if self._has_any_crop():
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

    def _on_view_ndisplay_changed(self, ndisplay):
        """napari's own 2D/3D button was pressed in one of the viewers: follow it.

        The 3D box is the single authority for how every viewer draws, so it
        is set here and its handler redraws everything, which brings the other
        viewers to the same mode too.
        """
        want = int(ndisplay) == 3
        if self.is3d.isChecked() != want:
            self.is3d.setChecked(want)

    # ---------------- UI actions ----------------
    def browse_folder(self):
        old = self.data_edit.text().strip()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select MINFLUX data folder")
        if path:
            path = os.path.normpath(path)
            if os.path.normpath(old) != path:
                self.reset_session()  # <-- key line

            self.data_edit.setText(path)
            self.set_default_output()
            self.refresh_files()

    def set_default_output(self):
        p = self.data_edit.text().strip()
        if p:
            self.out_edit.setText(p + "_filtered")

    def closeEvent(self, event):
        self._closing = True

        # Cancel *every* worker, not only the current one: rapid file switching can
        # leave several running (see start_worker_for_current) and they are all
        # parented to this window, so ~MainWindow would delete a live QThread.
        workers = [w for w in list(getattr(self, "_workers", ())) if w is not None]
        if self._current_worker is not None and self._current_worker not in workers:
            workers.append(self._current_worker)
        for w in workers:
            try:
                if w.isRunning():
                    w.cancel()
            except Exception:
                pass
        deadline = QtCore.QDeadlineTimer(2000)
        for w in workers:
            try:
                if not w.isRunning():
                    continue
                w.wait(max(0, deadline.remainingTime()))
                if w.isRunning():
                    # blew the budget anyway: detach so Qt does not destroy a running
                    # thread, and keep a reference so Python does not collect it either
                    w.setParent(None)
                    MainWindow._orphaned_workers.add(w)
                    w.finished.connect(lambda w=w: MainWindow._orphaned_workers.discard(w))
            except Exception:
                pass
        self._current_worker = None

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

    @staticmethod
    def _unique_bases(file_paths, root=None):
        """Map file path -> dataset key, the key every `_by_base` dict is keyed on.

        `refresh_files` globs recursively, so two subfolders can hold a `cell1.npy`
        each; keying on the plain basename stem made them overwrite each other's
        arrays, filters, colours and MBM paths — and collide again on export.

        A stem that occurs once keeps its plain name, so output filenames and the
        `<dir>/grd/mbm` defaults are unchanged for the common case. Only colliding
        stems fall back to the flattened relative path (`A__cell1`), which is unique
        because the paths are. The counter covers the residual case where that
        flattened name happens to equal some other file's plain stem.
        """
        stems = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
        counts = Counter(stems)

        # reserve the unaffected stems first so they are never mangled
        used = {s for s in stems if counts[s] == 1}
        keys = []
        for p, stem in zip(file_paths, stems):
            if counts[stem] == 1:
                keys.append(stem)
                continue
            rel = os.path.relpath(p, root) if root else p
            cand = os.path.splitext(rel)[0].replace(os.sep, "__").replace("/", "__").strip("._")
            key, n = cand or stem, 2
            while key in used:
                key = f"{cand or stem}_{n}"
                n += 1
            used.add(key)
            keys.append(key)
        return dict(zip(file_paths, keys))

    def base_of(self, path):
        """Dataset key for a file path. Falls back to the plain stem if the map is
        stale, so a caller can never silently get None as a dict key."""
        return self._base_by_file.get(path) or os.path.splitext(os.path.basename(path))[0]

    def all_bases(self):
        return [self.base_of(f) for f in self._all_files]

    def refresh_files(self):
        data_path = self.data_edit.text().strip()
        if not data_path or not os.path.isdir(data_path):
            self._all_files = []
        else:
            files = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
            files.sort()
            self._all_files = files

        # must precede every consumer below — they all key off base_of()
        self._base_by_file = self._unique_bases(self._all_files, data_path or None)
        n_disambiguated = sum(
            1 for f, b in self._base_by_file.items()
            if b != os.path.splitext(os.path.basename(f))[0]
        )
        if n_disambiguated:
            self.statusBar().showMessage(
                f"{n_disambiguated} file(s) share a name with another; "
                "their subfolder was added to keep them separate."
            )

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
        bases = self.all_bases()
        # a source picked in a previous folder no longer exists here, and leaving it
        # set would make _apply_mbm_alignment abort on a base that has no data
        if self._mbm_source_base is not None and self._mbm_source_base not in bases:
            self._mbm_source_base = None
        self.color_panel.rebuild(bases, self._color_settings_by_base, self._mbm_source_base)
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
        base = self.base_of(f)

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

        f = self._all_files[self._current_index]
        params = dict(
            min_trace_len=int(self.min_trace.value()),
            z_corr=float(self.zcorr.value()),
            scale=float(self.scale.value()),
            save_folder=self.out_edit.text().strip(),
            bin_size=float(self.bin_size.value()),
            base=self.base_of(f),
        )

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
        self._plot_arr = ctx["arr"]  # start from trace-filtered data
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

        # the columns this file can be coloured by
        try:
            self.color_panel.set_properties(base, property_columns(arr))
        except Exception:
            pass

        # Ensure caches exist
        if not hasattr(self, "_raw_arr_by_base") or self._raw_arr_by_base is None:
            self._raw_arr_by_base = {}
        if not hasattr(self, "_arr_by_base") or self._arr_by_base is None:
            self._arr_by_base = {}

        # ---- cache as "raw" (trace-filtered) for this base ----
        self._raw_arr_by_base[base] = arr
        self._arr_by_base[base] = arr

        # ---- EFO histogram ----
        # hist_edges replaces a bare np.arange(min, max + bin_size, bin_size): that
        # drops every point when all values are identical, raises on NaN/inf, and
        # allocates millions of edges for a small bin size on a wide range.
        hist_notes = []
        self.ax_efo.clear()
        efo_vals, efo_bins, efo_note = hist_edges(efo_vals, bin_size)
        if efo_note:
            hist_notes.append(f"EFO histogram: {efo_note}")
        if len(efo_vals) > 0:
            self.ax_efo.hist(
                efo_vals,
                bins=efo_bins,
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
        # self.ax_efo.set_xlabel("EFO")
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
        # bins is a count here, so the edge-count blowup cannot happen and numpy
        # widens the range itself when every value is identical — but a non-finite
        # value still makes it raise ("autodetected range ... is not finite")
        n_bad_cfr = int(np.count_nonzero(~np.isfinite(cfr_vals)))
        if n_bad_cfr:
            cfr_vals = cfr_vals[np.isfinite(cfr_vals)]
            hist_notes.append(f"CFR histogram: {n_bad_cfr} non-finite value(s) ignored")
        if hist_notes:
            self.statusBar().showMessage(" | ".join(hist_notes))
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
        # self.ax_cfr.set_xlabel("CFR")
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

        cs = self._color_settings_by_base.get(base)
        is3d = self.is3d.isChecked()
        if not self.uses_napari_storm:
            fig = make_plotly_fig(
                arr_to_plot, self.avg_tid.isChecked(), is3d, color_settings=cs,
                scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
            )
            self.update_plotly_fig(fig, reset_view=reset_view)
            return
        spec = build_render_spec(
            arr_to_plot, self.avg_tid.isChecked(), is3d, cs, name=base,
            footprint=self._render_footprint(), contrast=self._contrast,
        )
        self._main_specs = {base: spec}
        self.plot_view.set_scalebar_nm(getattr(self, "_scalebar_nm", 100.0))
        self.plot_view.show_datasets({base: spec}, is3d=is3d, reset_view=reset_view)

    def update_plotly_fig(self, fig, reset_view=False):
        """Plotly: push *fig* to the main view, with the 2-D scale bar."""
        if (not self.is3d.isChecked()) and fig is not None:
            add_scalebar_2d(fig, length_nm=float(getattr(self, "_scalebar_nm", 100.0)))
        self.plot_view.update_fig(
            fig,
            reset_view=reset_view,
            is3d=self.is3d.isChecked(),
            scalebar_nm=getattr(self, "_scalebar_nm", 100.0),
        )

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
                avg_df["n"] = df.groupby("tid")[f"loc_{dims[0]}"] .count()
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

        def abort(msg):
            """Leave a consistent "not aligned" state.

            _mbm_enabled must never be True alongside an empty _aligned_arr_by_base:
            every downstream `if mbm_enabled` would take the aligned branch and look
            for arrays that were never built. Reachable in practice by choosing a
            source whose file has not been loaded — on_mbm_align_clicked checks that
            *some* data is loaded, not that the source itself has any.
            """
            if self._mbm_enabled:
                # a previous run succeeded and rewrote _arr_by_base; undo that too
                self._remove_mbm_alignment()
            else:
                self._mbm_enabled = False
                self._aligned_arr_by_base.clear()
                self._T_by_base.clear()
            self.statusBar().showMessage(msg)

        src = self._mbm_source_base
        if not src:
            return abort("MBM: no source dataset selected")

        # align the EFO-filtered arrays (what will be saved)
        src_arr = self._filtered_by_base.get(src)
        if src_arr is None:
            src_arr = self._arr_by_base.get(src)
        if src_arr is None:
            return abort(f"MBM: no loaded data for source {src} — run that file first")

        mbm_src = self._mbm_dir_by_base.get(src)
        if not mbm_src or not os.path.isdir(mbm_src):
            return abort(f"MBM: source folder missing for {src}")

        # only now is alignment actually going to happen
        self._mbm_enabled = True
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
        # Alignment is only a coordinate transform layered on top of the EFO/CFR
        # filtered arrays, so dropping it restores the filtered data — not the
        # pre-filter raw data, which would put filtered-out points back on screen
        # while save_all() and BinningWindow still read _filtered_by_base.
        for base, raw in self._raw_arr_by_base.items():
            arr = self._filtered_by_base.get(base)
            self._arr_by_base[base] = raw if arr is None else arr
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
