"""Localization rendering through napari-storm.

pyflux used to draw its scatter plots with Plotly inside a QtWebEngineView.
This module replaces that with napari-storm: every localization is drawn as a
Gaussian on a napari canvas that lives *inside* the pyflux window -- no
separate napari window, no browser engine, no JavaScript bridge.

Two halves, deliberately separated:

* :func:`build_render_spec` turns a pyflux MINFLUX array plus one row of the
  colour-settings table into what napari-storm draws.  Pure numpy -- it imports
  neither Qt nor napari -- so it is unit-testable headlessly.
* :class:`NapariStormView` is the widget.  It owns one napari viewer whose
  canvas is reparented into pyflux's layout, and offers the same shape of API
  ``PlotlyView`` had: push datasets, read and apply the camera, report the
  visible rectangle for cropping, shut down.

Coordinates
-----------
pyflux stores ``loc`` in nanometres with *y negated*, so that Plotly's y-up
axes showed the sample the way Imspector does.  napari's canvas is y-down, so
the negation is undone here and the picture keeps the same orientation.  The
crop rectangle reported by :meth:`NapariStormView.visible_rect_2d` is
translated back into pyflux's convention, so the rest of the application never
sees napari's.
"""

import hashlib
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .plotting import (
    DEFAULT_ALPHA,
    DEFAULT_LUT,
    DEFAULT_MODE,
    DEFAULT_SIZE_2D,
    DEFAULT_SOLID,
    LUT_CHOICES,
    MODE_DEPTH,
    MODE_E2E,
    MODE_SOLID,
    MODE_TID,
    SOLID_COLOR_MAP,
    _resolve_mpl_cmap,
    scatter_points_and_color,
)

#: Below this much z extent (nm) a dataset is drawn flat, whatever the 3D box
#: says.  A 2-D MINFLUX acquisition writes z as float noise around 1e-13 m, not
#: as an exact zero.
FLAT_Z_EXTENT_NM = 1.0

#: Every dataset is drawn with the same contrast window, so a per-localization
#: value in [0, 1] is the colormap position, not an intensity to be rescaled.
CONTRAST_LIMITS = (0.0, 1.0)

#: Colormap for ``tid`` mode: one hue per trace, cyclic so the hash can land
#: anywhere in [0, 1).
TID_COLORMAP = "hsv"


@dataclass(frozen=True)
class ColormapSpec:
    """A colormap as data, so the planning half needs no napari import."""

    name: str
    colors: np.ndarray  # (N, 4) RGBA in [0, 1]


@dataclass(frozen=True)
class RenderSpec:
    """One dataset, ready for :meth:`NapariStormView.show_datasets`."""

    request: object  # napari_storm.core.RenderRequest
    colormap: ColormapSpec
    opacity: float
    contrast_limits: tuple = CONTRAST_LIMITS
    zdim_present: bool = False

    @property
    def n_localizations(self):
        return int(len(self.request.coords))


# ----------------------------------------------------------------------
# Colormaps
# ----------------------------------------------------------------------


def _hex_to_rgb(hexcol):
    s = hexcol.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def solid_colormap(solid):
    """Black-to-colour ramp for one of pyflux's solid colours."""
    if solid not in SOLID_COLOR_MAP:
        solid = DEFAULT_SOLID
    r, g, b = _hex_to_rgb(SOLID_COLOR_MAP[solid])
    colors = np.array([[0.0, 0.0, 0.0, 1.0], [r, g, b, 1.0]], dtype=float)
    return ColormapSpec(name=f"pyflux solid {solid}", colors=colors)


def lut_colormap(lut, steps=256):
    """One of pyflux's LUT choices, sampled the way its heatmaps sample it.

    Goes through :func:`pyflux.plotting._resolve_mpl_cmap`, which builds the
    colormap from the Plotly colorscale definition, so a rendering here uses
    exactly the colours the binning window would -- including the ``cu_*``
    custom ramps, which no colormap registry knows by name.
    """
    if lut not in LUT_CHOICES:
        lut = DEFAULT_LUT
    cmap = _resolve_mpl_cmap(lut)
    colors = np.asarray(cmap(np.linspace(0.0, 1.0, int(steps))), dtype=float)
    colors[:, 3] = 1.0
    return ColormapSpec(name=f"pyflux lut {lut}", colors=colors)


# ----------------------------------------------------------------------
# Per-localization colour values
# ----------------------------------------------------------------------


def _normalized(values, low=None, high=None):
    """Scale to [0, 1].  A constant column maps to 0 rather than to NaN."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=np.float32)
    if low is None:
        low = float(np.min(values[finite]))
    if high is None:
        high = float(np.max(values[finite]))
    if not (high > low):
        return np.zeros(len(values), dtype=np.float32)
    out = (values - low) / (high - low)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tid_color_values(tids):
    """Deterministic pseudo-random colormap position per trace id.

    The same recipe as :func:`pyflux.plotting.tid_to_color` -- an MD5 of the
    integer id -- so a trace keeps its colour across redraws, filters and
    sessions, and neighbouring ids do not get neighbouring hues.
    """
    tids = np.asarray(tids)
    unique, inverse = np.unique(tids, return_inverse=True)
    positions = np.empty(len(unique), dtype=np.float32)
    for i, tid in enumerate(unique):
        digest = hashlib.md5(str(int(tid)).encode("utf-8")).digest()
        positions[i] = int.from_bytes(digest[:2], "big") / 65536.0
    return positions[inverse.ravel()]


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------


def _settings_with_defaults(color_settings):
    cs = dict(color_settings or {})
    cs.setdefault("mode", DEFAULT_MODE)
    cs.setdefault("solid", DEFAULT_SOLID)
    cs.setdefault("lut", DEFAULT_LUT)
    cs.setdefault("alpha", DEFAULT_ALPHA)
    cs.setdefault("size", DEFAULT_SIZE_2D)
    return cs


def scatter_xyz(arr, avg_tid):
    """``(xyz, end_to_end, tids)`` for *arr*, non-finite rows dropped.

    The same rows Plotly was given, so switching renderer changes how the
    localizations are drawn and not which ones.
    """
    xyz, vals, tids = scatter_points_and_color(arr, avg_tid)
    if xyz is None or len(xyz) == 0:
        return None, None, None
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        return None, None, None
    if xyz.shape[1] < 3:
        xyz = np.column_stack([xyz, np.zeros(len(xyz))])
    vals = np.asarray(vals, dtype=np.float64) if vals is not None else np.zeros(len(xyz))
    tids = np.asarray(tids)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(vals)
    if not finite.all():
        xyz, vals, tids = xyz[finite], vals[finite], tids[finite]
    if len(xyz) == 0:
        return None, None, None
    return xyz, vals, tids


def build_render_spec(arr, avg_tid, is3d, color_settings=None, *, name="pyflux", depth_range=None):
    """What napari-storm should draw for one pyflux dataset.

    Args:
        arr: structured MINFLUX array with ``loc`` (N, 3) in nm and ``tid``,
            exactly what pyflux hands its Plotly builders.
        avg_tid: draw one localization per trace (its centroid) instead of all.
        is3d: give the localizations their z.  Off, everything is drawn on one
            plane, which is what the 2-D scatter plot showed.
        color_settings: one row of the colour-settings table -- ``mode``,
            ``solid``, ``lut``, ``alpha`` and ``size``.  ``size`` is the
            Gaussian sigma in nanometres; it was the marker size in pixels.
        name: layer name in the viewer.
        depth_range: ``(zmin, zmax)`` to colour depth against, so several
            datasets shown together share one scale.  None uses this
            dataset's own range.

    Returns None when there is nothing to draw.
    """
    from napari_storm.core import DatasetTraits, GaussianSettings, LocalizationTable, RenderPlanner

    cs = _settings_with_defaults(color_settings)
    xyz, end_to_end, tids = scatter_xyz(arr, avg_tid)
    if xyz is None:
        return None

    x = xyz[:, 0]
    y = -xyz[:, 1]  # undo pyflux's Plotly-facing flip; see the module docstring
    z = xyz[:, 2]

    z_extent = float(np.max(z) - np.min(z))
    zdim_present = bool(is3d) and z_extent > FLAT_Z_EXTENT_NM

    records = np.rec.fromarrays([x, y, z], names=("x", "y", "z"))
    table = LocalizationTable(
        records,
        position_columns={"x": "x", "y": "y", "z": "z"},
        position_scale_nm=1.0,
        copy=False,
    )

    sigma = float(cs["size"])
    sigma = max(sigma, 0.1)
    settings = GaussianSettings(mode=0, fixed_sigma_xy_nm=sigma, fixed_sigma_z_nm=sigma)
    traits = DatasetTraits(zdim_present=zdim_present)
    request = RenderPlanner().plan(table, settings, traits, name=str(name))

    mode = cs["mode"]
    if mode == MODE_SOLID:
        colormap = solid_colormap(cs["solid"])
        values = np.ones(len(x), dtype=np.float32)
    elif mode == MODE_DEPTH:
        colormap = lut_colormap(cs["lut"])
        low, high = (None, None) if depth_range is None else depth_range
        values = _normalized(z, low, high)
    elif mode == MODE_TID:
        colormap = lut_colormap(TID_COLORMAP)
        values = tid_color_values(tids)
    else:  # MODE_E2E
        colormap = lut_colormap(cs["lut"])
        values = _normalized(end_to_end)

    request = replace(request, values=np.ascontiguousarray(values, dtype=np.float32))

    alpha = float(cs["alpha"])
    alpha = max(0.0, min(1.0, alpha))
    return RenderSpec(
        request=request,
        colormap=colormap,
        opacity=alpha,
        contrast_limits=CONTRAST_LIMITS,
        zdim_present=zdim_present,
    )


def depth_range_of(arr_by_base, avg_tid):
    """The z range spanned by every dataset together, or None if there is none."""
    low = high = None
    for arr in arr_by_base.values():
        xyz, _vals, _tids = scatter_xyz(arr, avg_tid)
        if xyz is None:
            continue
        z = xyz[:, 2]
        lo, hi = float(np.min(z)), float(np.max(z))
        low = lo if low is None else min(low, lo)
        high = hi if high is None else max(high, hi)
    if low is None:
        return None
    return (low, high)


def build_render_specs(arr_by_base, avg_tid, is3d, color_settings_by_base, *, shared_depth_range=False):
    """One :class:`RenderSpec` per base, in the given order.

    With *shared_depth_range* every depth-coloured dataset is scaled against
    the z range of all of them, which is what the merged multicolor view needs
    so that the same depth is the same colour in every channel.  Bases with
    nothing to draw map to None, so the caller can close them.
    """
    depth_range = None
    if shared_depth_range:
        depth_bases = {
            b: a
            for b, a in arr_by_base.items()
            if _settings_with_defaults((color_settings_by_base or {}).get(b)).get("mode") == MODE_DEPTH
        }
        depth_range = depth_range_of(depth_bases, avg_tid) if depth_bases else None

    specs = {}
    for base, arr in arr_by_base.items():
        cs = (color_settings_by_base or {}).get(base)
        specs[base] = build_render_spec(
            arr, avg_tid, is3d, cs, name=base, depth_range=depth_range
        )
    return specs


# ----------------------------------------------------------------------
# The widget
# ----------------------------------------------------------------------


def _make_view_class():
    """Build the widget class lazily, so importing this module needs no Qt."""
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Signal

    class NapariStormView(QtWidgets.QWidget):
        """A napari canvas, embedded, drawing pyflux datasets through napari-storm.

        The napari main window is created and never shown; its viewer widget
        is taken out of it and placed in this widget's layout.  The window has
        to stay alive -- it owns the viewer, and dropping it takes the canvas
        with it -- so the viewer is held for the widget's lifetime and given
        back on :meth:`shutdown`.
        """

        #: (source_id, payload) -- the camera moved.  The payload holds
        #: ``mode`` ("2d"/"3d"), ``center``, ``zoom``, ``angles``,
        #: ``perspective``; :meth:`apply_view` takes the same dict back.
        viewChanged = Signal(str, object)
        #: (source_id) -- the pointer entered or pressed on this canvas.  The
        #: multicolor window crops whichever panel was touched last.
        activated = Signal(str)
        #: (ndisplay) -- the user switched 2-D/3-D from napari's own button.
        ndisplayChanged = Signal(int)

        def __init__(self, parent=None, title="napari-storm"):
            super().__init__(parent)
            import napari
            from napari.qt import get_stylesheet
            from napari_storm.napari_particles.selection import select_renderer

            self._id = hex(id(self))
            self._ids = {}  # base -> dataset id handed to the renderer
            self._next_id = 1  # never reused: the renderer keys resources by it
            self._is3d = None
            self._applying_view = False
            self._live_sync = False
            self._colormaps = {}
            self._closed = False

            self.viewer = napari.Viewer(show=False, show_welcome_screen=False, title=title)
            self._qt_viewer = self.viewer.window._qt_viewer
            self._napari_center = self._qt_viewer.parentWidget()
            # The stylesheet napari's main window would have cascaded down.
            try:
                self._qt_viewer.setStyleSheet(get_stylesheet(self.viewer.theme))
            except Exception:
                pass

            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self._qt_viewer)

            self.renderer = select_renderer(self.viewer)

            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.length = 100.0
            # napari < 0.8 reads the unit off the overlay; setting it warns of
            # the per-layer units that replace it in 0.8, which napari-storm
            # does not support yet, so the warning is noise here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.viewer.scale_bar.unit = "nm"
                except Exception:
                    pass
            # The axes overlay is a gizmo at the world origin, not a plot
            # grid; off unless the host asks for it.
            self.viewer.axes.visible = False

            camera_events = self.viewer.camera.events
            for field in ("center", "zoom", "angles", "perspective"):
                getattr(camera_events, field).connect(self._on_camera_changed)
            self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

            self._canvas_native = self._qt_viewer.canvas.native
            self._canvas_native.installEventFilter(self)

        # ------------------------------------------------------------ data

        def show_datasets(self, specs, is3d, reset_view=False):
            """Draw exactly the datasets in *specs* (base -> RenderSpec | None).

            Bases already open are updated in place; new ones are opened;
            open ones missing from *specs*, or mapped to None, are closed.
            """
            from napari_storm.core import LayerAppearance

            is3d = bool(is3d)
            if self._is3d is not None and is3d != self._is3d:
                # Flat and 3-D data are different geometry to the instanced
                # layer (its quad sits on the data plane), so switch by
                # rebuilding rather than by updating.
                self.clear()
            self._is3d = is3d

            for base, spec in specs.items():
                if spec is None:
                    self._close_base(base)
                    continue
                colormap = self._colormap(spec.colormap)
                dataset_id = self._ids.get(base)
                if dataset_id is not None and self.renderer.is_open(dataset_id):
                    self.renderer.update(dataset_id, spec.request)
                else:
                    dataset_id = self._next_id
                    self._next_id += 1
                    self._ids[base] = dataset_id
                    self.renderer.open(dataset_id, replace(spec.request, colormap=colormap))
                self._refresh_extent(dataset_id)
                self.renderer.set_appearance(
                    dataset_id,
                    LayerAppearance(
                        colormap=colormap,
                        opacity=float(spec.opacity),
                        contrast_limits=tuple(spec.contrast_limits),
                        visible=True,
                    ),
                )

            for base in [b for b in self._ids if b not in specs]:
                self._close_base(base)

            self._label_axes()
            self._set_ndisplay(3 if is3d else 2)
            if reset_view:
                self.reset_view()

        def clear(self):
            """Close every dataset; the canvas goes empty."""
            try:
                self.renderer.close_all()
            except Exception:
                pass
            self._ids.clear()

        def _close_base(self, base):
            dataset_id = self._ids.pop(base, None)
            if dataset_id is not None:
                try:
                    self.renderer.close(dataset_id)
                except Exception:
                    pass

        def open_bases(self):
            """The bases currently drawn, in the order they were opened."""
            return [b for b, i in self._ids.items() if self.renderer.is_open(i)]

        def _colormap(self, spec):
            """The napari colormap for a :class:`ColormapSpec`, built once."""
            cmap = self._colormaps.get(spec.name)
            if cmap is None:
                from napari.utils.colormaps import Colormap

                cmap = Colormap(colors=np.asarray(spec.colors, dtype=float), name=spec.name)
                self._colormaps[spec.name] = cmap
            return cmap

        def _refresh_extent(self, dataset_id):
            """Drop napari's cached extent for a layer whose rows just changed.

            napari caches a layer's extent and clears it in its own data
            setter, which the in-place update bypasses; left alone, reset_view
            would frame the previous rows. Clearing it here also keeps the
            layer list's cached extent -- and the units it reports to the
            canvas -- in step with the layers actually present.
            """
            layer = self.renderer.layer(dataset_id)
            if layer is None:
                return
            try:
                layer.refresh(thumbnail=False, data_displayed=False, highlight=False, extent=True)
            except TypeError:
                try:
                    layer.refresh()  # napari < 0.5 takes no arguments
                except Exception:
                    pass
            except Exception:
                pass

        def _label_axes(self):
            """Name the world axes, so the axes overlay reads x/y/z, not -1/-2/-3."""
            dims = self.viewer.dims
            labels = ("z", "y", "x")[-int(dims.ndim):]
            try:
                if tuple(dims.axis_labels) != labels:
                    dims.axis_labels = labels
            except Exception:
                pass

        # ------------------------------------------------------------ view

        def _set_ndisplay(self, ndisplay):
            if int(self.viewer.dims.ndisplay) != int(ndisplay):
                self.viewer.dims.ndisplay = int(ndisplay)

        def _on_ndisplay_changed(self, event=None):
            self.ndisplayChanged.emit(int(self.viewer.dims.ndisplay))

        def reset_view(self):
            try:
                self.viewer.reset_view()
            except Exception:
                pass

        def set_grid_visible(self, on, axis_titles=None):
            """napari has no plot grid; its axes overlay is the coordinate system."""
            self.viewer.axes.visible = bool(on)

        def set_scalebar_nm(self, length_nm):
            try:
                self.viewer.scale_bar.length = float(length_nm)
            except Exception:
                pass

        def set_live_sync(self, on):
            """Kept for API parity with PlotlyView.

            napari reports every camera move as it happens, so a synchronized
            panel always follows live; there is no separate gesture-end event
            to hold the broadcast back for.
            """
            self._live_sync = bool(on)

        def view_payload(self):
            cam = self.viewer.camera
            return {
                "mode": "3d" if int(self.viewer.dims.ndisplay) == 3 else "2d",
                "center": [float(c) for c in cam.center],
                "zoom": float(cam.zoom),
                "angles": [float(a) for a in cam.angles],
                "perspective": float(cam.perspective),
            }

        def _on_camera_changed(self, event=None):
            if self._applying_view or self._closed:
                return
            self.viewChanged.emit(self._id, self.view_payload())

        def apply_view(self, payload):
            """Move this camera to *payload* without re-broadcasting it."""
            if not isinstance(payload, dict):
                return
            is3d = int(self.viewer.dims.ndisplay) == 3
            mode = payload.get("mode")
            if mode is not None and (mode == "3d") != is3d:
                return  # captured in the other display mode; meaningless here
            self._applying_view = True
            try:
                cam = self.viewer.camera
                if payload.get("center") is not None:
                    cam.center = tuple(float(c) for c in payload["center"])
                if payload.get("zoom") is not None:
                    cam.zoom = float(payload["zoom"])
                if is3d:
                    if payload.get("angles") is not None:
                        cam.angles = tuple(float(a) for a in payload["angles"])
                    if payload.get("perspective") is not None:
                        cam.perspective = float(payload["perspective"])
            finally:
                self._applying_view = False

        def visible_rect_2d(self):
            """``(xmin, xmax, ymin, ymax)`` on screen now, in pyflux's coordinates.

            None in 3-D display, where a rectangle on screen is not a
            rectangle in the sample.
            """
            if int(self.viewer.dims.ndisplay) != 2:
                return None
            try:
                height, width = self._qt_viewer.canvas.size
            except Exception:
                return None
            cam = self.viewer.camera
            zoom = float(cam.zoom)
            if not (zoom > 0) or width <= 0 or height <= 0:
                return None
            cy, cx = float(cam.center[-2]), float(cam.center[-1])
            half_w = float(width) / (2.0 * zoom)
            half_h = float(height) / (2.0 * zoom)
            xmin, xmax = cx - half_w, cx + half_w
            # napari y is pyflux's y negated (module docstring)
            ymin, ymax = -(cy + half_h), -(cy - half_h)
            return (xmin, xmax, ymin, ymax)

        def eventFilter(self, obj, event):
            if obj is self._canvas_native and event.type() in (
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.Enter,
            ):
                self.activated.emit(self._id)
            return False

        # ------------------------------------------------------------ end

        def shutdown(self):
            if self._closed:
                return
            self._closed = True
            # The canvas would draw once more while its GL context is torn
            # down and log an error for it; stop the draw events first.
            try:
                self._qt_viewer.canvas._scene_canvas.events.draw.block()
                self._canvas_native.setUpdatesEnabled(False)
            except Exception:
                pass
            self.clear()
            try:
                self.renderer.detach()
            except Exception:
                pass
            # Hand the viewer widget back to the window that owns it, so the
            # window's own close tears everything down in the order it built it.
            try:
                self._canvas_native.removeEventFilter(self)
                self.layout().removeWidget(self._qt_viewer)
                if self._napari_center is not None and self._napari_center.layout() is not None:
                    self._napari_center.layout().addWidget(self._qt_viewer)
            except Exception:
                pass
            try:
                self.viewer.close()
            except Exception:
                pass

    return NapariStormView


_VIEW_CLASS = None


def NapariStormView(*args, **kwargs):
    """Construct the embedded viewer widget (see :func:`_make_view_class`)."""
    global _VIEW_CLASS
    if _VIEW_CLASS is None:
        _VIEW_CLASS = _make_view_class()
    return _VIEW_CLASS(*args, **kwargs)
