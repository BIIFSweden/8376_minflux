# tests/test_rendering_backend.py
#
# napari-storm is optional. With it, pyflux draws on embedded napari canvases
# and offers the contrast window; without it (or with PYFLUX_RENDERER=plotly)
# it draws the Plotly scatter plots it always did.

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


from pyflux.napari_storm_view import DEFAULT_CONTRAST, napari_storm_available  # noqa: E402


def _minflux(n=200, ntid=20, seed=0):
    rng = np.random.default_rng(seed)
    dt = [
        ("tim", "f8"), ("vld", "?"), ("loc", "f8", (3,)), ("itr", "i4"),
        ("tid", "i4"), ("efo", "f8"), ("cfr", "f8"),
    ]
    a = np.zeros(n, dtype=dt)
    a["tid"] = np.sort(rng.integers(0, ntid, n))
    a["loc"] = rng.uniform(0, 1000, (n, 3))
    a["loc"][:, 1] *= -1
    a["vld"] = True
    a["tim"] = np.arange(n, dtype=float)
    a["efo"] = rng.uniform(1e4, 1e5, n)
    a["cfr"] = rng.uniform(0, 1, n)
    return a


def _ctx(base, arr):
    return dict(
        base=base, arr=arr, efo_vals=arr["efo"], cfr_vals=arr["cfr"], total_tim=1.0,
        total_loc=len(arr), last_iteration_loc=len(arr), after_trace=len(arr),
    )


def _app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _spin(app, n=5):
    for _ in range(n):
        app.processEvents()


def _main_window_with(app, bases):
    from pyflux.gui import MainWindow

    win = MainWindow()
    win._all_files = [f"/data/{b}.npy" for b in bases]
    win.color_panel.rebuild(list(bases), {}, None)
    for i, base in enumerate(bases):
        win.on_need_efo(_ctx(base, _minflux(seed=i)))
    _spin(app)
    return win


def _close(app, win):
    win.close()
    _spin(app)


# ----------------------------
# choosing the renderer
# ----------------------------

def test_the_environment_can_force_plotly(monkeypatch):
    monkeypatch.setenv("PYFLUX_RENDERER", "plotly")
    assert napari_storm_available() is False


def test_without_napari_storm_the_renderer_is_plotly(monkeypatch):
    import builtins

    monkeypatch.delenv("PYFLUX_RENDERER", raising=False)
    real_import = builtins.__import__

    def no_napari_storm(name, *args, **kwargs):
        if name.startswith("napari_storm"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_napari_storm)
    assert napari_storm_available() is False
    import pyflux.gui as gui

    assert gui.rendering_backend() == gui.BACKEND_PLOTLY


# ----------------------------
# the Plotly path
# ----------------------------

def test_the_main_window_draws_with_plotly_when_told_to(monkeypatch):
    pytest.importorskip("pytestqt")
    monkeypatch.setenv("PYFLUX_RENDERER", "plotly")
    app = _app()
    from pyflux.gui import PlotlyView

    win = _main_window_with(app, ["a", "b"])
    try:
        assert not win.uses_napari_storm
        assert isinstance(win.plot_view, PlotlyView)
        assert not win.render_style.isEnabled()
        assert win._render_footprint() is None
        # the contrast window has no summed image to act on
        assert not win.contrast_slider.isEnabled() and not win.contrast_max.isEnabled()
        # α still means something: Plotly markers take an opacity
        assert win.color_panel._alpha_enabled is True

        pushed = []
        monkeypatch.setattr(win.plot_view, "update_fig", lambda fig, **kw: pushed.append((fig, kw)))
        win.redraw_scatter(reset_view=True)
        fig, kw = pushed[-1]
        assert fig.data and kw["reset_view"] is True and kw["is3d"] is False
        assert any(getattr(s, "type", "") == "line" for s in fig.layout.shapes)  # the 2-D scale bar

        win.is3d.setChecked(True)
        fig, kw = pushed[-1]
        assert kw["is3d"] is True and fig.data[0].type == "scatter3d"
    finally:
        _close(app, win)


def test_the_multicolor_window_draws_plotly_figures(monkeypatch):
    pytest.importorskip("pytestqt")
    monkeypatch.setenv("PYFLUX_RENDERER", "plotly")
    app = _app()
    from pyflux.gui import PlotlyView

    win = _main_window_with(app, ["a", "b"])
    try:
        win.open_multicolor()
        _spin(app)
        mc = win._multicolor_win
        assert not mc.uses_napari_storm
        assert isinstance(mc.merged_view, PlotlyView)
        assert all(isinstance(v, PlotlyView) for v in mc._views.values())
        assert mc._grid_chk.text() == "Grid" and mc._grid_on

        pushed = {}
        for base, view in mc._views.items():
            monkeypatch.setattr(view, "update_fig", lambda fig, b=base, **kw: pushed.__setitem__(b, (fig, kw)))
        win._refresh_multicolor_contents(reset_view=True)
        assert set(pushed) == {"a", "b"}
        assert all(fig.data for fig, _kw in pushed.values())

        merged = []
        monkeypatch.setattr(mc.merged_view, "update_fig", lambda fig, **kw: merged.append(fig))
        win.merged_chk.setChecked(True)
        win.on_merged_toggled()
        assert merged and len(merged[-1].data) == 2

        # Crop takes the last box selection, as it always did with Plotly
        from PySide6 import QtWidgets

        messages = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: messages.append(a[2]))
        crops = []
        mc._on_crop_requested = crops.append
        mc._crop_current_selection()  # nothing selected yet: a message, no crop
        assert crops == [] and messages and "Box select" in messages[-1]
        mc._on_plot_selection(mc.merged_view._id, {"mode": "2d", "xRange": [10, 20], "yRange": [-5, 5]})
        mc._crop_current_selection()
        assert crops[-1]["xRange"] == [10.0, 20.0] and crops[-1]["yRange"] == [-5.0, 5.0]
    finally:
        _close(app, win)


# ----------------------------
# the napari-storm path: the contrast window
# ----------------------------

def test_the_contrast_window_follows_the_rendering_style():
    pytest.importorskip("pytestqt")
    pytest.importorskip("napari")
    if not napari_storm_available():
        pytest.skip("napari-storm is not importable here")
    app = _app()
    from pyflux.gui import contrast_from_slider, contrast_slider_position

    win = _main_window_with(app, ["a"])
    try:
        assert win.uses_napari_storm
        combo = win.render_style
        if combo.findData("disc") >= 0:
            combo.setCurrentIndex(combo.findData("disc"))
            assert not win.contrast_slider.isEnabled()  # points are opaque
        combo.setCurrentIndex(combo.findData("gaussian"))
        assert win._contrast == DEFAULT_CONTRAST
        # the default colour mode is depth, whose values a napari-storm that
        # does not blend colour-coded localizations windows per localization,
        # so the slider only wakes up once a dataset's values are summed
        from pyflux.napari_storm_view import color_coded_supported

        assert win.contrast_slider.isEnabled() == color_coded_supported()
        win.color_panel._widgets_by_base["a"]["mode"].setCurrentText("solid")
        assert win._color_settings_by_base["a"]["mode"] == "solid"
        assert win.contrast_slider.isEnabled() and win.contrast_max.isEnabled()

        # spin and slider agree, and the open layer is re-windowed in place
        win.contrast_max.setValue(25.0)
        assert win._contrast == (0.0, 25.0)
        assert win.contrast_slider.value() == contrast_slider_position(25.0)
        view = win.plot_view
        assert view.renderer.appearance(view._ids["a"]).contrast_limits == (0.0, 25.0)

        win.contrast_slider.setValue(contrast_slider_position(4.0))
        upper = contrast_from_slider(win.contrast_slider.value())
        assert win.contrast_max.value() == pytest.approx(upper, abs=0.01)  # the spin shows two decimals
        assert view.renderer.appearance(view._ids["a"]).contrast_limits[1] == pytest.approx(upper)

        # a redraw carries the window along
        win.redraw_scatter(reset_view=False)
        assert win._main_specs["a"].contrast_limits[1] == pytest.approx(upper)
    finally:
        _close(app, win)


def test_the_slider_is_logarithmic_from_one_to_a_thousand_overlaps():
    from pyflux.gui import CONTRAST_MAX, CONTRAST_MIN, CONTRAST_SLIDER_STEPS, contrast_from_slider, contrast_slider_position

    assert contrast_from_slider(0) == pytest.approx(CONTRAST_MIN)
    assert contrast_from_slider(CONTRAST_SLIDER_STEPS) == pytest.approx(CONTRAST_MAX)
    for upper in (0.1, 1.0, 10.0, 250.0):
        assert contrast_from_slider(contrast_slider_position(upper)) == pytest.approx(upper, rel=0.02)
