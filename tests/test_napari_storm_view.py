# tests/test_napari_storm_view.py
#
# The napari-storm rendering path. The planning half is exercised without Qt
# or napari; the widget itself gets one end-to-end check that needs a display
# and is skipped where there is none.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pyflux.napari_storm_view import (  # noqa: E402
    CONTRAST_LIMITS,
    FLAT_Z_EXTENT_NM,
    POINT_MIN_PX,
    build_render_spec,
    build_render_specs,
    default_style,
    depth_range_of,
    lut_colormap,
    points_supported,
    rendering_styles,
    solid_colormap,
    tid_color_values,
)
from pyflux.plotting import LUT_CHOICES, SOLID_COLOR_CHOICES  # noqa: E402


def _minflux(n=120, ntid=12, seed=0, zspan=400.0):
    """The shape of array pyflux hands its plot builders: nm, y negated."""
    rng = np.random.default_rng(seed)
    dt = [
        ("tim", "f8"), ("vld", "?"), ("loc", "f8", (3,)), ("itr", "i4"),
        ("tid", "i4"), ("efo", "f8"), ("cfr", "f8"),
    ]
    a = np.zeros(n, dtype=dt)
    a["tid"] = np.sort(rng.integers(0, ntid, n))
    a["loc"][:, 0] = rng.uniform(0, 5000, n)
    a["loc"][:, 1] = -rng.uniform(0, 4000, n)
    a["loc"][:, 2] = rng.uniform(-zspan / 2, zspan / 2, n)
    a["vld"] = True
    a["tim"] = np.arange(n, dtype=float)
    a["efo"] = rng.uniform(1e4, 1e5, n)
    a["cfr"] = rng.uniform(0, 1, n)
    return a


def _cs(**kw):
    base = {"mode": "solid", "solid": "cyan", "lut": "turbo", "alpha": 0.8, "size": 5}
    base.update(kw)
    return base


# ----------------------------
# coordinates
# ----------------------------

def test_nothing_to_draw_gives_none():
    a = _minflux()
    assert build_render_spec(None, False, False, _cs()) is None
    assert build_render_spec(a[:0], False, False, _cs()) is None


def test_coords_are_zyx_with_y_unflipped():
    a = _minflux()
    spec = build_render_spec(a, False, True, _cs())
    coords = spec.request.coords
    assert coords.shape == (len(a), 3)
    # pyflux negates y for Plotly's y-up axes; napari is y-down, so it is
    # negated back and x is untouched
    assert np.allclose(coords[:, 2], a["loc"][:, 0], atol=1e-2)
    assert np.allclose(coords[:, 1], -a["loc"][:, 1], atol=1e-2)
    assert np.allclose(coords[:, 0], a["loc"][:, 2], atol=1e-2)
    assert spec.zdim_present


def test_2d_pins_every_localization_to_one_plane():
    a = _minflux()
    spec = build_render_spec(a, False, False, _cs())
    assert np.ptp(spec.request.coords[:, 0]) == 0.0
    assert not spec.zdim_present
    # x and y are still the data
    assert np.allclose(spec.request.coords[:, 2], a["loc"][:, 0], atol=1e-2)


def test_flat_data_stays_flat_in_3d():
    a = _minflux(zspan=0.0)
    a["loc"][:, 2] += np.random.default_rng(1).normal(0, FLAT_Z_EXTENT_NM / 10, len(a))
    spec = build_render_spec(a, False, True, _cs())
    assert not spec.zdim_present
    assert np.ptp(spec.request.coords[:, 0]) == 0.0


def test_non_finite_rows_are_dropped():
    a = _minflux()
    a["loc"][3, 0] = np.nan
    a["loc"][7, 2] = np.inf
    spec = build_render_spec(a, False, True, _cs())
    assert len(spec.request.coords) == len(a) - 2
    assert np.isfinite(spec.request.coords).all()


def test_avg_tid_draws_one_gaussian_per_trace():
    a = _minflux()
    spec = build_render_spec(a, True, True, _cs())
    assert len(spec.request.coords) == len(np.unique(a["tid"]))


def test_size_is_the_gaussian_width_in_nm():
    a = _minflux()
    narrow = build_render_spec(a, False, True, _cs(size=5)).request
    wide = build_render_spec(a, False, True, _cs(size=20)).request
    # the billboard edge carries the physical scale on every napari-storm release
    assert wide.size == pytest.approx(4.0 * narrow.size)
    assert (wide.sigmas >= narrow.sigmas).all()


# ----------------------------
# colour
# ----------------------------

def test_solid_mode_is_a_black_to_colour_ramp_at_full_value():
    a = _minflux()
    spec = build_render_spec(a, False, True, _cs(mode="solid", solid="orange", alpha=0.6))
    assert np.all(spec.request.values == 1.0)
    assert spec.opacity == pytest.approx(0.6)
    assert spec.contrast_limits == CONTRAST_LIMITS
    colors = spec.colormap.colors
    assert np.allclose(colors[0, :3], 0.0)
    assert np.allclose(colors[-1, :3], (1.0, 0x88 / 255, 0.0))


def test_every_solid_colour_and_lut_choice_builds():
    for solid in SOLID_COLOR_CHOICES:
        c = solid_colormap(solid).colors
        assert c.shape == (2, 4) and np.all((c >= 0) & (c <= 1))
    for lut in LUT_CHOICES:
        c = lut_colormap(lut).colors
        assert c.shape == (256, 4) and np.all((c >= 0) & (c <= 1))
        assert np.all(c[:, 3] == 1.0)


def test_depth_mode_maps_z_onto_the_colormap():
    a = _minflux()
    spec = build_render_spec(a, False, True, _cs(mode="depth", lut="viridis"))
    v = spec.request.values
    assert v.min() == pytest.approx(0.0) and v.max() == pytest.approx(1.0)
    z = a["loc"][:, 2]
    assert v[np.argmin(z)] == pytest.approx(0.0)
    assert v[np.argmax(z)] == pytest.approx(1.0)
    assert spec.colormap.name.endswith("viridis")


def test_depth_mode_colours_z_even_when_drawn_flat():
    a = _minflux()
    spec = build_render_spec(a, False, False, _cs(mode="depth"))
    assert np.ptp(spec.request.coords[:, 0]) == 0.0
    assert spec.request.values.max() == pytest.approx(1.0)


def test_depth_range_can_be_shared_across_datasets():
    a = _minflux(zspan=400.0)
    b = _minflux(seed=5, zspan=100.0)
    own = build_render_specs({"a": a, "b": b}, False, True,
                             {"a": _cs(mode="depth"), "b": _cs(mode="depth")})
    shared = build_render_specs({"a": a, "b": b}, False, True,
                                {"a": _cs(mode="depth"), "b": _cs(mode="depth")},
                                shared_depth_range=True)
    assert own["b"].request.values.max() == pytest.approx(1.0)
    # b spans a quarter of the shared range, so it cannot reach the ends
    assert shared["b"].request.values.max() < 0.8
    assert shared["b"].request.values.min() > 0.2
    lo, hi = depth_range_of({"a": a, "b": b}, False)
    assert lo == pytest.approx(min(a["loc"][:, 2].min(), b["loc"][:, 2].min()))
    assert hi == pytest.approx(max(a["loc"][:, 2].max(), b["loc"][:, 2].max()))


def test_tid_mode_gives_each_trace_one_stable_colour():
    a = _minflux()
    spec = build_render_spec(a, False, True, _cs(mode="tid"))
    v = spec.request.values
    assert np.all((v >= 0) & (v < 1))
    for tid in np.unique(a["tid"]):
        assert np.ptp(v[a["tid"] == tid]) == 0.0
    assert np.array_equal(tid_color_values(a["tid"]), tid_color_values(a["tid"]))
    assert len(np.unique(tid_color_values(np.arange(50)))) == 50


def test_end_to_end_mode_normalises_the_trace_displacement():
    a = _minflux()
    spec = build_render_spec(a, False, True, _cs(mode="end-to-end"))
    v = spec.request.values
    assert v.min() == pytest.approx(0.0) and v.max() == pytest.approx(1.0)
    for tid in np.unique(a["tid"]):
        assert np.ptp(v[a["tid"] == tid]) == 0.0


def test_missing_settings_fall_back_to_defaults():
    a = _minflux()
    spec = build_render_spec(a, False, True, None)
    assert spec is not None
    assert 0.0 <= spec.opacity <= 1.0


# ----------------------------
# rendering styles
# ----------------------------

def test_points_are_the_default():
    spec = build_render_spec(_minflux(), False, True, _cs())
    assert spec.footprint == "disc"


def test_any_style_can_be_asked_for():
    gaussian = build_render_spec(_minflux(), False, True, _cs(), footprint="gaussian")
    assert gaussian.footprint == "gaussian"
    # the size is the same number either way: radius or sigma, in nm
    assert gaussian.request.size == build_render_spec(_minflux(), False, True, _cs()).request.size
    assert build_render_spec(_minflux(), False, True, _cs(), footprint="ring").footprint == "ring"


def test_the_render_style_reaches_every_spec():
    arrs = {"a": _minflux(seed=1), "b": _minflux(seed=2)}
    for footprint in ("disc", "gaussian", "sphere"):
        specs = build_render_specs(arrs, False, True, {}, footprint=footprint, shared_depth_range=True)
        assert {spec.footprint for spec in specs.values()} == {footprint}


def test_the_styles_are_napari_storms_palette():
    napari_storm_core = pytest.importorskip("napari_storm.core")
    if not hasattr(napari_storm_core, "PALETTE"):
        pytest.skip("the installed napari-storm has no palette")
    styles = rendering_styles()
    assert [s.name for s in styles] == [f.name for f in napari_storm_core.PALETTE]
    assert styles[0].name == "gaussian" and not styles[0].opaque
    assert {s.name for s in styles if s.opaque} >= {"disc", "ring", "sphere"}
    assert default_style() == "disc"


def test_a_napari_storm_without_the_palette_draws_gaussians(monkeypatch):
    napari_storm_core = pytest.importorskip("napari_storm.core")
    monkeypatch.delattr(napari_storm_core, "PALETTE", raising=False)
    assert [s.name for s in rendering_styles()] == ["gaussian"]
    assert default_style() == "gaussian"
    assert points_supported() is False


# ----------------------------
# the widget
# ----------------------------

def test_view_draws_updates_and_closes_datasets():
    pytest.importorskip("pytestqt")
    pytest.importorskip("napari")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyflux.napari_storm_view import NapariStormView

    a, b = _minflux(seed=1), _minflux(seed=2)
    view = NapariStormView()
    view.resize(400, 300)
    view.show()
    for _ in range(10):
        app.processEvents()
    try:
        specs = {
            "a": build_render_spec(a, False, False, _cs(), name="a"),
            "b": build_render_spec(b, False, False, _cs(mode="depth"), name="b"),
        }
        view.show_datasets(specs, is3d=False, reset_view=True)
        assert sorted(view.open_bases()) == ["a", "b"]
        assert view.viewer.dims.ndisplay == 2
        assert view.renderer.layer(view._ids["a"]).n_localizations == len(a)

        rect = view.visible_rect_2d()
        assert rect is not None and all(np.isfinite(rect))
        xmin, xmax, ymin, ymax = rect
        # after reset_view the data centre is on screen, in pyflux's own frame
        assert xmin < a["loc"][:, 0].mean() < xmax
        assert ymin < a["loc"][:, 1].mean() < ymax

        seen = []
        view.viewChanged.connect(lambda sid, payload: seen.append(payload))
        view.viewer.camera.zoom = view.viewer.camera.zoom * 1.5
        assert seen and seen[-1]["mode"] == "2d"
        n = len(seen)
        view.apply_view({"mode": "2d", "center": [1.0, 100.0, 200.0], "zoom": 0.5})
        assert len(seen) == n  # applied views are not re-broadcast

        view.show_datasets(
            {"a": build_render_spec(a, False, True, _cs(), name="a")}, is3d=True
        )
        assert view.open_bases() == ["a"]
        assert view.viewer.dims.ndisplay == 3
        assert view.visible_rect_2d() is None
    finally:
        view.shutdown()
        for _ in range(5):
            app.processEvents()


def _view_with(specs, is3d=False):
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyflux.napari_storm_view import NapariStormView

    view = NapariStormView()
    view.show_datasets(specs, is3d=is3d, reset_view=True)
    for _ in range(5):
        app.processEvents()
    return view


def test_the_view_draws_points_by_default_and_any_style_on_request():
    pytest.importorskip("napari")
    if not points_supported():
        pytest.skip("the installed napari-storm cannot draw points")
    a = _minflux(seed=1)
    view = _view_with({"a": build_render_spec(a, False, False, _cs(), name="a")})
    try:
        dataset_id = view._ids["a"]
        appearance = view.renderer.appearance(dataset_id)
        assert appearance.footprint == "disc"
        assert appearance.min_size_px == POINT_MIN_PX

        for footprint in ("gaussian", "ring", "glow"):
            view.show_datasets(
                {"a": build_render_spec(a, False, False, _cs(), name="a", footprint=footprint)},
                is3d=False,
            )
            assert view.renderer.appearance(dataset_id).footprint == footprint
    finally:
        view.shutdown()


def test_without_point_support_the_view_draws_gaussians(monkeypatch):
    pytest.importorskip("napari")
    import pyflux.napari_storm_view as nsv

    monkeypatch.setattr(
        nsv, "rendering_styles", lambda: (nsv.RenderStyle("gaussian", "Gaussian", False),)
    )
    view = _view_with({"a": build_render_spec(_minflux(), False, False, _cs(), name="a")})
    try:
        assert view.points_supported is False
        appearance = view.renderer.appearance(view._ids["a"])
        assert getattr(appearance, "footprint", "gaussian") in (None, "gaussian")
    finally:
        view.shutdown()


def _main_window():
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyflux.gui import MainWindow

    return MainWindow()


def test_the_main_window_offers_the_palette_with_points_first():
    pytest.importorskip("napari")
    if not points_supported():
        pytest.skip("the installed napari-storm cannot draw points")
    win = _main_window()
    try:
        combo = win.render_style
        offered = [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i)]
        assert offered == [s.name for s in rendering_styles()]
        assert combo.isEnabled()
        assert win._render_footprint() == "disc"
        assert win.color_panel._alpha_enabled is False  # α does nothing for markers

        combo.setCurrentIndex(combo.findData("glow"))
        assert win._render_footprint() == "glow"
        assert win.color_panel._alpha_enabled is True  # an additive style
        combo.setCurrentIndex(combo.findData("sphere"))
        assert win.color_panel._alpha_enabled is False
    finally:
        win.plot_view.shutdown()
        win.deleteLater()


def test_the_main_window_falls_back_to_gaussians(monkeypatch):
    pytest.importorskip("napari")
    import pyflux.gui as gui

    import pyflux.napari_storm_view as nsv

    only_gaussian = (nsv.RenderStyle("gaussian", "Gaussian", False),)
    monkeypatch.setattr(gui, "rendering_styles", lambda: only_gaussian)
    monkeypatch.setattr(nsv, "rendering_styles", lambda: only_gaussian)
    win = _main_window()
    try:
        assert not win.render_style.isEnabled()
        assert win._render_footprint() == "gaussian"
    finally:
        win.plot_view.shutdown()
        win.deleteLater()
