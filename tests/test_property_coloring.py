# tests/test_property_coloring.py
#
# Colouring localizations by any numeric column of the file ("property" mode):
# the helpers that read the column, the Plotly and the napari-storm renderings
# of it, and the colour settings table that chooses it.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pyflux.plotting import (  # noqa: E402
    DEFAULT_PROPERTY,
    DEFAULT_PROPERTY_CHOICES,
    MODE_CHOICES,
    MODE_PROPERTY,
    make_plotly_fig,
    make_plotly_fig_merged,
    parse_limits,
    property_column,
    property_columns,
    property_limits,
    property_values,
    shared_property_ranges,
)


def _minflux(n=60, ntid=6, seed=0):
    rng = np.random.default_rng(seed)
    dt = [
        ("tim", "f8"), ("vld", "?"), ("loc", "f8", (3,)), ("itr", "i4"),
        ("tid", "i4"), ("efo", "f8"), ("cfr", "f8"), ("dcr", "f8", (2,)),
    ]
    a = np.zeros(n, dtype=dt)
    a["tid"] = np.sort(rng.integers(0, ntid, n))
    a["loc"] = rng.uniform(0, 1000, (n, 3))
    a["loc"][:, 1] *= -1
    a["vld"] = True
    a["tim"] = np.arange(n, dtype=float)
    a["efo"] = rng.uniform(1e4, 1e5, n)
    a["cfr"] = rng.uniform(0, 1, n)
    a["dcr"][:, 0] = rng.uniform(0, 1, n)
    a["dcr"][:, 1] = 1 - a["dcr"][:, 0]
    return a


def _cs(**kw):
    base = {"mode": MODE_PROPERTY, "property": "efo", "lut": "turbo", "alpha": 0.8, "size": 5}
    base.update(kw)
    return base


# ----------------------------
# reading the column
# ----------------------------

def test_property_mode_is_offered():
    assert MODE_PROPERTY in MODE_CHOICES
    assert DEFAULT_PROPERTY in DEFAULT_PROPERTY_CHOICES


def test_the_numeric_columns_are_listed_with_vectors_split():
    assert property_columns(_minflux()) == ["tim", "itr", "tid", "efo", "cfr", "dcr_x", "dcr_y"]
    assert property_columns(None) == []
    assert property_columns(np.zeros(3)) == []


def test_a_column_or_a_vector_component_is_read():
    a = _minflux()
    np.testing.assert_array_equal(property_column(a, "efo"), a["efo"])
    np.testing.assert_array_equal(property_column(a, "dcr_y"), a["dcr"][:, 1])
    assert property_column(a, "nope") is None
    assert property_column(a, "dcr_7") is None


def test_per_trace_values_are_the_mean_ignoring_nan():
    a = _minflux(n=6, ntid=3)
    a["tid"] = [2, 2, 1, 1, 1, 3]
    a["efo"] = [1.0, 3.0, 10.0, np.nan, 20.0, 7.0]
    np.testing.assert_array_equal(property_values(a, "efo", False), a["efo"])
    # sorted by tid, as scatter_points_and_color orders the trace centroids
    np.testing.assert_allclose(property_values(a, "efo", True), [15.0, 2.0, 7.0])


def test_a_trace_without_a_single_value_is_nan():
    a = _minflux(n=4, ntid=2)
    a["tid"] = [0, 0, 1, 1]
    a["efo"] = [np.nan, np.nan, 1.0, 3.0]
    values = property_values(a, "efo", True)
    assert np.isnan(values[0]) and values[1] == 2.0


def test_automatic_limits_are_the_inner_percentiles():
    values = np.arange(1001, dtype=float)
    assert property_limits(values) == (10.0, 990.0)
    assert property_limits(values, (None, 500.0)) == (10.0, 500.0)
    assert property_limits(values, (0.0, 1.0)) == (0.0, 1.0)
    assert property_limits([np.nan, np.nan]) is None
    assert property_limits([], (1.0, 2.0)) == (1.0, 2.0)


@pytest.mark.parametrize("text, expected", [
    ("", None), ("auto", None), (" , ", None), ("0.2, 0.8", (0.2, 0.8)),
    (", 0.8", (None, 0.8)), ("0.2,", (0.2, None)), ("1; 2", (1.0, 2.0)),
    ("1, 0", None), ("a, b", None), ("1", None), (None, None),
])
def test_limits_are_parsed_from_the_table_entry(text, expected):
    assert parse_limits(text) == expected


def test_datasets_colouring_by_the_same_property_share_a_range():
    a, b = _minflux(seed=1), _minflux(seed=2)
    b["efo"] += 1e5
    settings = {"a": _cs(), "b": _cs(), "c": _cs(property="cfr"), "d": {"mode": "depth"}}
    ranges = shared_property_ranges({"a": a, "b": b, "c": a, "d": b}, False, settings)
    assert set(ranges) == {"efo", "cfr"}
    low, high = ranges["efo"]
    assert low == property_limits(np.r_[a["efo"], b["efo"]])[0]
    assert high == property_limits(np.r_[a["efo"], b["efo"]])[1]
    assert ranges["cfr"] == property_limits(a["cfr"])


def test_typed_limits_fix_the_shared_range():
    a, b = _minflux(seed=1), _minflux(seed=2)
    settings = {"a": _cs(limits=(None, 5e4)), "b": _cs(limits=(1e4, None))}
    (low, high), = shared_property_ranges({"a": a, "b": b}, False, settings).values()
    assert (low, high) == (1e4, 5e4)


# ----------------------------
# Plotly
# ----------------------------

def test_plotly_colours_the_property_against_its_limits():
    a = _minflux()
    fig = make_plotly_fig(a, False, False, _cs(property="cfr", limits=(0.2, 0.8)))
    marker = fig.data[0].marker
    assert (marker.cmin, marker.cmax) == (0.2, 0.8)
    np.testing.assert_allclose(np.asarray(marker.color), a["cfr"])
    assert marker.colorbar.title.text == "cfr"


def test_plotly_puts_a_missing_value_at_the_bottom_of_the_scale():
    a = _minflux()
    a["efo"][3] = np.nan
    fig = make_plotly_fig(a, False, False, _cs(limits=(1e4, 1e5)))
    colors = np.asarray(fig.data[0].marker.color)
    assert len(colors) == len(a)  # the localization is still drawn
    assert colors[3] == 1e4


def test_plotly_merged_view_shares_the_property_range():
    a, b = _minflux(seed=1), _minflux(seed=2)
    b["efo"] += 1e5
    fig = make_plotly_fig_merged({"a": a, "b": b}, False, False, {"a": _cs(), "b": _cs()})
    assert [tr.marker.cmin for tr in fig.data] == [fig.data[0].marker.cmin] * 2
    assert [tr.marker.cmax for tr in fig.data] == [fig.data[0].marker.cmax] * 2
    assert sum(bool(tr.marker.showscale) for tr in fig.data) == 1


# ----------------------------
# napari-storm
# ----------------------------

def test_napari_storm_maps_the_property_onto_the_colormap():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    a = _minflux()
    spec = build_render_spec(a, False, True, _cs(property="cfr", limits=(0.0, 1.0)), name="a")
    np.testing.assert_allclose(spec.request.values, a["cfr"], atol=1e-6)
    assert spec.color_coded
    spec = build_render_spec(a, False, True, _cs(property="cfr", limits=(0.25, 0.75)), name="a")
    expected = np.clip((a["cfr"] - 0.25) / 0.5, 0, 1)
    np.testing.assert_allclose(spec.request.values, expected, atol=1e-6)


def test_napari_storm_automatic_limits_are_the_inner_percentiles():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    a = _minflux(n=1001)
    a["efo"] = np.arange(1001, dtype=float)
    spec = build_render_spec(a, False, True, _cs(), name="a")
    assert spec.request.values[10] == pytest.approx(0.0)
    assert spec.request.values[990] == pytest.approx(1.0)
    assert spec.request.values[500] == pytest.approx(0.5, abs=1e-3)


def test_napari_storm_keeps_the_values_aligned_when_rows_are_dropped():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    a = _minflux()
    a["loc"][5, 0] = np.nan  # that row is not drawn
    spec = build_render_spec(a, False, True, _cs(property="cfr", limits=(0.0, 1.0)), name="a")
    assert len(spec.request.values) == len(a) - 1
    np.testing.assert_allclose(spec.request.values, np.delete(a["cfr"], 5), atol=1e-6)


def test_napari_storm_colours_a_missing_value_at_the_bottom():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    a = _minflux()
    a["efo"][3] = np.nan
    spec = build_render_spec(a, False, True, _cs(limits=(1e4, 1e5)), name="a")
    assert len(spec.request.values) == len(a)
    assert spec.request.values[3] == 0.0


def test_napari_storm_per_trace_property_matches_the_centroids():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    a = _minflux(n=6, ntid=3)
    a["tid"] = [2, 2, 1, 1, 1, 3]
    a["efo"] = [1.0, 3.0, 10.0, np.nan, 20.0, 7.0]
    spec = build_render_spec(a, True, True, _cs(limits=(0.0, 20.0)), name="a")
    np.testing.assert_allclose(spec.request.values, [0.75, 0.1, 0.35], atol=1e-6)


def test_napari_storm_merged_view_shares_the_property_range():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_specs

    a, b = _minflux(seed=1), _minflux(seed=2)
    b["efo"] += 1e5
    settings = {"a": _cs(limits=(0.0, 2e5)), "b": _cs()}
    specs = build_render_specs({"a": a, "b": b}, False, True, settings, shared_depth_range=True)
    np.testing.assert_allclose(specs["a"].request.values, a["efo"] / 2e5, atol=1e-6)
    np.testing.assert_allclose(specs["b"].request.values, b["efo"] / 2e5, atol=1e-6)
    # side by side, each dataset is scaled on its own
    specs = build_render_specs({"a": a, "b": b}, False, True, settings)
    assert specs["b"].request.values.min() == pytest.approx(0.0)


def test_an_unknown_property_draws_everything_in_one_colour():
    pytest.importorskip("napari_storm")
    from pyflux.napari_storm_view import build_render_spec

    spec = build_render_spec(_minflux(), False, True, _cs(property="nope"), name="a")
    assert np.all(spec.request.values == 0.0)


# ----------------------------
# the colour settings table
# ----------------------------

def _panel(bases=("a",), settings=None):
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyflux.gui import ColorSettingsPanel

    panel = ColorSettingsPanel()
    panel.rebuild(list(bases), settings or {}, None)
    return panel


def test_the_table_has_a_property_and_a_limits_column():
    panel = _panel()
    w = panel._widgets_by_base["a"]
    headers = [panel.table.horizontalHeaderItem(i).text() for i in range(panel.table.columnCount())]
    assert headers[panel.COL_PROPERTY] == "Property"
    assert headers[panel.COL_LIMITS] == "Limits"
    offered = [w["property"].itemText(i) for i in range(w["property"].count())]
    assert offered == DEFAULT_PROPERTY_CHOICES
    # only the property mode reads them
    assert not w["property"].isEnabled() and not w["limits"].isEnabled()
    w["mode"].setCurrentText(MODE_PROPERTY)
    assert w["property"].isEnabled() and w["limits"].isEnabled()
    assert w["palette"].isEnabled()  # a LUT, as for depth


def test_the_table_emits_the_property_and_its_limits():
    panel = _panel()
    seen = []
    panel.changed.connect(lambda base, payload: seen.append((base, payload)))
    w = panel._widgets_by_base["a"]
    w["mode"].setCurrentText(MODE_PROPERTY)
    w["property"].setCurrentText("cfr")
    w["limits"].setText("0.2, 0.8")
    w["limits"].editingFinished.emit()
    base, payload = seen[-1]
    assert base == "a"
    assert payload["mode"] == MODE_PROPERTY
    assert payload["property"] == "cfr"
    assert payload["limits"] == (0.2, 0.8)
    assert payload["lut"] == w["palette"].currentText()


def test_the_table_restores_a_saved_property_and_limits():
    panel = _panel(settings={"a": _cs(property="cfr", limits=(None, 0.8))})
    w = panel._widgets_by_base["a"]
    assert w["mode"].currentText() == MODE_PROPERTY
    assert w["property"].currentText() == "cfr"
    assert parse_limits(w["limits"].text()) == (None, 0.8)


def test_a_loaded_file_offers_its_own_columns_and_keeps_the_choice():
    panel = _panel(settings={"a": _cs(property="cfr")})
    w = panel._widgets_by_base["a"]
    panel.set_properties("a", ["tim", "cfr", "eco"])
    assert [w["property"].itemText(i) for i in range(w["property"].count())] == ["tim", "cfr", "eco"]
    assert w["property"].currentText() == "cfr"
    seen = []
    panel.changed.connect(lambda base, payload: seen.append(payload))
    panel.set_properties("a", ["tim", "eco"])  # the chosen column is gone
    assert w["property"].currentText() == "tim"
    assert seen and seen[-1]["property"] == "tim"
    # a row built after that lists the file's columns straight away
    panel.rebuild(["a"], {}, None)
    w = panel._widgets_by_base["a"]
    assert [w["property"].itemText(i) for i in range(w["property"].count())] == ["tim", "eco"]
