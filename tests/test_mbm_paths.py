# tests/test_mbm_paths.py
#
# Finding a measurement's MBM bead data. Imspector writes it as a Zarr group
# grd/mbm inside the measurement folder; the exported .npy may sit inside that
# folder or next to it. pyflux used to look only at <folder of the .npy>/grd/mbm
# and failed with a bare "path does not exist" otherwise.

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pyflux.core import (  # noqa: E402
    MbmDataError,
    compute_mbm_transform,
    find_mbm_dir,
    is_mbm_group,
    load_mbm_points,
    resolve_mbm_dir,
)

POINTS_DTYPE = [("gri", "<u4"), ("xyz", "<f8", (3,))]


def _beads(n_beads=5, per_bead=20, seed=0, shift=(0.0, 0.0, 0.0)):
    """Bead tracks: `per_bead` positions of each of `n_beads` beads, in metres."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-5e-6, 5e-6, (n_beads, 3))
    centres[:, 2] = rng.uniform(-20e-9, 20e-9, n_beads)  # beads lie on the coverslip
    pts = np.zeros(n_beads * per_bead, dtype=POINTS_DTYPE)
    pts["gri"] = np.repeat(np.arange(n_beads), per_bead)
    pts["xyz"] = np.repeat(centres, per_bead, axis=0) + rng.normal(0, 1e-10, (len(pts), 3))
    pts["xyz"] += np.asarray(shift)
    return pts


def _write_mbm(measurement_dir, points=None):
    """A grd/mbm group with a `points` array inside *measurement_dir*."""
    group = os.path.join(measurement_dir, "grd", "mbm")
    os.makedirs(os.path.dirname(group), exist_ok=True)
    root = zarr.open_group(group, mode="w")
    root.create_dataset("points", data=_beads() if points is None else points)
    return group


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.zeros(1))
    return path


# ----------------------------
# where the MBM data is found
# ----------------------------

def test_found_in_the_measurement_folder_next_to_the_npy(tmp_path):
    # the layout that used to fail: base/<measurement>.npy + base/<measurement>/grd/mbm
    npy = _touch(str(tmp_path / "meas_A.npy"))
    group = _write_mbm(str(tmp_path / "meas_A"))
    _write_mbm(str(tmp_path / "meas_B"))  # another measurement's beads
    assert find_mbm_dir(npy, str(tmp_path)) == group


def test_found_next_to_an_npy_inside_its_measurement_folder(tmp_path):
    npy = _touch(str(tmp_path / "meas_A" / "export.npy"))
    group = _write_mbm(str(tmp_path / "meas_A"))
    assert find_mbm_dir(npy, str(tmp_path)) == group


def test_found_in_a_zarr_store_named_after_the_file(tmp_path):
    npy = _touch(str(tmp_path / "meas_A.npy"))
    group = _write_mbm(str(tmp_path / "meas_A.zarr"))
    assert find_mbm_dir(npy) == group


def test_found_in_the_only_measurement_folder_below(tmp_path):
    npy = _touch(str(tmp_path / "export_2024.npy"))
    group = _write_mbm(str(tmp_path / "raw" / "measurement 1"))
    assert find_mbm_dir(npy) == group


def test_the_folder_named_after_the_file_wins_among_several(tmp_path):
    npy = _touch(str(tmp_path / "cell3.npy"))
    _write_mbm(str(tmp_path / "cell1_raw"))
    group = _write_mbm(str(tmp_path / "cell3_raw"))
    assert find_mbm_dir(npy) == group


def test_another_files_measurement_folder_is_not_taken(tmp_path):
    # meas_B has no MBM data; the one group below belongs to meas_A
    _touch(str(tmp_path / "meas_A.npy"))
    npy_b = _touch(str(tmp_path / "meas_B.npy"))
    _write_mbm(str(tmp_path / "meas_A"))
    assert find_mbm_dir(npy_b, str(tmp_path)) is None


def test_a_longer_name_is_not_the_file(tmp_path):
    npy = _touch(str(tmp_path / "cell3.npy"))
    _write_mbm(str(tmp_path / "cell30"))
    _write_mbm(str(tmp_path / "cell31"))
    assert find_mbm_dir(npy) is None


def test_an_ambiguous_choice_is_not_guessed(tmp_path):
    npy = _touch(str(tmp_path / "export.npy"))
    _write_mbm(str(tmp_path / "m1"))
    _write_mbm(str(tmp_path / "m2"))
    assert find_mbm_dir(npy) is None


def test_found_in_a_folder_above_the_npy_up_to_the_data_folder(tmp_path):
    npy = _touch(str(tmp_path / "meas_A" / "exports" / "x.npy"))
    group = _write_mbm(str(tmp_path / "meas_A"))
    assert find_mbm_dir(npy, str(tmp_path)) == group
    # never above the folder the user picked
    _write_mbm(str(tmp_path))
    assert find_mbm_dir(npy, str(tmp_path / "meas_A")) == group


def test_nothing_found_is_none(tmp_path):
    assert find_mbm_dir(_touch(str(tmp_path / "x.npy")), str(tmp_path)) is None


def test_a_folder_without_points_is_not_mbm_data(tmp_path):
    os.makedirs(tmp_path / "meas_A" / "grd" / "mbm")
    npy = _touch(str(tmp_path / "meas_A.npy"))
    assert not is_mbm_group(str(tmp_path / "meas_A" / "grd" / "mbm"))
    assert find_mbm_dir(npy) is None


# ----------------------------
# what a user may pick
# ----------------------------

def test_the_measurement_grd_or_mbm_folder_all_resolve(tmp_path):
    group = _write_mbm(str(tmp_path / "meas_A"))
    for picked in (tmp_path / "meas_A", tmp_path / "meas_A" / "grd", Path(group)):
        assert resolve_mbm_dir(str(picked)) == group
    assert resolve_mbm_dir(str(tmp_path)) is None
    assert resolve_mbm_dir("") is None
    assert resolve_mbm_dir(None) is None


def test_points_load_from_the_measurement_folder(tmp_path):
    pts = _beads()
    _write_mbm(str(tmp_path / "meas_A"), pts)
    loaded = load_mbm_points(str(tmp_path / "meas_A"))
    np.testing.assert_array_equal(loaded["gri"], pts["gri"])
    np.testing.assert_allclose(loaded["xyz"][:, 1], -pts["xyz"][:, 1])  # y flipped as for the data


def test_a_wrong_folder_says_what_is_missing(tmp_path):
    with pytest.raises(MbmDataError, match="no MBM bead data in .*grd/mbm"):
        load_mbm_points(str(tmp_path))


def test_the_transform_is_found_from_measurement_folders(tmp_path):
    shift = (40e-9, -25e-9, 0.0)
    _write_mbm(str(tmp_path / "ref"), _beads(seed=3))
    _write_mbm(str(tmp_path / "mov"), _beads(seed=3, shift=shift))
    T, common = compute_mbm_transform(str(tmp_path / "ref"), str(tmp_path / "mov"), scale=1e9)
    assert len(common) >= 3
    # moving onto the reference undoes the shift (y negated by the loader), in nm
    np.testing.assert_allclose(T[:3, 3], [-40.0, -25.0, 0.0], atol=0.5)


# ----------------------------
# the main window
# ----------------------------

def _window(qapp, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    from pyflux.gui import MainWindow

    shown = []

    def record(box):
        shown.append((box.text(), box.informativeText()))
        return int(QtWidgets.QMessageBox.Cancel)

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", record)
    win = MainWindow()
    win.data_edit.setText(str(tmp_path))
    return win, shown


def _close(qapp, win):
    win.close()
    for _ in range(5):
        qapp.processEvents()


def test_the_default_paths_find_the_measurement_folders(tmp_path, qapp, monkeypatch):
    pytest.importorskip("pytestqt")
    a = _touch(str(tmp_path / "meas_A.npy"))
    b = _touch(str(tmp_path / "meas_B.npy"))
    ga = _write_mbm(str(tmp_path / "meas_A"))
    win, _shown = _window(qapp, tmp_path, monkeypatch)
    try:
        win._all_files = [a, b]
        win._ensure_default_mbm_paths()
        assert win._mbm_dir_by_base[win.base_of(a)] == ga
        assert win._mbm_dir_by_base[win.base_of(b)] == ""
        assert win._mbm_dir_for(win.base_of(a)) == ga
        assert win._mbm_dir_for(win.base_of(b)) is None
    finally:
        _close(qapp, win)


def test_align_without_mbm_data_says_to_update_parameters(tmp_path, qapp, monkeypatch):
    pytest.importorskip("pytestqt")
    a = _touch(str(tmp_path / "meas_A.npy"))
    b = _touch(str(tmp_path / "meas_B.npy"))
    _write_mbm(str(tmp_path / "meas_A"))
    win, shown = _window(qapp, tmp_path, monkeypatch)
    try:
        win._all_files = [a, b]
        base_a, base_b = win.base_of(a), win.base_of(b)
        win._filtered_by_base = {base_a: np.zeros(0), base_b: np.zeros(0)}
        win._mbm_source_base = base_a
        applied = []
        monkeypatch.setattr(win, "_apply_mbm_alignment", lambda: applied.append(True))

        win.on_mbm_align_clicked()

        assert not applied
        text, details = shown[-1]
        assert "could not be found" in text and "Parameters" in text
        assert base_b in details and "no MBM folder found" in details
        assert base_a not in details  # the source has its data

        # a measurement folder picked in Parameters is enough
        _write_mbm(str(tmp_path / "elsewhere" / "meas_B"))
        win._mbm_dir_by_base[base_b] = str(tmp_path / "elsewhere" / "meas_B")
        win.on_mbm_align_clicked()
        assert applied
    finally:
        _close(qapp, win)


def test_the_parameters_dialog_fills_in_the_found_folders(tmp_path, qapp):
    pytest.importorskip("pytestqt")
    from pyflux.gui import ParametersDialog

    a = _touch(str(tmp_path / "meas_A.npy"))
    b = _touch(str(tmp_path / "meas_B.npy"))
    ga = _write_mbm(str(tmp_path / "meas_A"))
    dlg = ParametersDialog()
    dlg.set_mbm_rows([a, b], {}, ["meas_A", "meas_B"], data_root=str(tmp_path))
    assert dlg.mbm_map() == {"meas_A": ga, "meas_B": ""}
    assert dlg.mbm_table.cellWidget(1, 1).placeholderText().startswith("not found")
