from PySide6.QtWebEngineCore import QWebEngineProfile
import tempfile
import os
from PySide6.QtCore import QUrl
import glob
import numpy as np
import pandas as pd
import sys
from PySide6 import QtWebEngineCore
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

import plotly.graph_objects as go
import plotly.io as pio


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
    Coloring: end-to-end distance per tid.
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


def make_plotly_html(arr, avg_tid: bool, is3d: bool):
    xyz, vals, tids_plot = scatter_points_and_color(arr, avg_tid)
    if xyz is None:
        return "<html><body style='font-family:sans-serif'>No data</body></html>"

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if is3d:
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=8, opacity=0.85, color=vals, colorscale="Turbo",
                        colorbar=dict(title="end-to-end")),
            customdata=np.stack([tids_plot, vals], axis=1),
            hovertemplate=("x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
                           "tid=%{customdata[0]}<br>end2end=%{customdata[1]:.3f}"
                           "<extra></extra>")
        )
        fig = go.Figure([trace])
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=20, b=0),
            uirevision="keep",
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
        )
    else:
        trace = go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=8,# if avg_tid else 3,
                opacity=0.9,# if avg_tid else 0.25,
                color=vals,
                colorscale="Turbo",
                colorbar=dict(title="end-to-end"),
            ),
            customdata=np.stack([tids_plot, vals], axis=1),
            hovertemplate=("x=%{x:.3f}<br>y=%{y:.3f}<br>"
                           "tid=%{customdata[0]}<br>end2end=%{customdata[1]:.3f}"
                           "<extra></extra>")
        )
        fig = go.Figure([trace])
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=20, b=0),
            dragmode="pan",
            uirevision="keep",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.layout.margin.autoexpand=False


    return pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",   # <-- IMPORTANT
        config={"scrollZoom": True, "displaylogo": False, "responsive": False},
    )


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

        save_path = os.path.join(p["save_folder"], f"{base}.csv")
        avg_save_path = os.path.join(p["save_folder"], f"{base}_stats.csv")
        save_to_csv(df, save_path)
        avg_df.to_csv(avg_save_path, index=True)

        items = [
            ("File", base),
            ("Total imaging time (min)", f"{total_tim/60:.2f}"),
            ("Total raw localizations", str(total_loc)),
            ("Last iteration localizations", str(last_iteration_loc)),
            ("After trace filtering", str(after_trace)),
            ("After EFO filtering", str(len(df))),
            ("Localization precision", str(loc_prec) if loc_prec is not None else "—"),
            ("EFO range", f"{xmin:.2f} ... {xmax:.2f}"),
            ("% remaining", f"{(len(df)/total_loc*100):.2f}%"),
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

        # widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ---- top area: controls + output ----
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        left = QtWidgets.QVBoxLayout()
        top.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        top.addLayout(right, 2)

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

        # params
        params_box = QtWidgets.QGroupBox("Parameters")
        left.addWidget(params_box)
        pg = QtWidgets.QGridLayout(params_box)

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

        pg.addWidget(QtWidgets.QLabel("Min trace length:"), 0, 0)
        pg.addWidget(self.min_trace, 0, 1)
        pg.addWidget(QtWidgets.QLabel("Z correction factor:"), 0, 2)
        pg.addWidget(self.zcorr, 0, 3)
        pg.addWidget(QtWidgets.QLabel("Scale factor:"), 1, 0)
        pg.addWidget(self.scale, 1, 1)
        pg.addWidget(QtWidgets.QLabel("EFO histogram bin size:"), 1, 2)
        pg.addWidget(self.bin_size, 1, 3)

        self._workers = set()

        # run row
        runrow = QtWidgets.QHBoxLayout()
        left.addLayout(runrow)

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self.run_start)
        runrow.addWidget(self.run_btn)

        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        runrow.addWidget(self.file_combo, 1)

        self.avg_tid = QtWidgets.QCheckBox("avg loc (tid)")
        self.avg_tid.stateChanged.connect(self.redraw_scatter)
        runrow.addWidget(self.avg_tid)

        self.is3d = QtWidgets.QCheckBox("3D")
        self.is3d.stateChanged.connect(self.redraw_scatter)
        runrow.addWidget(self.is3d)

        self.current_lbl = QtWidgets.QLabel("Current file: —")
        left.addWidget(self.current_lbl)

        # output table
        outbox = QtWidgets.QGroupBox("Output")
        right.addWidget(outbox)
        ov = QtWidgets.QVBoxLayout(outbox)

        self.out_table = QtWidgets.QTableWidget(0, 2)
        self.out_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.out_table.horizontalHeader().setStretchLastSection(True)
        self.out_table.verticalHeader().setVisible(False)
        self.out_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        ov.addWidget(self.out_table)

        # ---- bottom area: histogram + plotly ----
        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom, 1)

        # histogram panel
        hist_box = QtWidgets.QGroupBox("EFO selection")
        bottom.addWidget(hist_box, 1)
        hv = QtWidgets.QVBoxLayout(hist_box)

        self.fig = Figure(figsize=(5, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        hv.addWidget(self.canvas)
        hv.addWidget(NavigationToolbar2QT(self.canvas, self))

        ctrl = QtWidgets.QHBoxLayout()
        hv.addLayout(ctrl)
        self.range_lbl = QtWidgets.QLabel("Selected range: —")
        ctrl.addWidget(self.range_lbl, 1)

        self.apply_btn = QtWidgets.QPushButton("Apply EFO")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_preview)
        ctrl.addWidget(self.apply_btn)

        self.continue_btn = QtWidgets.QPushButton("Continue")
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

        self.web = QWebEngineView()

        # ---- debug hooks (ADD HERE) ----
        def _js_console(level, msg, line, source):
            print("JS:", msg, "line", line, "source:", source)

        # Note: PySide6 signature differs by version; this is a common working form:
        self.web.page().javaScriptConsoleMessage = _js_console

        self.web.page().loadFinished.connect(
            lambda ok: print("WEB loadFinished:", ok, "url:", self.web.url().toString())
        )

        self.web.page().renderProcessTerminated.connect(
            lambda status, code: print("WebEngine crashed:", status, code)
        )
        # -------------------------------

        s = self.web.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        pv.addWidget(self.web)
        # status
        self.statusBar().showMessage("Ready.")

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

        # stop/cancel any running workers
        try:
            if self._current_worker is not None and self._current_worker.isRunning():
                self._current_worker.cancel()
        except Exception:
            pass

        # give Qt/WebEngine a moment to shut down cleanly, then hard-exit
        event.ignore()
        QtCore.QTimer.singleShot(1500, lambda: os._exit(0))
        return
        
        

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

        self._current_index = 0 if self._all_files else -1
        if self._all_files:
            self.file_combo.setCurrentIndex(0)
            self.current_lbl.setText(f"Current file: {self.file_combo.currentText()}")
        else:
            self.current_lbl.setText("Current file: —")

    def on_file_changed(self, idx):
        if getattr(self, "_closing", False):
            return
        if idx < 0 or idx >= len(self._all_files):
            return

        if self._current_worker is not None and self._current_worker.isRunning():
            self._current_worker.cancel()
            return  # don't switch files while a worker is still running
        self._current_index = idx
        self.current_lbl.setText(f"Current file: {self.file_combo.currentText()}")

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
        w.finished.connect(lambda: self._workers.discard(w))
        w.finished.connect(w.deleteLater)

        w.need_efo.connect(self.on_need_efo)
        w.status.connect(self.statusBar().showMessage)
        w.done_one.connect(self.on_done_one)

        self._current_worker = w
        w.start()

    # ---------------- EFO selection ----------------
    def on_need_efo(self, ctx):
        self._current_ctx = ctx
        efo_vals = np.asarray(ctx["efo_vals"])
        base = ctx["base"]
        bin_size = float(self.bin_size.value())

        # histogram
        self.ax.clear()
        bins = np.arange(efo_vals.min(), efo_vals.max() + bin_size, bin_size)
        self.ax.hist(efo_vals, bins=bins, edgecolor="darkorange", color="darkorange")
        self.ax.set_xlabel("EFO")
        self.ax.set_ylabel("Count")
        self.ax.set_title(base)
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
            props=dict(facecolor="yellow", alpha=0.3, edgecolor="black", linewidth=2),
            interactive=True, drag_from_anywhere=True
        )
        self._span.extents = (self._span_xmin, self._span_xmax)
        onselect(self._span_xmin, self._span_xmax)

        self.apply_btn.setEnabled(True)
        self.continue_btn.setEnabled(False)

        # initial scatter
        self.redraw_scatter()

    def redraw_scatter(self):
        if self._current_ctx is None:
            return
        arr = self._current_ctx["arr"]
        html = make_plotly_html(arr, self.avg_tid.isChecked(), self.is3d.isChecked())
        self.show_plotly(html)

    def apply_preview(self):
        if self._current_ctx is None:
            return
        arr = self._current_ctx["arr"]
        xmin, xmax = self._span_xmin, self._span_xmax
        mask = (arr["efo"] >= xmin) & (arr["efo"] <= xmax)
        arr_efo = arr[mask]

        # update scatter
        html = make_plotly_html(arr_efo, self.avg_tid.isChecked(), self.is3d.isChecked())
        self.show_plotly(html)

        # preview localization precision
        lp = preview_localization_precision(arr_efo)
        lp = tuple(float(f"{v:.2f}") for v in lp) if lp is not None else None

        items = [
            ("File", self._current_ctx["base"]),
            ("Total imaging time (min)", f"{self._current_ctx['total_tim']/60:.2f}"),
            ("Total raw localizations", str(self._current_ctx["total_loc"])),
            ("Last iteration localizations", str(self._current_ctx["last_iteration_loc"])),
            ("After trace filtering", str(self._current_ctx["after_trace"])),
            ("After EFO filtering", str(int(np.count_nonzero(mask)))),
            ("Localization precision", str(lp) if lp is not None else "—"),
            ("EFO range (preview)", f"{xmin:.2f} ... {xmax:.2f}"),
            ("% remaining", f"{(int(np.count_nonzero(mask))/self._current_ctx['total_loc']*100):.2f}%"),
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
        if self._current_index >= len(self._all_files):
            self.statusBar().showMessage("Done.")
            self.run_btn.setEnabled(True)
            self.apply_btn.setEnabled(False)
            self.continue_btn.setEnabled(False)
            return

        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(self._current_index)
        self.file_combo.blockSignals(False)
        self.current_lbl.setText(f"Current file: {self.file_combo.currentText()}")
        self.start_worker_for_current()


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Initialize WebEngine AFTER QApplication exists
    from PySide6.QtWebEngineCore import QWebEngineProfile
    QWebEngineProfile.defaultProfile()

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()