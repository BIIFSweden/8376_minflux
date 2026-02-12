import os
import glob
import queue
import threading
import time

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector


# -------------------- helpers --------------------
def save_to_csv(df, save_path):
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


def add_wheel_zoom(fig, ax, base_scale=1.2):
    def zoom_fun(event):
        if event.inaxes != ax:
            return
        scale_factor = 1 / base_scale if event.button == "up" else base_scale

        xdata, ydata = event.xdata, event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", zoom_fun)


def add_drag_pan_fast(fig, ax, button=1, redraw_interval=0.015):
    state = {"pressed": False, "xlim": None, "ylim": None, "press_xy_pix": None, "last_draw": 0.0}

    def on_press(event):
        if event.inaxes != ax or event.button != button:
            return
        state["pressed"] = True
        state["xlim"] = ax.get_xlim()
        state["ylim"] = ax.get_ylim()
        state["press_xy_pix"] = (event.x, event.y)

    def on_motion(event):
        if not state["pressed"]:
            return

        x0_pix, y0_pix = state["press_xy_pix"]
        dx_pix = event.x - x0_pix
        dy_pix = event.y - y0_pix

        bbox = ax.bbox
        x0, x1 = state["xlim"]
        y0, y1 = state["ylim"]

        dx_data = dx_pix * (x1 - x0) / bbox.width
        dy_data = dy_pix * (y1 - y0) / bbox.height

        ax.set_xlim(x0 - dx_data, x1 - dx_data)
        ax.set_ylim(y0 - dy_data, y1 - dy_data)

        t = time.time()
        if t - state["last_draw"] > redraw_interval:
            fig.canvas.draw_idle()
            state["last_draw"] = t

    def on_release(event):
        if event.button == button:
            state["pressed"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


# -------------------- GUI --------------------
class MinfluxFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self._scatter_xlim = None
        self._scatter_ylim = None
        self._scatter_capture_enabled = True
        self.title("MINFLUX Filter GUI (embedded plot + stats)")
        self.geometry("1150x750")

        self.data_path = tk.StringVar(value="")
        self.save_folder = tk.StringVar(value="")

        self.min_trace_length = tk.IntVar(value=3)
        self.z_correction_factor = tk.DoubleVar(value=0.7)
        self.scale_factor = tk.DoubleVar(value=1e9)
        self.bin_size = tk.DoubleVar(value=3e3)
        self._efo_cancelled = False
        # file navigation
        self._all_npy_files = []     # full paths, sorted
        self._current_index = 0      # index into _all_npy_files
        self._is_running = False
        self._jump_to_index = None   # jump after current file finishes
        self._scatter_file_id = None   # e.g. base name or full path
        self.file_choice = tk.StringVar(value="")
        self.current_file_var = tk.StringVar(value="Current file: —")

        # worker <-> main thread communication
        self._req_q = queue.Queue()
        self._worker = None
        self._worker_busy = False
        # EFO selection handshake (main-thread plot)
        self._waiting_for_efo = False
        self._efo_event = None
        self._efo_result_holder = None

        # preview context
        self._efo_ctx = None
        self._applied_once = False
        self.avg_tid_var = tk.BooleanVar(value=False)
        # stats history
        self._stats_history = {}

        self._build_ui()
        self.after(100, self._poll_requests)

    # ---------------- UI ----------------
    def _build_ui(self):
        self._build_top_split_panel()
        self._build_efo_panel()

        status_bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        status_bar.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_bar, textvariable=self.status_var, anchor="w").pack(fill="x")

    def _build_top_split_panel(self):
        top_area = ttk.Frame(self, padding=(10, 10, 10, 0))
        top_area.pack(fill="x")

        top_area.columnconfigure(0, weight=3)
        top_area.columnconfigure(1, weight=2)

        left = ttk.Frame(top_area)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(1, weight=1)

        ttk.Label(left, text="MINFLUX data folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.data_path, width=85).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(left, text="Browse...", command=self.browse_folder).grid(row=0, column=2)

        ttk.Label(left, text="Output folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.save_folder, width=85).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(left, text="Set default", command=self.set_default_output).grid(row=1, column=2)

        params = ttk.LabelFrame(left, text="Parameters", padding=10)
        params.grid(row=2, column=0, columnspan=3, sticky="we", pady=(10, 0))

        ttk.Label(params, text="Min trace length:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(params, from_=1, to=999, textvariable=self.min_trace_length, width=10).grid(
            row=0, column=1, sticky="w", padx=6
        )

        ttk.Label(params, text="Z correction factor:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(params, textvariable=self.z_correction_factor, width=12).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(params, text="Scale factor:").grid(row=1, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.scale_factor, width=12).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(params, text="EFO histogram bin size:").grid(row=1, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(params, textvariable=self.bin_size, width=12).grid(row=1, column=3, sticky="w", padx=6)

        btns = ttk.Frame(left, padding=(0, 10, 0, 10))
        btns.grid(row=3, column=0, columnspan=3, sticky="w")

        self.run_btn = ttk.Button(btns, text="Run", command=self.run)
        self.run_btn.pack(side="left")

        self.file_combo = ttk.Combobox(
            btns, textvariable=self.file_choice, state="readonly", width=55, values=[]
        )
        self.file_combo.pack(side="left", padx=10)
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)

        self.avg_tid_chk = ttk.Checkbutton(
            btns,
            text="avg loc (tid)",
            variable=self.avg_tid_var,
            command=self._on_avg_tid_toggled
        )
        self.avg_tid_chk.pack(side="left", padx=(0, 10))

        ttk.Label(left, textvariable=self.current_file_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        # RIGHT: Output panel
        out_frame = ttk.LabelFrame(top_area, text="Output", padding=10)
        out_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(1, weight=1)

        toprow = ttk.Frame(out_frame)
        toprow.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        toprow.columnconfigure(1, weight=1)

        ttk.Label(toprow, text="Show:").grid(row=0, column=0, sticky="w")
        self.stats_choice = tk.StringVar(value="")
        self.stats_combo = ttk.Combobox(toprow, textvariable=self.stats_choice, state="readonly", values=[])
        self.stats_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.stats_combo.bind("<<ComboboxSelected>>", self._on_stats_selected)

        ttk.Button(toprow, text="Clear", command=self.clear_stats_history).grid(row=0, column=2, sticky="e")

        table_frame = ttk.Frame(out_frame)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.stats_tree = ttk.Treeview(
            table_frame, columns=("key", "value"), show="headings", selectmode="browse"
        )
        self.stats_tree.heading("key", text="Key")
        self.stats_tree.heading("value", text="Value")
        self.stats_tree.column("key", width=200, anchor="w")
        self.stats_tree.column("value", width=360, anchor="w")
        self.stats_tree.grid(row=0, column=0, sticky="nsew")

        ysb = ttk.Scrollbar(table_frame, orient="vertical", command=self.stats_tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        self.stats_tree.configure(yscrollcommand=ysb.set)

        xsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.stats_tree.xview)
        xsb.grid(row=1, column=0, sticky="ew")
        self.stats_tree.configure(xscrollcommand=xsb.set)

    def _on_avg_tid_toggled(self):
        # If nothing loaded yet, nothing to do
        if self._efo_ctx is None:
            return

        # Keep current zoom/view
        cur_xlim = self.ax_scatter.get_xlim()
        cur_ylim = self.ax_scatter.get_ylim()

        # Re-render from the currently active array:
        # If user has applied an EFO range, use the current span-filtered array.
        arr = self._efo_ctx["arr"]
        if self._span_xmin is not None and self._span_xmax is not None:
            arr = arr[(arr["efo"] >= self._span_xmin) & (arr["efo"] <= self._span_xmax)]

        self._render_scatter(arr)

        # restore view (your “keep zoom on redraw” behavior)
        self.ax_scatter.set_xlim(cur_xlim)
        self.ax_scatter.set_ylim(cur_ylim)
        self.canvas_scatter.draw_idle()
        
    def _build_efo_panel(self):
        content = ttk.Frame(self, padding=(10, 10, 10, 10))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Left: histogram
        hist_panel = ttk.LabelFrame(content, text="EFO selection", padding=10)
        hist_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        hist_panel.rowconfigure(0, weight=1)
        hist_panel.columnconfigure(0, weight=1)

        self.fig_hist = Figure(figsize=(5, 4.5), dpi=100)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_hist.set_title("Run processing to show EFO histogram here.")
        self.ax_hist.set_xlabel("EFO")
        self.ax_hist.set_ylabel("Count")

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=hist_panel)
        self.canvas_hist.draw()
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        NavigationToolbar2Tk(self.canvas_hist, hist_panel, pack_toolbar=False).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        ctrl = ttk.Frame(hist_panel)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="Selected range:").grid(row=0, column=0, sticky="w")
        self.efo_range_var = tk.StringVar(value="—")
        ttk.Label(ctrl, textvariable=self.efo_range_var).grid(row=0, column=1, sticky="w", padx=8)

        self.apply_efo_btn = ttk.Button(ctrl, text="Apply EFO", command=self._apply_efo_preview, state="disabled")
        self.apply_efo_btn.grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.continue_btn = ttk.Button(ctrl, text="Continue", command=self._continue_to_next_file, state="disabled")
        self.continue_btn.grid(row=0, column=3, sticky="e")

        self._span = None
        self._span_xmin = None
        self._span_xmax = None

        # Right: scatter
        scatter_panel = ttk.LabelFrame(content, text="Scatter plot", padding=10)
        scatter_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        scatter_panel.rowconfigure(0, weight=1)
        scatter_panel.columnconfigure(0, weight=1)

        self.fig_scatter = Figure(figsize=(5, 4.5), dpi=100)
        self.ax_scatter = self.fig_scatter.add_subplot(111)
        self.ax_scatter.set_xlabel("X")
        self.ax_scatter.set_ylabel("Y")

        self.canvas_scatter = FigureCanvasTkAgg(self.fig_scatter, master=scatter_panel)
        self.canvas_scatter.draw()
        self.canvas_scatter.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        NavigationToolbar2Tk(self.canvas_scatter, scatter_panel, pack_toolbar=False).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        add_wheel_zoom(self.fig_scatter, self.ax_scatter)
        add_drag_pan_fast(self.fig_scatter, self.ax_scatter, button=1)


    # ---------------- status / folders ----------------
    def set_status(self, msg: str):
        self.status_var.set(msg)

    def _status_from_worker(self, msg: str):
        self._req_q.put({"type": "STATUS", "msg": msg})

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select MINFLUX data folder")
        if path:
            self.data_path.set(path)
            self.set_default_output()
            self._refresh_file_list()

    def set_default_output(self):
        p = self.data_path.get().strip()
        if p:
            self.save_folder.set(p + "_filtered")

    def _refresh_file_list(self):
        data_path = self.data_path.get().strip()
        if not data_path or not os.path.isdir(data_path):
            self._all_npy_files = []
        else:
            files = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
            files.sort()
            self._all_npy_files = files

        display = [os.path.relpath(f, data_path) for f in self._all_npy_files]
        self.file_combo["values"] = display

        if display:
            self._current_index = min(self._current_index, len(display) - 1)
            self.file_choice.set(display[self._current_index])
            self.current_file_var.set(f"Current file: {display[self._current_index]}")
        else:
            self.file_choice.set("")
            self.current_file_var.set("Current file: —")

    # ---------------- stats history ----------------
    def clear_stats_history(self):
        self._stats_history = {}
        self.stats_combo["values"] = []
        self.stats_choice.set("")
        self._render_stats_table([])

    def _on_stats_selected(self, event=None):
        name = self.stats_choice.get()
        self._render_stats_table(self._stats_history.get(name, []))

    def _render_stats_table(self, items):
        for row in self.stats_tree.get_children():
            self.stats_tree.delete(row)
        for k, v in items:
            self.stats_tree.insert("", "end", values=(k, v))

    def push_stats(self, display_name, items, auto_select=True):
        self._stats_history[display_name] = items
        names = list(self._stats_history.keys())
        self.stats_combo["values"] = names
        if auto_select:
            self.stats_choice.set(display_name)
            self._render_stats_table(items)

    # ---------------- file selection behavior ----------------
    def _on_file_selected(self, event=None):
        data_path = self.data_path.get().strip()
        disp = self.file_choice.get()
        if not disp or not self._all_npy_files:
            return

        chosen_full = os.path.normpath(os.path.join(data_path, disp))
        if chosen_full not in self._all_npy_files:
            return

        idx = self._all_npy_files.index(chosen_full)

        # If currently waiting for EFO, cancel immediately (no Continue needed)
        if self._waiting_for_efo:
            self._cancel_current_efo_wait()

        # Jump now. If a worker is still running, wait until it finishes (ONE_FILE_DONE will fire),
        # then it will auto-advance; we override that by setting _jump_to_index.
        if getattr(self, "_worker_busy", False):
            self._jump_to_index = idx
            self.set_status(f"Jumping to: {disp} (as soon as current worker finishes)")
            return

        self._current_index = idx
        self.current_file_var.set(f"Current file: {disp}")

        # start session if needed
        if not self._is_running:
            self._is_running = True
            self.run_btn.config(state="disabled")
            self.clear_stats_history()

        self._start_worker_for_current_file()

    # ---------------- run control ----------------
    def run(self):
        self._refresh_file_list()
        data_path = self.data_path.get().strip()
        save_folder = self.save_folder.get().strip()

        if not data_path or not os.path.isdir(data_path):
            messagebox.showerror("Error", "Please select a valid MINFLUX data folder.")
            return
        if not save_folder:
            messagebox.showerror("Error", "Please set an output folder.")
            return

        os.makedirs(save_folder, exist_ok=True)

        if not self._all_npy_files:
            messagebox.showerror("Error", "No .npy files found in the selected data folder.")
            return

        self._is_running = True
        self._jump_to_index = None

        cur_disp = os.path.relpath(self._all_npy_files[self._current_index], data_path)
        self.current_file_var.set(f"Current file: {cur_disp}")

        self.run_btn.config(state="disabled")
        self.apply_efo_btn.config(state="disabled")
        self.continue_btn.config(state="disabled")

        self.clear_stats_history()
        self.set_status("Starting...")

        self._start_worker_for_current_file()

    def _start_worker_for_current_file(self):
        if not self._all_npy_files:
            return
        if self._current_index < 0 or self._current_index >= len(self._all_npy_files):
            self.set_status("No more files.")
            self._is_running = False
            self.run_btn.config(state="normal")
            return

        self.apply_efo_btn.config(state="disabled")
        self.continue_btn.config(state="disabled")
        self._worker_busy = True
        self._worker = threading.Thread(target=self._worker_run_one_file, daemon=True)
        self._worker.start()

    # ---------------- histogram + scatter ----------------
    def _show_efo_histogram_in_main_plot(self, efo_values, title, bin_size):
        efo_values = np.asarray(efo_values)
        self.ax_hist.clear()

        bins = np.arange(efo_values.min(), efo_values.max() + bin_size, bin_size)
        self.ax_hist.hist(efo_values, bins=bins, edgecolor="darkorange", color="darkorange")
        self.ax_hist.set_xlabel("EFO")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.set_title(title)
        self.ax_hist.set_xlim(efo_values.min() - 50000, efo_values.max() + 50000)

        self._span_xmin = float(efo_values.min())
        self._span_xmax = float(efo_values.max())

        def onselect(xmin, xmax):
            self._span_xmin, self._span_xmax = float(xmin), float(xmax)
            self.efo_range_var.set(f"{self._span_xmin:.2f}  ...  {self._span_xmax:.2f}")

        self._span = SpanSelector(
            self.ax_hist,
            onselect,
            direction="horizontal",
            useblit=True,
            props=dict(facecolor="yellow", alpha=0.3, edgecolor="black", linewidth=2),
            button=1,
            interactive=True,
            drag_from_anywhere=True,
        )

        self._span.extents = (self._span_xmin, self._span_xmax)
        onselect(self._span_xmin, self._span_xmax)

        self.canvas_hist.draw()
        self.apply_efo_btn.config(state="normal")
        self.continue_btn.config(state="disabled")
        self._applied_once = False

        self.set_status(f"Adjust EFO span, click 'Apply EFO' to preview, then 'Continue' ({title})")

    def _render_scatter(self, arr):
        if arr is None or len(arr) == 0:
            # Keep limits as-is; just clear points
            self._scatter_capture_enabled = False
            try:
                self.ax_scatter.clear()
                self.ax_scatter.set_xlabel("X")
                self.ax_scatter.set_ylabel("Y")
                # restore prior limits if we have them
                if self._scatter_xlim is not None and self._scatter_ylim is not None:
                    self.ax_scatter.set_xlim(self._scatter_xlim)
                    self.ax_scatter.set_ylim(self._scatter_ylim)
            finally:
                self._scatter_capture_enabled = True
            self.canvas_scatter.draw_idle()
            return

        use_mean = bool(self.avg_tid_var.get())

        if not use_mean:
            # ---- RAW mode (existing behavior): one point per localization ----
            locs_plot = arr["loc"][:, :2]  # (N,2)
            tids_for_color = arr["tid"]    # (N,)
        else:
            # ---- MEAN-per-tid mode: one point per tid ----
            # sort by tid for efficient reduceat
            order = np.argsort(arr["tid"])
            a = arr[order]
            tids = a["tid"]
            locs = a["loc"]  # (N,3)

            starts = np.r_[0, np.flatnonzero(tids[1:] != tids[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(a)])

            loc_sum = np.add.reduceat(locs, starts, axis=0)
            loc_mean = loc_sum / counts[:, None]  # (n_tids,3)

            locs_plot = loc_mean[:, :2]           # (n_tids,2)
            tids_for_color = tids[starts]         # (n_tids,)

        # ---- end-to-end distance per tid (for coloring) ----
        # compute for each tid in arr (using first and last localization in time order within tid)
        end_to_end = {}
        for tid in np.unique(arr["tid"]):
            pts = arr["loc"][arr["tid"] == tid]
            if len(pts) < 2:
                end_to_end[tid] = 0.0
            else:
                d = pts[-1] - pts[0]
                end_to_end[tid] = float(np.sqrt(np.sum(d * d)))

        vals = np.array([end_to_end[tid] for tid in tids_for_color], dtype=float)

        from matplotlib import cm, colors
        vmax = float(vals.max())
        vmin = float(vals.min())
        if vmax == vmin:
            vmax = vmin + 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cm.get_cmap("turbo")(norm(vals))

        self._scatter_capture_enabled = False
        try:
            self.ax_scatter.clear()
            self.ax_scatter.scatter(
                locs_plot[:, 0], locs_plot[:, 1],
                c=rgba,
                s=50,# if not use_mean else 18,
                alpha=0.9,# if use_mean else 0.25,
                linewidths=0,
                marker=".",
                rasterized=True
            )
            self.ax_scatter.set_xlabel("X")
            self.ax_scatter.set_ylabel("Y")
            self.ax_scatter.set_aspect("equal", adjustable="box")


            # first draw: autoscale is fine, then store as baseline
            self.ax_scatter.autoscale(enable=True, axis="both")
            xlim = self.ax_scatter.get_xlim()
            ylim = self.ax_scatter.get_ylim()
            xlim, ylim = self._pad_limits_to_axes_aspect(xlim, ylim)
            self.ax_scatter.set_xlim(xlim)
            self.ax_scatter.set_ylim(ylim)
            self._scatter_xlim, self._scatter_ylim = xlim, ylim
        finally:
            self._scatter_capture_enabled = True

        self.canvas_scatter.draw_idle()

    def _pad_limits_to_axes_aspect(self, xlim, ylim):
        # Make (xrange/yrange) match the axes pixel aspect while keeping equal scaling
        x0, x1 = xlim
        y0, y1 = ylim
        xr = x1 - x0
        yr = y1 - y0
        if xr <= 0 or yr <= 0:
            return xlim, ylim

        # axes box aspect in pixels (width/height)
        bbox = self.ax_scatter.get_window_extent().transformed(self.fig_scatter.dpi_scale_trans.inverted())
        ax_w, ax_h = bbox.width, bbox.height
        if ax_h == 0:
            return xlim, ylim
        target = ax_w / ax_h

        cur = xr / yr
        if cur < target:
            # need more x-range
            new_xr = target * yr
            pad = (new_xr - xr) / 2
            x0 -= pad
            x1 += pad
        else:
            # need more y-range
            new_yr = xr / target
            pad = (new_yr - yr) / 2
            y0 -= pad
            y1 += pad

        return (x0, x1), (y0, y1)

    # ---------------- Apply / Continue ----------------
    def _apply_efo_preview(self):
        if not self._waiting_for_efo or self._efo_ctx is None:
            return

        xmin, xmax = float(self._span_xmin), float(self._span_xmax)

        arr = self._efo_ctx["arr"]
        base = self._efo_ctx["base"]
        display_name = self._efo_ctx["display_name"]
        total_tim = self._efo_ctx["total_tim"]
        total_loc = self._efo_ctx["total_loc"]
        last_iteration_loc = self._efo_ctx["last_iteration_loc"]
        after_trace = self._efo_ctx["after_trace"]

        mask = (arr["efo"] >= xmin) & (arr["efo"] <= xmax)
        after_efo = int(np.count_nonzero(mask))

        # redraw scatter with filtered data
        arr_efo = arr[mask]
        loc_prec_prev = self._preview_localization_precision(arr_efo)
        loc_prec_prev = tuple(float(f'{v:.2f}') for v in loc_prec_prev) if loc_prec_prev is not None else None
        # snapshot current view (this is the zoom the user currently sees)
        cur_xlim = self.ax_scatter.get_xlim()
        cur_ylim = self.ax_scatter.get_ylim()

        self._render_scatter(arr_efo)

        # restore the user's view
        self.ax_scatter.set_xlim(cur_xlim)
        self.ax_scatter.set_ylim(cur_ylim)

        # then pad limits to match the axes rectangle while keeping equal aspect
        xlim, ylim = self._pad_limits_to_axes_aspect(cur_xlim, cur_ylim)
        self.ax_scatter.set_xlim(xlim)
        self.ax_scatter.set_ylim(ylim)

        self.canvas_scatter.draw_idle()
        items = [
            ("File", base),
            ("Total imaging time (min)", f"{total_tim/60:.2f}"),
            ("Total raw localizations", str(total_loc)),
            ("Last iteration localizations", str(last_iteration_loc)),
            ("After trace filtering", str(after_trace)),
            ("After EFO filtering", str(after_efo)),
            ("Localization precision", str(loc_prec_prev) if loc_prec_prev is not None else "—"),
            ("% remaining", f"{(after_efo/total_loc*100):.2f}%"),
        ]
        
        self.push_stats(display_name, items, auto_select=True)

        self._applied_once = True
        self.continue_btn.config(state="normal")
        self.set_status("Preview updated. Adjust span and Apply again, or click Continue.")

    def _continue_to_next_file(self):
        if not self._waiting_for_efo:
            return

        self.apply_efo_btn.config(state="disabled")
        self.continue_btn.config(state="disabled")

        if self._efo_result_holder is not None:
            self._efo_result_holder["result"] = (float(self._span_xmin), float(self._span_xmax))
        if self._efo_event is not None:
            self._efo_event.set()

        self._waiting_for_efo = False
        self._efo_ctx = None
        self.set_status("Continuing processing…")

    def _cancel_current_efo_wait(self):
        if not self._waiting_for_efo:
            return

        # mark cancelled so worker returns None
        self._efo_cancelled = True

        # ensure the worker unblocks
        if self._efo_result_holder is not None:
            self._efo_result_holder["result"] = None
        if self._efo_event is not None:
            self._efo_event.set()

        # reset GUI-side waiting state
        self._waiting_for_efo = False
        self._efo_ctx = None
        self.apply_efo_btn.config(state="disabled")
        self.continue_btn.config(state="disabled")

    # ---------------- worker/main queue ----------------
    def _poll_requests(self):
        try:
            while True:
                req = self._req_q.get_nowait()

                if req["type"] == "EFO_SELECT_EMBEDDED":
                    self._efo_cancelled = False
                    self._waiting_for_efo = True
                    self._efo_event = req["event"]
                    self._efo_result_holder = req["result_holder"]
                    self._efo_ctx = req.get("ctx")

                    self._show_efo_histogram_in_main_plot(
                        efo_values=req["efo_values"],
                        title=req["title"],
                        bin_size=req["bin_size"],
                    )
                    if self._efo_ctx is not None:
                        file_id = self._efo_ctx.get("base")  # or use full path if you prefer
                        if file_id != self._scatter_file_id:
                            self._scatter_file_id = file_id
                            self._scatter_xlim = None
                            self._scatter_ylim = None
                        self._render_scatter(self._efo_ctx.get("arr"))

                elif req["type"] == "STATS_KV":
                    self.push_stats(req["display_name"], req["items"], auto_select=True)

                elif req["type"] == "STATUS":
                    self.set_status(req["msg"])

                elif req["type"] == "CURRENT_FILE":
                    disp = req["display"]
                    self.current_file_var.set(f"Current file: {disp}")
                    self.file_choice.set(disp)

                elif req["type"] == "ONE_FILE_DONE":
                    self._worker_busy = False
                    # apply pending jump, else advance
                    if self._jump_to_index is not None:
                        self._current_index = self._jump_to_index
                        self._jump_to_index = None
                    else:
                        self._current_index += 1

                    # end?
                    if self._current_index >= len(self._all_npy_files):
                        self._is_running = False
                        self.run_btn.config(state="normal")
                        self.apply_efo_btn.config(state="disabled")
                        self.continue_btn.config(state="disabled")
                        self.set_status("Done.")
                    else:
                        self._start_worker_for_current_file()

        except queue.Empty:
            pass

        self.after(100, self._poll_requests)

    def _preview_localization_precision(self, arr):
        """
        arr: structured numpy array with fields 'tid' and 'loc' (Nx3)
        Returns tuple of median per-trace std for each loc dimension (x,y,z if present).
        """
        if arr is None or len(arr) == 0:
            return None

        locs = arr["loc"]
        tids = arr["tid"]

        # number of loc dimensions (2 or 3)
        n_dim = locs.shape[1]

        meds = []
        for d in range(n_dim):
            stds = []
            for tid in np.unique(tids):
                pts = locs[tids == tid, d]
                if len(pts) >= 2:
                    stds.append(float(np.std(pts, ddof=1)))
            if len(stds) == 0:
                meds.append(float("nan"))
            else:
                meds.append(float(np.median(stds)))
        return tuple(meds)
    
    def _request_efo_range(self, efo_values, title, ctx):
        ev = threading.Event()
        holder = {"result": None}
        self._req_q.put(
            {
                "type": "EFO_SELECT_EMBEDDED",
                "efo_values": efo_values,
                "title": title,
                "bin_size": float(self.bin_size.get()),
                "event": ev,
                "result_holder": holder,
                "ctx": ctx,
            }
        )
        ev.wait()
        # If GUI cancelled (user jumped), return None => worker skips saving
        if self._efo_cancelled:
            return None
        return holder["result"]

    # ---------------- worker ----------------
    def _worker_run_one_file(self):
        try:
            self._process_one_file(self._all_npy_files[self._current_index], self._current_index)
        except Exception as e:
            self._status_from_worker(f"ERROR: {e}")
        finally:
            self._req_q.put({"type": "ONE_FILE_DONE"})

    def _process_one_file(self, file, idx):
        data_path = self.data_path.get().strip()
        save_folder = self.save_folder.get().strip()

        min_trace_length = int(self.min_trace_length.get())
        z_correction_factor = float(self.z_correction_factor.get())
        scale_factor = float(self.scale_factor.get())

        base = os.path.splitext(os.path.basename(file))[0]
        disp = os.path.relpath(file, data_path)

        self._req_q.put({"type": "CURRENT_FILE", "display": disp})
        self._status_from_worker(f"Processing file {idx+1}/{len(self._all_npy_files)}: {base}")

        display_name = f"[{idx+1}/{len(self._all_npy_files)}] {base}"
        save_path = os.path.join(save_folder, f"{base}.csv")
        avg_save_path = os.path.join(save_folder, f"{base}_stats.csv")

        MFX_Data = np.load(file, allow_pickle=False)

        total_tim = MFX_Data["tim"][-1] - MFX_Data["tim"][0]
        total_loc = len(MFX_Data)

        # vld filter
        MFX_Data = MFX_Data[MFX_Data["vld"] == True]
        if len(MFX_Data) == 0:
            self._status_from_worker(f"Skipping {base}: no valid localizations.")
            return

        # z correction + scaling
        MFX_Data = MFX_Data.copy()
        MFX_Data["loc"][:, -1] *= z_correction_factor
        MFX_Data["loc"] *= scale_factor

        # final iteration only
        MFX_Data_vld_fnl = MFX_Data[MFX_Data["itr"] == max(MFX_Data["itr"])]

        # trace filtering
        unique_tids, inv_idx, locs_per_tid = np.unique(
            MFX_Data_vld_fnl["tid"], return_inverse=True, return_counts=True
        )
        MFX_Data_vld_fnl_filt = MFX_Data_vld_fnl[locs_per_tid[inv_idx] >= min_trace_length]
        if len(MFX_Data_vld_fnl_filt) == 0:
            self._status_from_worker(f"Skipping {base}: no data after trace filtering.")
            return

        # EFO values (max itr)
        max_itr = np.max(MFX_Data_vld_fnl_filt["itr"])
        efo_vals = MFX_Data_vld_fnl_filt["efo"][MFX_Data_vld_fnl_filt["itr"] == max_itr]
        if len(efo_vals) == 0:
            self._status_from_worker(f"Skipping {base}: no EFO values at max iteration.")
            return

        self._status_from_worker(f"Waiting for EFO selection: {base} ({idx+1}/{len(self._all_npy_files)})")
        ctx = {
            "arr": MFX_Data_vld_fnl_filt,
            "base": base,
            "display_name": display_name,
            "total_tim": total_tim,
            "total_loc": total_loc,
            "last_iteration_loc": len(MFX_Data_vld_fnl),
            "after_trace": len(MFX_Data_vld_fnl_filt),
        }

        chosen = self._request_efo_range(efo_vals, title=base, ctx=ctx)
        if chosen is None:
            self._status_from_worker(f"Skipping {base}: no EFO range selected.")
            return

        xmin, xmax = chosen

        # apply EFO filtering
        MFX_filtered = MFX_Data_vld_fnl_filt[
            (MFX_Data_vld_fnl_filt["efo"] >= xmin) & (MFX_Data_vld_fnl_filt["efo"] <= xmax)
        ]
        df = np_to_df(MFX_filtered)

        # per-trace stats
        dims = [c.replace("loc_", "") for c in df.columns if c.startswith("loc_")]
        avg_df = pd.DataFrame()
        if len(df) > 0 and dims:
            for dim in dims:
                avg_df[f"loc_{dim}_mean"] = df.groupby("tid")[f"loc_{dim}"].mean()
                avg_df[f"loc_{dim}_std"] = df.groupby("tid")[f"loc_{dim}"].std()
            avg_df["n"] = df.groupby("tid")[f"loc_{dims[0]}"].count()
            avg_df["tim_tot"] = df.groupby("tid")["tim"].sum()

        # --- localization precision: median of per-trace std per dimension ---
        loc_prec = None
        if len(avg_df) > 0 and dims:
            meds = []
            for dim in dims:
                s = avg_df[f"loc_{dim}_std"].dropna()
                meds.append(float(s.median()) if len(s) else float("nan"))
            loc_prec = tuple(meds)   # e.g. (prec_x, prec_y, prec_z)
        loc_prec = tuple(float(f"{v:.2f}") for v in loc_prec)
        # save
        save_to_csv(df, save_path)
        avg_df.to_csv(avg_save_path, index=True)

        # final stats to output panel
        items = [
            ("File", base),
            ("Total imaging time (min)", f"{total_tim/60:.2f}"),
            ("Total raw localizations", str(total_loc)),
            ("Last iteration localizations", str(len(MFX_Data_vld_fnl))),
            ("After trace filtering", str(len(MFX_Data_vld_fnl_filt))),
            ("After EFO filtering", str(len(df))),
            ("Localization precision", str(loc_prec) if loc_prec is not None else "—"),
            ("% remaining", f"{len(df)/total_loc*100:.2f}%"),
        ]
        self._req_q.put({"type": "STATS_KV", "display_name": display_name, "items": items})

        self._status_from_worker(
            f"Saved ({idx+1}/{len(self._all_npy_files)}): {os.path.basename(save_path)} + {os.path.basename(avg_save_path)}"
        )


if __name__ == "__main__":
    app = MinfluxFilterGUI()
    app.mainloop()