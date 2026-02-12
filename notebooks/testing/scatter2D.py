import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import webview

# -----------------------------
# Load + process your MINFLUX data (same logic as your script)
# -----------------------------
MFX_PATH = '/Users/maximiliansenftleben/Data/8376_alm/8376_minflux/data/for Max_MFX_echange_2colour_2D/241008-133841_NUP96_minflux/241008-133841_NUP96_minflux.npy'

MFX_Data = np.load(MFX_PATH)
print('all raw                  ', len(MFX_Data))

MFX_Data = MFX_Data[MFX_Data['vld'] == True]
print('only valid_              ', len(MFX_Data))

MFX_Data = MFX_Data[MFX_Data['itr'] == np.max(MFX_Data['itr'])]
print('last iteration only      ', len(MFX_Data))

# Number of final localizations per trace
unique_tids, inv_idx, locs_per_tid = np.unique(
    MFX_Data['tid'], return_inverse=True, return_counts=True
)
print(f'min trace                 {np.min(locs_per_tid)}')

# Keep only traces with at least 4 localizations (your code used >3)
MFX_Data = MFX_Data[locs_per_tid[inv_idx] > 3]
print('after trace length filt  ', len(MFX_Data))

_, locs_per_tid_filt = np.unique(MFX_Data['tid'], return_counts=True)
print(f'Minimum trace             {np.min(locs_per_tid_filt)}')

# End-to-end distance per trace
tids = np.unique(MFX_Data['tid'])
end_to_end_dis = np.empty(len(tids), dtype=float)

for i, tid in enumerate(tids):
    tr = MFX_Data['loc'][MFX_Data['tid'] == tid, :]
    d_xyz = tr[-1, :] - tr[0, :]
    end_to_end_dis[i] = np.sqrt(np.sum(d_xyz**2))

# Map each localization to its trace's end-to-end distance
# (unique() here returns tids in sorted order, matching 'tids' above)
_, inv_indx = np.unique(MFX_Data['tid'], return_inverse=True)
vals = end_to_end_dis[inv_indx]  # per-point value for coloring

locs = MFX_Data['loc']
locs = locs * 1e9
x, y = locs[:, 0], locs[:, 1]

# -----------------------------
# Build Plotly figure (WebGL) + enable wheel zoom
# -----------------------------
fig = go.Figure(
    data=go.Scattergl(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=3,                 # tune: 2-6 depending on density
            opacity=0.9,
            color=vals,
            colorscale="Turbo",
            colorbar=dict(title="end-to-end"),
        ),
        customdata=np.stack([MFX_Data["tid"], vals], axis=1),
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "tid=%{customdata[0]}<br>"
            "end2end=%{customdata[1]:.3f}"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="MINFLUX localizations (colored by trace end-to-end distance)",
    template="plotly_white",
    dragmode="pan",               # left-drag pans (like your mpl pan)
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.layout.margin.autoexpand=False
# optional: keep equal aspect ratio like typical localization plots
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(uirevision="keep")  # preserves current zoom/pan state cleanly

# Convert to HTML for the embedded webview
html = pio.to_html(
    fig,
    full_html=True,
    include_plotlyjs="cdn",        # change to True for fully offline (bigger HTML)
    config={
        "scrollZoom": True,        # <-- mousewheel zoom
        "displaylogo": False,
        "responsive": False,
    },
)

# -----------------------------
# Show in a native WebView window (wheel zoom works)
# -----------------------------
webview.create_window("MINFLUX viewer", html=html, width=900, height=900)
webview.start()