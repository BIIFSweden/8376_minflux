import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import webview

# -----------------------------
# Load + process your MINFLUX data (same as before)
# -----------------------------
MFX_PATH = '/Users/maximiliansenftleben/Data/8376_alm/8376_minflux/data/for Max_MFX_echange_2colour_2D/241008-133841_NUP96_minflux/241008-133841_NUP96_minflux.npy'

MFX_Data = np.load(MFX_PATH)
MFX_Data = MFX_Data[MFX_Data['vld'] == True]
MFX_Data = MFX_Data[MFX_Data['itr'] == np.max(MFX_Data['itr'])]

unique_tids, inv_idx, locs_per_tid = np.unique(
    MFX_Data['tid'], return_inverse=True, return_counts=True
)
MFX_Data = MFX_Data[locs_per_tid[inv_idx] > 3]

tids = np.unique(MFX_Data['tid'])
end_to_end_dis = np.empty(len(tids), dtype=float)
for i, tid in enumerate(tids):
    tr = MFX_Data['loc'][MFX_Data['tid'] == tid, :]
    d_xyz = tr[-1, :] - tr[0, :]
    end_to_end_dis[i] = np.sqrt(np.sum(d_xyz**2))

_, inv_indx = np.unique(MFX_Data['tid'], return_inverse=True)
vals = end_to_end_dis[inv_indx]

locs = MFX_Data['loc']
locs = locs * 1e9
x, y, z = locs[:, 0], locs[:, 1], locs[:, 2]

# -----------------------------
# 3D Plotly figure
# -----------------------------
fig = go.Figure(
    data=go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=2,                 # 3D markers are visually larger; keep small
            opacity=0.85,
            color=vals,
            colorscale="Turbo",
            colorbar=dict(title="end-to-end"),
        ),
        # If you want hover, keep this; otherwise comment out and disable hover below
        customdata=np.stack([MFX_Data["tid"], vals], axis=1),
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "z=%{z:.3f}<br>"
            "tid=%{customdata[0]}<br>"
            "end2end=%{customdata[1]:.3f}"
            "<extra></extra>"
        ),
    )
)

# Keep aspect ratio sensible in 3D (prevents weird stretching)
fig.update_layout(
    title="MINFLUX localizations (3D, colored by trace end-to-end distance)",
    template="plotly_white",
    width=900, height=900,
    uirevision="keep",
    margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(
        aspectmode="data",   # respects data scale in x/y/z
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
    ),
)

# If you want maximum smoothness, uncomment:
# fig.update_traces(hoverinfo="skip", hovertemplate=None)

html = pio.to_html(
    fig,
    full_html=True,
    include_plotlyjs="cdn",
    config={
        "scrollZoom": True,     # wheel zoom works in 3D too (zooms camera)
        "responsive": False,
        "displaylogo": False,
    },
)

webview.create_window("MINFLUX 3D viewer", html=html, width=920, height=920)
webview.start()