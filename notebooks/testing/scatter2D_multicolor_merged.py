import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import webview
import os
import glob

# -----------------------------
# Define filtering function to avoid code duplication
# -----------------------------
def filter_minflux_data(file_path, verbose=True):
    """Load and filter MINFLUX data from a single file."""
    MFX_Data = np.load(file_path)
    if verbose:
        print(f'\n--- Processing: {os.path.basename(file_path)} ---')
        print('all raw                  ', len(MFX_Data))

    # Keep only valid localizations
    MFX_Data = MFX_Data[MFX_Data['vld'] == True]
    if verbose:
        print('only valid_              ', len(MFX_Data))

    # Keep only last iteration
    MFX_Data = MFX_Data[MFX_Data['itr'] == np.max(MFX_Data['itr'])]
    if verbose:
        print('last iteration only      ', len(MFX_Data))

    # Number of final localizations per trace
    unique_tids, inv_idx, locs_per_tid = np.unique(
        MFX_Data['tid'], return_inverse=True, return_counts=True
    )
    if verbose:
        print(f'min trace                 {np.min(locs_per_tid)}')

    # Keep only traces with at least 4 localizations
    MFX_Data = MFX_Data[locs_per_tid[inv_idx] > 3]
    if verbose:
        print('after trace length filt  ', len(MFX_Data))

    _, locs_per_tid_filt = np.unique(MFX_Data['tid'], return_counts=True)
    if verbose:
        print(f'Minimum trace             {np.min(locs_per_tid_filt)}')

    return MFX_Data

# -----------------------------
# Load + process both MINFLUX data files
# -----------------------------
path = '/Users/maximiliansenftleben/Data/8376_alm/8376_minflux/data/for Max_MFX_echange_2colour_2D/'
files = [i for i in glob.glob(os.path.join(path, "**", "*.npy"), recursive=True) if i.endswith('.npy')]

print(f"Found {len(files)} files:")
for f in files:
    print(f"  - {os.path.basename(f)}")

# Process both files
MFX_Data_1 = filter_minflux_data(files[0])
MFX_Data_2 = filter_minflux_data(files[1])

# Convert locations to nanometers
locs_1 = MFX_Data_1['loc'] * 1e9
locs_2 = MFX_Data_2['loc'] * 1e9

x1, y1 = locs_1[:, 0], locs_1[:, 1]
x2, y2 = locs_2[:, 0], locs_2[:, 1]

# -----------------------------
# Build Plotly figure with two traces (different colors)
# -----------------------------
fig = go.Figure()

# File 1 - Orange
fig.add_trace(
    go.Scattergl(
        x=x1, y=y1,
        mode="markers",
        name=os.path.basename(files[0]),
        marker=dict(
            size=3,
            opacity=0.9,
            color="orange",
        ),
        customdata=MFX_Data_1["tid"],
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "tid=%{customdata}<br>"
            f"file={os.path.basename(files[0])}"
            "<extra></extra>"
        ),
    )
)

# File 2 - Red
fig.add_trace(
    go.Scattergl(
        x=x2, y=y2,
        mode="markers",
        name=os.path.basename(files[1]),
        marker=dict(
            size=3,
            opacity=0.9,
            color="red",
        ),
        customdata=MFX_Data_2["tid"],
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "tid=%{customdata}<br>"
            f"file={os.path.basename(files[1])}"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="MINFLUX localizations (2 channels merged)",
    template="plotly_white",
    dragmode="pan",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    ),
)
fig.layout.margin.autoexpand = False
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(uirevision="keep")

# Convert to HTML for the embedded webview
html = pio.to_html(
    fig,
    full_html=True,
    include_plotlyjs="cdn",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": False,
    },
)

# -----------------------------
# Show in a native WebView window
# -----------------------------
webview.create_window("MINFLUX viewer (2 channels)", html=html, width=900, height=900)
webview.start()