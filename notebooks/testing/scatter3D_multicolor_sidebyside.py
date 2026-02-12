import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

x1, y1, z1 = locs_1[:, 0], locs_1[:, 1], locs_1[:, 2]
x2, y2, z2 = locs_2[:, 0], locs_2[:, 1], locs_2[:, 2]

# -----------------------------
# Build 3D Plotly figure with side-by-side subplots
# -----------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(os.path.basename(files[0]), os.path.basename(files[1])),
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    horizontal_spacing=0.02,
)

# File 1 - Orange (left subplot, scene)
fig.add_trace(
    go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode="markers",
        name="Channel 1",
        marker=dict(
            size=2,
            opacity=0.8,
            color="orange",
        ),
        customdata=MFX_Data_1["tid"],
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "z=%{z:.3f}<br>"
            "tid=%{customdata}"
            "<extra></extra>"
        ),
    ),
    row=1, col=1
)

# File 2 - Red (right subplot, scene2)
fig.add_trace(
    go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode="markers",
        name="Channel 2",
        marker=dict(
            size=2,
            opacity=0.8,
            color="red",
        ),
        customdata=MFX_Data_2["tid"],
        hovertemplate=(
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "z=%{z:.3f}<br>"
            "tid=%{customdata}"
            "<extra></extra>"
        ),
    ),
    row=1, col=2
)

# -----------------------------
# Configure both 3D scenes with identical settings
# -----------------------------
scene_settings = dict(
    xaxis=dict(
        title="X (nm)",
        backgroundcolor="white",
        gridcolor="lightgray",
        showbackground=True,
    ),
    yaxis=dict(
        title="Y (nm)",
        backgroundcolor="white",
        gridcolor="lightgray",
        showbackground=True,
    ),
    zaxis=dict(
        title="Z (nm)",
        backgroundcolor="white",
        gridcolor="lightgray",
        showbackground=True,
    ),
    aspectmode='data',
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
    ),
)

fig.update_layout(
    title="MINFLUX localizations 3D - Synchronized view (2 channels)",
    template="plotly_white",
    margin=dict(l=10, r=10, t=60, b=10),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    scene=scene_settings,
    scene2=scene_settings,
)
fig.update_layout(uirevision="keep")

# -----------------------------
# Generate base HTML
# -----------------------------
base_html = pio.to_html(
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
# Add custom JavaScript for camera synchronization
# -----------------------------
sync_script = """
<script>
(function() {
    // Wait for Plotly to be ready
    function waitForPlotly() {
        if (typeof Plotly === 'undefined') {
            setTimeout(waitForPlotly, 100);
            return;
        }
        setupSync();
    }
    
    function setupSync() {
        // Find the plot div (Plotly creates it with a specific class)
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (!gd) {
            setTimeout(setupSync, 100);
            return;
        }
        
        var isUpdating = false;
        
        // Listen for camera changes on the plot
        gd.on('plotly_relayout', function(eventData) {
            if (isUpdating) return;
            
            isUpdating = true;
            var update = {};
            
            // If scene (left) camera changed, update scene2 (right)
            if (eventData['scene.camera']) {
                update['scene2.camera'] = eventData['scene.camera'];
            }
            // If scene2 (right) camera changed, update scene (left)
            if (eventData['scene2.camera']) {
                update['scene.camera'] = eventData['scene2.camera'];
            }
            
            // Apply the synchronized camera update
            if (Object.keys(update).length > 0) {
                Plotly.relayout(gd, update).then(function() {
                    isUpdating = false;
                });
            } else {
                isUpdating = false;
            }
        });
        
        console.log('3D camera sync enabled');
    }
    
    waitForPlotly();
})();
</script>
"""

# Insert the sync script before closing </body> tag
html = base_html.replace('</body>', sync_script + '</body>')

# -----------------------------
# Show in a native WebView window
# -----------------------------
webview.create_window("MINFLUX 3D viewer (synchronized)", html=html, width=1600, height=800)
webview.start()