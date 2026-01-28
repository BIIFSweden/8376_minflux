import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # for Tkinter embedding
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import time

import time
def add_wheel_zoom(fig, ax, base_scale=1.2):
    def zoom_fun(event):
        # Only zoom if the mouse is over this axes
        if event.inaxes != ax:
            return
        # event.button is 'up' or 'down'
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        xdata, ydata = event.xdata, event.ydata  # mouse position in data coords
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width  = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width  * (1 - relx), xdata + new_width  * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', zoom_fun)




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
        ax.set_ylim(y0 - dy_data, y1 - dy_data)  # <- not inverted anymore

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


MFX_Data = '/Users/maximiliansenftleben/Data/8376_alm/8376_minflux/data/for Max_MFX_echange_2colour_2D/241008-133841_NUP96_minflux/241008-133841_NUP96_minflux.npy'
MFX_Data = np.load(MFX_Data)
print('all raw                  ',len(MFX_Data))
MFX_Data =  MFX_Data[MFX_Data['vld'] == True]
print('only valid_              ', len(MFX_Data))
MFX_Data =  MFX_Data[MFX_Data['itr'] == max(MFX_Data['itr'])]
print('last iteration only      ',len(MFX_Data))
# Filtering for valid and last localizations
MFX_Data =  MFX_Data[(MFX_Data['itr'] == max(MFX_Data['itr'])) & (MFX_Data['itr'] == max(MFX_Data['itr']))]
#print('last iteration only.diff ',len(MFX_Data))
  # Get number of final localizations per trace 
unique_tids, inv_idx, locs_per_tid = np.unique(MFX_Data['tid'],  return_inverse=True,return_counts=True) 
print(f'min trace                 {np.min(locs_per_tid)}')
# Keep only traces with at least 3 localizations
MFX_Data = MFX_Data[locs_per_tid[inv_idx]>3] 
print('after trace length filt  ',len(MFX_Data))
# Get number of final localizations per trace 
_,locs_per_tid_filt = np.unique(MFX_Data['tid'], return_counts=True) 
print(f'Minimum trace             {np.min(locs_per_tid_filt)}')

end_to_end_dis = []
for tid in np.unique(MFX_Data['tid']):
     d_xyz = MFX_Data['loc'][MFX_Data['tid'] == tid,:][-1,:]- MFX_Data['loc'][MFX_Data['tid'] == tid,:][0,:]
     end_to_end_dis.append(np.sqrt(np.sum(d_xyz**2)) ) 
end_to_end_dis = np.vstack(end_to_end_dis)    


_, inv_indx = np.unique(MFX_Data['tid'], return_inverse = True)

# get localizations
locs = MFX_Data['loc']
vals = end_to_end_dis[inv_indx].ravel()
norm = colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
rgba = cm.get_cmap("turbo")(norm(vals))          # (N,4) float colors computed once

fig, ax = plt.subplots(figsize=(5,5), dpi=200)

sc = ax.scatter(
    locs[:,0], locs[:,1],
    c=rgba,                # fixed RGBA, no cmap/norm during redraw
    s=50,
    alpha=0.9,
    linewidths=0,
    marker='.',
    rasterized=True
)
#% Create a figure with 1 row, 2 columns

# First subplot
ax.set_title('Scatter Plot 1')
ax.set_xlabel('X1')
ax.set_ylabel('Y1')

add_wheel_zoom(fig, ax)
add_drag_pan_fast(fig, ax, button=1)
plt.show()

# Layout adjustment

