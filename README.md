# Pyflux

A desktop GUI for loading, filtering, visualizing, aligning, clustering, and exporting MINFLUX localization data from .npy files.

![Demo GIF](./assets/vid.gif)

- Read MINFLUX .npy exports in both Imspector layouts: one row per iteration
  (Imspector 24.10 and later) and the legacy layout with the iterations nested
  under `itr` (older Imspector versions, and files saved by pyMINFLUX)
- Filter localizations by trace length, EFO, and CFR
- View data in interactive 2D/3D scatter plots
- Compare datasets in a multicolor viewer
- Generate heatmaps, Gaussian renderings, and TIFF exports
- Align datasets using MBM bead-based registration
- Run DBSCAN clustering on track-averaged localizations
- Export filtered data and summary statistics to CSV

Built with PySide6, napari-storm, Plotly, Pandas, SciPy, and Zarr.

## Rendering

Localizations are drawn by [napari-storm](https://github.com/napari-storm/napari-storm)
in a napari canvas embedded in the pyflux window (no separate napari window
opens). This replaced the Plotly scatter plots in the main window and the
multicolor viewer; the binning and DBSCAN windows still use Plotly.

- **Points are the default.** Each localization is an opaque point, and points
  hide each other by depth, so overlaps and 3-D views come out right whatever
  order the points are drawn in. Points are never smaller than 5 screen pixels,
  so a zoomed-out overview stays visible.
- **Render as** (under *3D*) lists napari-storm's rendering styles.
  *Gaussian (scientific)* draws each localization as a Gaussian summed with its
  neighbours, like a super-resolution image. The alternative visualisations are
  points, outlined points, rings, spheres, crosses, squares and diamonds, which
  are opaque and hide each other by depth, and napari-particles' sprites, which
  glow and add up.
- **Size (nm)** in the colour settings table is a marker's radius, or the
  Gaussian width (σ), in nanometres. It used to be the marker size in pixels.
- **α** applies to the Gaussian and the sprites, and is greyed out for the
  opaque markers.
- The styles come from napari-storm's footprint palette, newer than 2.1.0.
  With 2.1.0 pyflux draws Gaussians and the *Render as* menu is locked.
- Colour modes are unchanged: *solid*, *depth* (z), *tid* and *end-to-end*.
  In the merged multicolor view, depth uses one z scale for all channels.
- **Crop** in the multicolor viewer keeps what a 2-D panel is showing: pan and
  zoom to the region, then press Crop. In grid mode the panel the pointer touched
  last is used; a synchronized panel crops its whole group.
- The **3D** box switches every viewer; napari's own 2D/3D button does the same.
- The scale bar and the axes overlay are napari's; the *Axes* box in the
  multicolor viewer hides the axes.

## Running from source

```
pip install -r requirements.txt
python src/pyflux/main.py
```

`pytest -q` runs the tests; `tests/test_napari_storm_view.py` covers the
napari-storm path, including one test that opens a viewer and needs a display.

## Installation

- Download the current release
- Unzip
- For Macos: Allow app to run under Privacy & Security/Security -> Run anyway
- For Windows: Trust developer

## Manual

Soon to come.

## TODO 

- add more filtering options
- fix mbm bead plots

## Known bugs pyflux

- sometimes opens second window

## License

[MIT](LICENSE)

## Contact

[SciLifeLab BioImage Informatics Unit](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Maximilian Senftleben](mailto:maximilian.senftleben@scilifelab.se)
