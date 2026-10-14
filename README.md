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

Built with PySide6, Plotly, Pandas, SciPy and Zarr; napari-storm for rendering
when it is installed.

## Rendering

Localizations are drawn by [napari-storm](https://github.com/napari-storm/napari-storm)
in a napari canvas embedded in the pyflux window (no separate napari window
opens) when napari-storm is installed. It replaces the Plotly scatter plots in
the main window and the multicolor viewer; the binning and DBSCAN windows use
Plotly either way. **napari-storm is optional**: without it pyflux draws the
Plotly scatter plots it always did, and the controls below that need
napari-storm are greyed out.

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
- **Contrast** windows the summed image of the Gaussian and the sprites: the
  value is the number of overlapping localizations drawn at the full colour,
  so lower is brighter. It is greyed out for the opaque markers, which have no
  summed image, and without napari-storm.
- **Size (nm)** in the colour settings table is a marker's radius, or the
  Gaussian width (σ), in nanometres. It used to be the marker size in pixels.
- **α** is the opacity. It applies to the Gaussian and the sprites, and to
  Plotly markers; it is greyed out for the opaque napari-storm markers.
- The styles come from napari-storm's footprint palette, newer than 2.1.0.
  With 2.1.0 pyflux draws Gaussians and the *Render as* menu is locked.
- **Colour modes**, per dataset in the colour settings table: *solid*, *depth*
  (z), *tid* (one colour per trace), *end-to-end* (the displacement of each
  trace) and *property*, which colours by any numeric column of the file —
  EFO, CFR, photon counts (`eco`, `ecc`), `efc`, `fbg`, `dcr`, time, trace id,
  iteration... The *Property* column picks the column (the file's own columns
  are listed once it is loaded) and *Limits* the range the LUT spans, as
  `low, high`; empty means the 1st to 99th percentile of the values, and one
  end can be left empty to keep it automatic. In the merged multicolor view,
  depth uses one z scale for all channels and a property one range for every
  channel coloured by it.
- **Crop** in the multicolor viewer keeps what a 2-D panel is showing: pan and
  zoom to the region, then press Crop. In grid mode the panel the pointer touched
  last is used; a synchronized panel crops its whole group. (With Plotly, Crop
  keeps the last box selection, as before.)
- The **3D** box switches every viewer; napari's own 2D/3D button does the same.
- The scale bar and the axes overlay are napari's; the *Axes* box in the
  multicolor viewer hides the axes (*Grid* with Plotly).

## MINFLUX sequences

Whatever the sequence, the localizations shown are those of its last
iteration, and the CFR of each is taken from the last iteration at which the
CFR was measured — in a 3-D sequence the second-to-last, the final iteration
carrying a constant instead. An iteration is recognised as measured when its
CFR values vary; NaN entries (iterations a re-localization skipped, in the
legacy layout) are ignored in that test.

## MBM alignment

*Align* registers every dataset onto the one ticked as *mbm source*, using
the MBM bead tracks Imspector writes as a `grd/mbm` folder inside each
measurement's folder. pyflux looks for that folder next to each .npy file,
inside a folder named after the file (`<name>/grd/mbm`, `<name>.zarr/grd/mbm`),
a few folders further down, and in the folders above the file up to the data
folder. It never takes a folder that may belong to another file. When the
bead data of a file cannot be found, *Align* lists the files and offers to
open *Parameters*, where *MBM folders (per file)* takes the measurement
folder (or its `grd/mbm` folder) for each.

## Running from source

```
pip install -r requirements.txt
python src/pyflux/main.py
```

That draws with Plotly. For napari-storm rendering also install the optional
requirements (napari, napari-storm and what they need):

```
pip install -r requirements.txt -r requirements-napari-storm.txt
```

pyflux uses napari-storm whenever it can be imported; `PYFLUX_RENDERER=plotly`
in the environment forces the Plotly path on a machine that has it.

`pytest -q` runs the tests; `tests/test_napari_storm_view.py` covers the
napari-storm path, including one test that opens a viewer and needs a display,
and `tests/test_rendering_backend.py` the Plotly fallback.

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
