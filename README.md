# Pyflux

A desktop GUI for loading, filtering, visualizing, aligning, clustering, and exporting MINFLUX localization data from .npy files.

- Filter localizations by trace length, EFO, and CFR
- View data in interactive 2D/3D scatter plots
- Compare datasets in a multicolor viewer
- Generate heatmaps, Gaussian renderings, and TIFF exports
- Align datasets using MBM bead-based registration
- Run DBSCAN clustering on track-averaged localizations
- Export filtered data and summary statistics to CSV

Built with PySide6, Plotly, Matplotlib, NumPy, Pandas, SciPy, and Zarr.

## TODO 

- add more filtering options
- fix mbm bead plots

## Bugs pyflux

- sometimes opens second window
- when building with pyinstaller, scipy is not detected even though it is installed in .venv, see (workaround)[https://stackoverflow.com/questions/49559770/how-do-you-resolve-hidden-imports-not-found-warnings-in-pyinstaller-for-scipy?rq=1]

## License

[MIT](LICENSE)

## Contact

[SciLifeLab BioImage Informatics Unit](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Maximilian Senftleben](mailto:maximilian.senftleben@scilifelab.se)
