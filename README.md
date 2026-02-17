# Pyflux

App for analyzing MINFLUX data.

## TODO 

- when building with pyinstaller, scipy is not detected even though it is installed in .venv, see (workaround)[https://stackoverflow.com/questions/49559770/how-do-you-resolve-hidden-imports-not-found-warnings-in-pyinstaller-for-scipy?rq=1]
- Problem with kaleido/chrome, also only an issue with built app, 
- check out resolution of exported imgs
 
## Bugs pyflux

- Save as SVG/PDF does NOT work in built app on arm64, error message:
```
'The browser seemed to close immediately after starting.', 'You can set the `logging.Logger` level lower to see more output.', 'You may try installing a known working copy of Chrome by running ', '`$ choreo_get_chrome`.It may be your browser auto-updated and will now work upon restart. The browser we tried to start is located at /Applications/Google Chrome.app/Contents/MacOS/Google Chrome.'
```
- js: Canvas2D: Multiple readback operations using getImageData are faster with the willReadFrequently attribute set to true. See: https://html.spec.whatwg.org/multipage/canvas.html#concept-canvas-will-read-frequently
- js: Uncaught TypeError: Cannot read properties of undefined (reading 'setViewport') sometimes shows up when zooming
- 2026-02-12 11:28:57.603 python[36522:16515968] The class 'NSOpenPanel' overrides the method identifier.  This method is implemented by class 'NSWindow'
- 2026-02-12 11:38:57.928 python[36522:16515968] TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1) to send to port com.apple.tsm.uiserver
- /Users/maximiliansenftleben/miniconda3/envs/8376_flux/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '

## License

[MIT](LICENSE)

## Contact

[SciLifeLab BioImage Informatics Unit](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Maximilian Senftleben](mailto:maximilian.senftleben@scilifelab.se)
