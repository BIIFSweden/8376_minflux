# 
Collaboration with ALM, discussion about support for 3D light-sheet and MINFLUX data

Meeting on 2025-10-17 with the Advanced Light Microscopy facility at SciLifeLab, discussed that there is a need for future image analysis support for data that ALM generates 

[Download code](../../archive/refs/heads/main.zip)

## Installation

- Clone this repository
- Install environment with conda env create -f environment.yml

## ToDos

## Bugs pyflux

- js: Canvas2D: Multiple readback operations using getImageData are faster with the willReadFrequently attribute set to true. See: https://html.spec.whatwg.org/multipage/canvas.html#concept-canvas-will-read-frequently
- js: Uncaught TypeError: Cannot read properties of undefined (reading 'setViewport') sometimes shows up when zooming
- 2026-02-12 11:28:57.603 python[36522:16515968] The class 'NSOpenPanel' overrides the method identifier.  This method is implemented by class 'NSWindow'
- 2026-02-12 11:38:57.928 python[36522:16515968] TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1) to send to port com.apple.tsm.uiserver

## License

[MIT](LICENSE)

## Contact

[SciLifeLab BioImage Informatics Unit](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Maximilian Senftleben](mailto:maximilian.senftleben@scilifelab.se)
