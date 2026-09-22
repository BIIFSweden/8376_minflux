import os
import sys
from pathlib import Path

# napari reaches Qt through qtpy, which must pick the binding pyflux uses.
# Set before any Qt import; an explicit QT_API in the environment still wins.
os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtGui, QtWidgets  # noqa: E402
from PySide6.QtWebEngineCore import QWebEngineProfile  # noqa: E402

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyflux.gui import MainWindow, apply_gui_theme, handle_download_requested  # noqa: E402


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    apply_gui_theme(app)

    profile = QWebEngineProfile.defaultProfile()
    profile.downloadRequested.connect(handle_download_requested)

    w = MainWindow()
    w.show()

    QtWidgets.QApplication.processEvents()  # ensure window is realized

    # pick the screen where the window currently is
    screen = w.screen()
    if screen is None:
        screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
    if screen is None:
        screen = QtWidgets.QApplication.primaryScreen()

    if screen is not None:
        g = screen.availableGeometry()
        if g.width() < 1500 or g.height() < 950:
            w.showMaximized()
        else:
            w.resize(1500, 950)
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
