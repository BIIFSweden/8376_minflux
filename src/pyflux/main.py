import sys
from pathlib import Path

from PySide6 import QtGui, QtWidgets
from PySide6.QtWebEngineCore import QWebEngineProfile

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyflux.gui import MainWindow, apply_gui_theme, handle_download_requested


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
