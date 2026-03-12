from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def create_app():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("TPV_Golf")
    app.setOrganizationName("TPV")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    return app, window
