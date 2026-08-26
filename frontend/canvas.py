from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from frontend.floating_menu import FloatingMenu


class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.menu = FloatingMenu(self)
        self.menu.move(40, 40)

        self.video_label.lower()
        self.menu.raise_()

    def resizeEvent(self, event):
        self.video_label.setGeometry(self.rect())
        self.menu.raise_()

        super().resizeEvent(event)