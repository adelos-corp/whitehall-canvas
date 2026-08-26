import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from backend.camera.camera import Camera
from frontend.canvas import Canvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Whitehall Canvas")

        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        self.camera = Camera()
        self.camera.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

        self.showFullScreen()

    def update_frame(self):
        frame = self.camera.read()

        # Mirror the front-facing camera so the canvas behaves like a mirror.
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for Qt.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame.shape
        bytes_per_line = channels * width

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        scaled_pixmap = pixmap.scaled(
            self.canvas.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.canvas.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        self.camera.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
