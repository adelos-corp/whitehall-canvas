import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from backend.camera.camera import Camera
from backend.vision.hand_gesture import HandGestureDetector
from backend.vision.invisibility import InvisibilityEffect
from frontend.canvas import Canvas


class MainWindow(QMainWindow):
    BACKGROUND_CAPTURE_FRAMES = 90

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whitehall Canvas")
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        self.camera = Camera()
        self.camera.start()
        self.gesture = HandGestureDetector()
        self.invisibility = InvisibilityEffect()
        self.capture_count = 0
        self.fist_frames = 0
        self.open_frames = 0
        self.invisible = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        self.showFullScreen()

    def update_frame(self):
        frame = self.camera.read()
        frame = cv2.flip(frame, 1)

        # Keep the existing Liquid Glass button untouched.
        self.canvas.menu.set_frame(frame)

        # Establish the clean background for the first ~1.5 seconds.
        # Step out of frame while Whitehall captures the scene.
        if not self.invisibility.ready():
            self.capture_count += 1
            self.invisibility.capture_background(frame)

        fist, hand_found = self.gesture.is_fist(frame)

        # Require consecutive frames so landmark jitter does not flicker the
        # effect on and off.
        if hand_found and fist:
            self.fist_frames += 1
            self.open_frames = 0
        elif hand_found:
            self.open_frames += 1
            self.fist_frames = 0
        else:
            self.fist_frames = max(0, self.fist_frames - 1)
            self.open_frames = max(0, self.open_frames - 1)

        if self.fist_frames >= 4:
            self.invisible = True
        elif self.open_frames >= 3:
            self.invisible = False

        if self.capture_count < self.BACKGROUND_CAPTURE_FRAMES:
            self.invisible = False

        output = self.invisibility.apply(frame, self.invisible)
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width

        image = QImage(
            rgb.data,
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
        self.gesture.close()
        self.camera.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
