import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from backend.camera.camera import Camera
from backend.vision.hand_gesture import HandGestureDetector
from backend.vision.invisibility import InvisibilityEffect
from frontend.canvas import Canvas


class MainWindow(QMainWindow):
    BACKGROUND_CAPTURE_FRAMES = 90
    COUNTDOWN_SECONDS = 6

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
        self.countdown_frames = self.COUNTDOWN_SECONDS * 60

        self.countdown_label = QLabel(self.canvas)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("color: white; background: transparent;")
        self.countdown_label.setFont(QFont("Bodoni MT Condensed", 28))
        self.countdown_label.setGeometry(0, 35, self.canvas.width(), 55)
        self.countdown_label.raise_()
        self._update_countdown_label()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        self.showFullScreen()

    def _update_countdown_label(self):
        seconds = max(0, (self.countdown_frames + 59) // 60)
        if seconds > 0:
            self.countdown_label.setText(f"Please wait {seconds} seconds before entering the frame.")
            self.countdown_label.show()
        else:
            self.countdown_label.hide()

    def update_frame(self):
        frame = self.camera.read()
        frame = cv2.flip(frame, 1)

        if self.countdown_frames > 0:
            self.countdown_frames -= 1
            self._update_countdown_label()

        # Keep the existing Liquid Glass button untouched.
        self.canvas.menu.set_frame(frame)

        # Establish the clean background for the first ~1.5 seconds.
        # Step out of frame while Whitehall captures the scene.
        if not self.invisibility.ready():
            self.capture_count += 1
            self.invisibility.capture_background(frame)

        # DEBUG: show hand detection and fist state only. Do not trigger invisibility.
        hand_box = self.gesture.detect_hand_box(frame)
        fist, hand_found = self.gesture.is_fist(frame)
        if hand_box is not None:
            x1, y1, x2, y2 = hand_box
            box_color = (0, 0, 255) if fist else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        fist = False if not hand_found else fist

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

        output = frame
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
