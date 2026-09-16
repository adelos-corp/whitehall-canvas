import sys

import cv2
from PySide6.QtCore import QTimer, Qt, QElapsedTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from backend.camera.camera import Camera
from backend.vision.hand_gesture import HandGestureDetector
from backend.vision.invisibility import InvisibilityEffect
from frontend.canvas import Canvas


class MainWindow(QMainWindow):
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
        self.background_captured = False
        self.fist_frames = 0
        self.open_frames = 0
        self.invisible = False
        self.locked_hand_center = None
        self.hand_lost_frames = 0
        self.l_frames = 0
        self.countdown_timer = QElapsedTimer()
        self.countdown_timer.start()

        self.countdown_label = QLabel(self.canvas)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("color: white; background: transparent;")
        self.countdown_label.setFont(QFont("SF Pro Display", 55, QFont.Weight.Bold))
        self.countdown_label.setGeometry(0, 28, self.canvas.width(), 82)
        self.countdown_label.raise_()
        self._update_countdown_label()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        self.showFullScreen()

    def _update_countdown_label(self):
        elapsed_ms = self.countdown_timer.elapsed()
        remaining_ms = max(0, self.COUNTDOWN_SECONDS * 1000 - elapsed_ms)
        seconds = (remaining_ms + 999) // 1000
        if remaining_ms > 0:
            self.countdown_label.setText(f"Please wait {seconds} seconds before entering the frame.")
            self.countdown_label.show()
        else:
            self.countdown_label.hide()

    def update_frame(self):
        frame = self.camera.read()
        frame = cv2.flip(frame, 1)

        if self.countdown_label.isVisible():
            self._update_countdown_label()

        # Keep the existing Liquid Glass button untouched.
        # Menu button temporarily hidden for the demo flow.
        self.canvas.menu.hide()

        # During the six-second startup countdown, continuously refresh the
        # background snapshot. The final frame captured before the countdown
        # ends becomes the clean scene for the invisibility effect.
        if not self.background_captured:
            elapsed_ms = self.countdown_timer.elapsed()
            self.invisibility.capture_background(frame)
            if elapsed_ms >= self.COUNTDOWN_SECONDS * 1000:
                self.background_captured = True

        hand_box, fist, l_shape, hand_found, selected_center = self.gesture.detect_hand_state(frame, self.locked_hand_center)
        # L is deliberately gated over consecutive frames. A single noisy
        # landmark frame must never terminate a demo.
        if hand_found and l_shape:
            self.l_frames += 1
        else:
            self.l_frames = max(0, self.l_frames - 1)
        stable_l = self.l_frames >= 5

        if stable_l:
            # Show the orange L state for one rendered frame before exiting.
            if hand_box is not None:
                x1, y1, x2, y2 = hand_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(image).scaled(self.canvas.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.canvas.video_label.setPixmap(pixmap)
            QTimer.singleShot(350, self.close)
            return
        if hand_found:
            self.locked_hand_center = selected_center
            self.hand_lost_frames = 0
        elif self.background_captured:
            self.hand_lost_frames += 1
            if self.hand_lost_frames > 30:
                self.locked_hand_center = None
                self.hand_lost_frames = 0
        if hand_box is not None:
            x1, y1, x2, y2 = hand_box
            if stable_l:
                box_color = (0, 165, 255)  # Orange
            elif fist:
                box_color = (0, 0, 255)    # Red
            else:
                box_color = (0, 255, 0)    # Green
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

        if self.fist_frames >= 6:
            self.invisible = True
        elif self.open_frames >= 5:
            self.invisible = False

        if not self.background_captured:
            self.invisible = False

        output = self.invisibility.apply(frame, self.invisible) if self.background_captured else frame
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "countdown_label"):
            self.countdown_label.setGeometry(0, 28, self.canvas.width(), 82)
            self.countdown_label.raise_()

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
