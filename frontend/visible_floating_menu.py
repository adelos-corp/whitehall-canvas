import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class VisibleFloatingMenu(QWidget):
    """Native draggable glass control with live camera refraction."""

    SIZE = 96
    DISTORTION = 0.18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.dragging = False
        self.drag_offset = QPoint()
        self.frame = None
        self.refraction = None

    def set_frame(self, frame):
        self.frame = frame
        self._build_refraction()

    def _build_refraction(self):
        if self.frame is None or self.parent() is None:
            return

        frame = self.frame
        fh, fw = frame.shape[:2]
        cw = max(1, self.parent().width())
        ch = max(1, self.parent().height())

        # Match the KeepAspectRatio presentation used by MainWindow.
        scale = min(cw / fw, ch / fh)
        shown_w = fw * scale
        shown_h = fh * scale
        offset_x = (cw - shown_w) * 0.5
        offset_y = (ch - shown_h) * 0.5

        menu_cx = self.x() + self.SIZE * 0.5
        menu_cy = self.y() + self.SIZE * 0.5

        src_cx = (menu_cx - offset_x) / scale
        src_cy = (menu_cy - offset_y) / scale
        src_radius = (self.SIZE * 0.5) / scale

        # Grab a slightly larger source area so the lens has room to bend pixels.
        side = max(12, int(src_radius * 2.35))
        half = side / 2
        x0 = int(src_cx - half)
        y0 = int(src_cy - half)

        padded = cv2.copyMakeBorder(
            frame,
            side,
            side,
            side,
            side,
            cv2.BORDER_REFLECT_101,
        )
        x0 += side
        y0 += side
        source = padded[y0:y0 + side, x0:x0 + side]

        if source.size == 0:
            return

        source = cv2.resize(
            source,
            (self.SIZE, self.SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        h, w = source.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy) / (w * 0.5)

        # Barrel lens distortion: pixels near the glass edge are displaced
        # more strongly, producing a real-time refracted camera image.
        strength = np.clip(1.0 - r, 0.0, 1.0)
        factor = 1.0 + self.DISTORTION * strength * strength

        map_x = cx + dx / factor
        map_y = cy + dy / factor

        refracted = cv2.remap(
            source,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        rgba = cv2.cvtColor(refracted, cv2.COLOR_BGR2BGRA)

        # Circular alpha mask with a soft edge.
        yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
        distance = np.sqrt((xx2 - cx) ** 2 + (yy2 - cy) ** 2)
        outer = w * 0.5 - 2.0
        inner = w * 0.5 - 6.0
        alpha = np.clip((outer - distance) / max(1.0, outer - inner), 0.0, 1.0)
        alpha[distance <= inner] = 1.0
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

        rgba = np.ascontiguousarray(rgba)
        self.refraction = QImage(
            rgba.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_ARGB32,
        ).copy()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Live refracted camera pixels.
        if self.refraction is not None:
            painter.drawImage(0, 0, self.refraction)

        # Subtle glass tint.
        painter.setBrush(QColor(255, 255, 255, 20))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)

        # Bright glass rim.
        rim = QPen(QColor(255, 255, 255, 210))
        rim.setWidth(2)
        painter.setPen(rim)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)

        inner = QPen(QColor(255, 255, 255, 80))
        inner.setWidth(1)
        painter.setPen(inner)
        painter.drawEllipse(7, 7, self.SIZE - 14, self.SIZE - 14)

        # Three-line menu icon.
        icon = QPen(QColor(255, 255, 255, 250))
        icon.setWidth(5)
        icon.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(icon)

        x1 = int(self.SIZE * 0.30)
        x2 = int(self.SIZE * 0.70)
        for fraction in (0.36, 0.50, 0.64):
            y = int(self.SIZE * fraction)
            painter.drawLine(x1, y, x2, y)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_offset = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.pos() + event.position().toPoint() - self.drag_offset)
            self._build_refraction()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()
