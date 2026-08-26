import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QPainterPath
from PySide6.QtWidgets import QWidget


class VisibleFloatingMenu(QWidget):
    """Draggable circular optical lens over the live camera feed."""

    SIZE = 96
    REFRACTION_STRENGTH = 0.42
    EDGE_POWER = 2.4
    DYNAMIC_WAVE = 0.018
    DISPERSION_STRENGTH = 0.085
    BRIGHT_SOURCE_BOOST = 1.8
    RIM_WIDTH = 0.075
    RIM_STRENGTH = 0.24
    GLASS_TINT = 0.10
    GLASS_HAZE = 0.00
    GLASS_GLOSS = 0.12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dragging = False
        self.drag_offset = QPoint()
        self.frame = None
        self.refraction = None
        self.frame_index = 0

    def set_frame(self, frame):
        if frame is None:
            return
        self.frame = frame
        self.frame_index += 1
        self._build_refraction()

    def _build_refraction(self):
        if self.frame is None or self.parentWidget() is None:
            return

        frame = self.frame
        fh, fw = frame.shape[:2]
        canvas = self.parentWidget()
        cw = max(1, canvas.width())
        ch = max(1, canvas.height())

        scale = min(cw / fw, ch / fh)
        shown_w = fw * scale
        shown_h = fh * scale
        offset_x = (cw - shown_w) * 0.5
        offset_y = (ch - shown_h) * 0.5

        menu_cx = self.x() + self.SIZE * 0.5
        menu_cy = self.y() + self.SIZE * 0.5
        source_cx = (menu_cx - offset_x) / scale
        source_cy = (menu_cy - offset_y) / scale

        source_radius = (self.SIZE * 0.5) / scale
        side = max(32, int(source_radius * 3.0))
        half = side * 0.5
        x0 = int(source_cx - half)
        y0 = int(source_cy - half)

        padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_REFLECT_101)
        x0 += side
        y0 += side
        source = padded[y0:y0 + side, x0:x0 + side]
        if source.size == 0:
            return

        source = cv2.resize(source, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)

        h, w = source.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        dx = xx - cx
        dy = yy - cy
        radius_px = np.sqrt(dx * dx + dy * dy)
        radius = radius_px / (w * 0.5)
        inside = radius <= 1.0
        r = np.clip(radius, 0.0, 1.0)

        lens_curve = self.REFRACTION_STRENGTH * np.power(r, self.EDGE_POWER)
        phase = self.frame_index * 0.045
        wave = np.sin(dx * 0.11 + np.cos(dy * 0.075 + phase) * 1.7 + phase) * self.DYNAMIC_WAVE

        safe_r = np.maximum(radius_px, 0.001)
        nx = dx / safe_r
        ny = dy / safe_r
        base_shift = (lens_curve + wave * (1.0 - r)) * (w * 0.5)
        base_x = np.where(inside, xx - nx * base_shift, xx)
        base_y = np.where(inside, yy - ny * base_shift, yy)

        dispersion = self.DISPERSION_STRENGTH * np.power(r, 3.0) * (w * 0.5)
        red_x, red_y = base_x - nx * dispersion, base_y - ny * dispersion
        green_x, green_y = base_x, base_y
        blue_x, blue_y = base_x + nx * dispersion, base_y + ny * dispersion

        red = cv2.remap(source[:, :, 2], red_x.astype(np.float32), red_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        green = cv2.remap(source[:, :, 1], green_x.astype(np.float32), green_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        blue = cv2.remap(source[:, :, 0], blue_x.astype(np.float32), blue_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        refracted = np.dstack((blue, green, red)).astype(np.float32)

        luminance = (0.114 * source[:, :, 0] + 0.587 * source[:, :, 1] + 0.299 * source[:, :, 2]) / 255.0
        bright = np.clip((luminance - 0.55) / 0.45, 0.0, 1.0)
        bright_edge = bright * np.power(r, 2.2) * inside.astype(np.float32)
        boost = 1.0 + self.BRIGHT_SOURCE_BOOST * bright_edge
        refracted[:, :, 2] = np.clip(refracted[:, :, 2] + (red.astype(np.float32) - green.astype(np.float32)) * (boost - 1.0), 0, 255)
        refracted[:, :, 0] = np.clip(refracted[:, :, 0] + (blue.astype(np.float32) - green.astype(np.float32)) * (boost - 1.0), 0, 255)

        # Subtle Liquid Glass tint. Procedural haze is disabled.
        tint_strength = self.GLASS_TINT * (0.35 + 0.65 * np.power(1.0 - r, 1.6)) * inside.astype(np.float32)
        refracted = refracted * (1.0 - tint_strength[:, :, None]) + 255.0 * tint_strength[:, :, None]

        circle_alpha = np.clip((1.0 - radius) / 0.035, 0.0, 1.0) * inside.astype(np.float32)
        rim = np.exp(-((1.0 - r) / self.RIM_WIDTH) ** 2) * inside.astype(np.float32)
        light_x = np.cos(phase * 0.35) * 0.7
        light_y = np.sin(phase * 0.27) * 0.7
        radial_dot = nx * light_x + ny * light_y
        specular = np.clip((radial_dot + 1.0) * 0.5, 0.0, 1.0) * rim
        specular = np.power(specular, 7.0) * self.RIM_STRENGTH

        gloss_center = 0.5 + 0.25 * np.sin(phase * 0.25)
        gloss = np.exp(-((xx / w - gloss_center) ** 2) / 0.025)
        gloss *= np.exp(-((yy / h - 0.32) ** 2) / 0.16)
        gloss *= self.GLASS_GLOSS * inside.astype(np.float32)

        rgba = cv2.cvtColor(refracted.astype(np.uint8), cv2.COLOR_BGR2BGRA).astype(np.float32)
        rgba[:, :, 0] = np.clip(rgba[:, :, 0] + specular * 255.0 + gloss * 255.0, 0, 255)
        rgba[:, :, 1] = np.clip(rgba[:, :, 1] + specular * 255.0 + gloss * 255.0, 0, 255)
        rgba[:, :, 2] = np.clip(rgba[:, :, 2] + specular * 255.0 + gloss * 255.0, 0, 255)
        rgba[:, :, 3] = np.clip(np.maximum(circle_alpha, rim * 0.52) * 255.0, 0.0, 255.0)

        rgba = np.ascontiguousarray(rgba.astype(np.uint8))
        self.refraction = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)
        painter.setClipPath(path)
        if self.refraction is not None:
            painter.drawImage(0, 0, self.refraction)
        painter.setBrush(QColor(255, 255, 255, 7))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)
        painter.setClipping(False)
        icon = QPen(QColor(255, 255, 255, 250))
        icon.setWidth(5)
        icon.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(icon)
        x1, x2 = int(self.SIZE * 0.30), int(self.SIZE * 0.70)
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
