import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import QWidget


class FloatingMenu(QWidget):
    SIZE = 96

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(self.SIZE, self.SIZE)

        self.dragging = False
        self.drag_offset = QPoint()

        self.frame = None
        self.refracted_image = None

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

        self.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

    def set_frame(self, frame):
        """
        Receive the current mirrored BGR camera frame.
        """

        self.frame = frame

        if frame is None:
            return

        self.update_refraction()

    def update_refraction(self):
        """
        Generate a refracted section of the live camera frame
        corresponding to the menu's current screen position.
        """

        if self.frame is None:
            return

        frame = self.frame

        frame_height, frame_width = frame.shape[:2]

        # -------------------------------------------------
        # Determine where the camera image is displayed
        # inside the Canvas.
        # -------------------------------------------------

        if self.parent() is None:
            return

        canvas_width = self.parent().width()
        canvas_height = self.parent().height()

        scale = min(
            canvas_width / frame_width,
            canvas_height / frame_height
        )

        displayed_width = int(frame_width * scale)
        displayed_height = int(frame_height * scale)

        offset_x = (canvas_width - displayed_width) / 2
        offset_y = (canvas_height - displayed_height) / 2

        # Menu center in Canvas coordinates
        menu_center_x = self.x() + self.SIZE / 2
        menu_center_y = self.y() + self.SIZE / 2

        # Convert Canvas coordinates → camera coordinates
        source_center_x = (
            menu_center_x - offset_x
        ) / scale

        source_center_y = (
            menu_center_y - offset_y
        ) / scale

        # Size of source region required
        source_radius = int(
            (self.SIZE / 2) / scale
        )

        crop_size = source_radius * 2 + 8

        x1 = int(source_center_x - crop_size / 2)
        y1 = int(source_center_y - crop_size / 2)

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # -------------------------------------------------
        # Pad frame if the menu reaches an edge.
        # -------------------------------------------------

        padded = cv2.copyMakeBorder(
            frame,
            crop_size,
            crop_size,
            crop_size,
            crop_size,
            cv2.BORDER_REFLECT
        )

        x1 += crop_size
        x2 += crop_size
        y1 += crop_size
        y2 += crop_size

        crop = padded[y1:y2, x1:x2]

        if crop.size == 0:
            return

        # -------------------------------------------------
        # Resize source region to menu resolution.
        # -------------------------------------------------

        crop = cv2.resize(
            crop,
            (self.SIZE, self.SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # -------------------------------------------------
        # REAL RADIAL REFRACTION
        # -------------------------------------------------

        h, w = crop.shape[:2]

        yy, xx = np.mgrid[0:h, 0:w]

        cx = (w - 1) / 2
        cy = (h - 1) / 2

        dx = xx - cx
        dy = yy - cy

        radius = np.sqrt(
            dx * dx + dy * dy
        )

        normalized_radius = radius / (w / 2)

        # Glass refraction strength
        distortion = (
            1.0
            + 0.16 * np.maximum(
                0,
                1 - normalized_radius
            )
        )

        map_x = cx + dx / distortion
        map_y = cy + dy / distortion

        refracted = cv2.remap(
            crop,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # -------------------------------------------------
        # Circular mask
        # -------------------------------------------------

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.circle(
            mask,
            (int(cx), int(cy)),
            int(w / 2 - 4),
            255,
            -1
        )

        # Transparent outside the glass
        bgra = cv2.cvtColor(
            refracted,
            cv2.COLOR_BGR2BGRA
        )

        bgra[:, :, 3] = mask

        # -------------------------------------------------
        # Convert to QImage
        # -------------------------------------------------

        bgra = np.ascontiguousarray(bgra)

        self.refracted_image = QImage(
            bgra.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_ARGB32
        ).copy()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        # -------------------------------------------------
        # LIVE REFRACTED CAMERA IMAGE
        # -------------------------------------------------

        if self.refracted_image is not None:
            painter.drawImage(
                0,
                0,
                self.refracted_image
            )

        # -------------------------------------------------
        # GLASS TINT
        # -------------------------------------------------

        painter.setBrush(
            QBrush(
                QColor(
                    255,
                    255,
                    255,
                    18
                )
            )
        )

        painter.setPen(
            Qt.PenStyle.NoPen
        )

        painter.drawEllipse(
            4,
            4,
            self.SIZE - 8,
            self.SIZE - 8
        )

        # -------------------------------------------------
        # GLASS OUTER EDGE
        # -------------------------------------------------

        outer_pen = QPen(
            QColor(
                255,
                255,
                255,
                190
            )
        )

        outer_pen.setWidth(2)

        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawEllipse(
            4,
            4,
            self.SIZE - 8,
            self.SIZE - 8
        )

        # -------------------------------------------------
        # INNER EDGE
        # -------------------------------------------------

        inner_pen = QPen(
            QColor(
                255,
                255,
                255,
                70
            )
        )

        inner_pen.setWidth(1)

        painter.setPen(inner_pen)

        painter.drawEllipse(
            7,
            7,
            self.SIZE - 14,
            self.SIZE - 14
        )

        # -------------------------------------------------
        # THREE-LINE MENU ICON
        # -------------------------------------------------

        icon_pen = QPen(
            QColor(
                255,
                255,
                255,
                245
            )
        )

        icon_pen.setWidth(5)

        icon_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )

        painter.setPen(icon_pen)

        x1 = int(self.SIZE * 0.30)
        x2 = int(self.SIZE * 0.70)

        y1 = int(self.SIZE * 0.36)
        y2 = int(self.SIZE * 0.50)
        y3 = int(self.SIZE * 0.64)

        painter.drawLine(
            x1, y1,
            x2, y1
        )

        painter.drawLine(
            x1, y2,
            x2, y2
        )

        painter.drawLine(
            x1, y3,
            x2, y3
        )

        painter.end()

    # -----------------------------------------------------
    # DRAGGING
    # -----------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:

            self.dragging = True

            self.drag_offset = (
                event.position().toPoint()
            )

            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:

            new_position = (
                self.pos()
                + event.position().toPoint()
                - self.drag_offset
            )

            self.move(new_position)

            # Recalculate refraction immediately
            self.update_refraction()

            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:

            self.dragging = False

            event.accept()