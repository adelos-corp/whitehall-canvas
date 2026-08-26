from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class VisibleFloatingMenu(QWidget):
    SIZE = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dragging = False
        self.drag_offset = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(255, 255, 255, 65))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)
        rim = QPen(QColor(255, 255, 255, 230))
        rim.setWidth(2)
        painter.setPen(rim)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)
        inner = QPen(QColor(255, 255, 255, 90))
        inner.setWidth(1)
        painter.setPen(inner)
        painter.drawEllipse(7, 7, self.SIZE - 14, self.SIZE - 14)
        icon = QPen(QColor(255, 255, 255, 255))
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
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()
