import cv2
import struct

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
    QOpenGLTexture,
)


class FloatingMenu(QOpenGLWidget):
    """Native Qt GPU-rendered Liquid Glass menu."""

    SIZE = 96

    VERTEX_SHADER = """
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 vTexCoord;

        void main()
        {
            vTexCoord = texCoord;
            gl_Position = vec4(position, 0.0, 1.0);
        }
    """

    FRAGMENT_SHADER = """
        varying vec2 vTexCoord;
        uniform sampler2D cameraTexture;
        uniform vec2 centerUV;
        uniform vec2 radiusUV;
        uniform float strength;
        uniform float time;

        float hash(vec2 p)
        {
            p = fract(p * vec2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }

        float noise(vec2 p)
        {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));

            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        float turbulence(vec2 p)
        {
            float value = 0.0;
            float amplitude = 0.5;

            for (int i = 0; i < 3; ++i)
            {
                value += noise(p) * amplitude;
                p *= 2.0;
                amplitude *= 0.5;
            }

            return value;
        }

        void main()
        {
            vec2 local = vTexCoord - vec2(0.5);
            float r = length(local) * 2.0;

            if (r > 1.0)
                discard;

            vec2 direction = normalize(local + vec2(0.00001));
            float edge = smoothstep(1.0, 0.05, r);

            float t = turbulence(local * 5.0 + vec2(time * 0.015));
            float wave = (t - 0.5) * 2.0;

            vec2 displacement = direction * wave * strength * edge;
            displacement += direction * (0.035 * (1.0 - r * r));

            vec2 uv = centerUV + (local * 2.0) * radiusUV
                    + displacement * radiusUV;

            vec3 refracted = texture2D(cameraTexture, uv).rgb;
            refracted *= 1.08;
            refracted += vec3(0.015);

            float innerRim = smoothstep(0.98, 0.88, r);
            vec3 glass = refracted + vec3(0.10) * innerRim;
            glass = mix(glass, vec3(1.0), 0.055);

            gl_FragColor = vec4(glass, 0.96);
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)

        self.dragging = False
        self.drag_offset = QPoint()
        self.frame = None
        self.texture = None
        self.program = None
        self.vbo = None
        self.vao = None
        self.time = 0.0

    def set_frame(self, frame):
        if frame is None:
            return

        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update()

    def initializeGL(self):
        self.gl = self.context().functions()
        self.gl.glEnable(self.gl.GL_BLEND)
        self.gl.glBlendFunc(self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        self.program = QOpenGLShaderProgram(self)
        self.program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex,
            self.VERTEX_SHADER,
        )
        self.program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            self.FRAGMENT_SHADER,
        )

        if not self.program.link():
            raise RuntimeError(self.program.log())

        vertices = struct.pack(
            "16f",
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        )

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(vertices)

        self.vao = QOpenGLVertexArrayObject(self)
        self.vao.create()
        self.vao.bind()

        stride = 16
        position_loc = self.program.attributeLocation("position")
        tex_loc = self.program.attributeLocation("texCoord")

        self.program.enableAttributeArray(position_loc)
        self.program.setAttributeBuffer(
            position_loc, self.gl.GL_FLOAT, 0, 2, stride
        )
        self.program.enableAttributeArray(tex_loc)
        self.program.setAttributeBuffer(
            tex_loc, self.gl.GL_FLOAT, 8, 2, stride
        )

        self.vao.release()
        self.vbo.release()

    def _upload_texture(self):
        if self.frame is None:
            return

        image = QImage(
            self.frame.data,
            self.frame.shape[1],
            self.frame.shape[0],
            self.frame.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

        if self.texture is None:
            self.texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            self.texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            self.texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            self.texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            self.texture.create()

        self.texture.setData(image)

    def paintGL(self):
        self.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

        if self.program is None or self.frame is None:
            self._paint_icon()
            return

        self._upload_texture()

        canvas = self.parentWidget()
        if canvas is None:
            return

        frame_h, frame_w = self.frame.shape[:2]
        canvas_w = max(1, canvas.width())
        canvas_h = max(1, canvas.height())

        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        displayed_w = frame_w * scale
        displayed_h = frame_h * scale
        offset_x = (canvas_w - displayed_w) * 0.5
        offset_y = (canvas_h - displayed_h) * 0.5

        menu_cx = self.x() + self.SIZE * 0.5
        menu_cy = self.y() + self.SIZE * 0.5

        center_u = (menu_cx - offset_x) / displayed_w
        center_v = (menu_cy - offset_y) / displayed_h
        radius_u = (self.SIZE * 0.5) / displayed_w
        radius_v = (self.SIZE * 0.5) / displayed_h

        self.program.bind()
        self.texture.bind(0)
        self.program.setUniformValue("cameraTexture", 0)
        self.program.setUniformValue("centerUV", float(center_u), float(center_v))
        self.program.setUniformValue("radiusUV", float(radius_u), float(radius_v))
        self.program.setUniformValue("strength", 0.16)
        self.program.setUniformValue("time", float(self.time))

        self.vao.bind()
        self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, 4)
        self.vao.release()

        self.texture.release()
        self.program.release()

        self._paint_icon()
        self.time += 0.016

    def _paint_icon(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(255, 255, 255, 190))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(4, 4, self.SIZE - 8, self.SIZE - 8)

        inner = QPen(QColor(255, 255, 255, 75))
        inner.setWidth(1)
        painter.setPen(inner)
        painter.drawEllipse(7, 7, self.SIZE - 14, self.SIZE - 14)

        icon = QPen(QColor(255, 255, 255, 245))
        icon.setWidth(5)
        icon.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(icon)

        x1 = int(self.SIZE * 0.30)
        x2 = int(self.SIZE * 0.70)
        for y in (0.36, 0.50, 0.64):
            yy = int(self.SIZE * y)
            painter.drawLine(x1, yy, x2, yy)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_offset = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            new_position = (
                self.pos()
                + event.position().toPoint()
                - self.drag_offset
            )
            self.move(new_position)
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()
