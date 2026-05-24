"""小型可复用 Qt 组件。"""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class CameraView(QGroupBox):
    """单路 RGB 相机的带边框实时预览组件。"""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        rotate_preview_180: bool = False,
    ) -> None:
        super().__init__(title, parent)
        self.rotate_preview_180 = rotate_preview_180
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 300)
        self.image_label.setStyleSheet(
            "QLabel { background: #151515; color: #d8d8d8; border: 1px solid #333; }"
        )
        self.status_label = QLabel("Waiting")
        self.status_label.setAlignment(Qt.AlignCenter)
        self._image: QImage | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.status_label)

    def set_status(self, text: str) -> None:
        """更新预览图像下方的相机状态文字。"""

        self.status_label.setText(text)

    def set_image(self, image: QImage) -> None:
        """保存新图像，并按当前组件尺寸重绘。"""

        self._image = image
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        if self.rotate_preview_180:
            pixmap = pixmap.transformed(QTransform().rotate(180), Qt.SmoothTransformation)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


class JoystickWidget(QWidget):
    """鼠标拖拽式二维遥感控件，输出归一化 XY 方向。"""

    direction_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setMouseTracking(True)
        self._direction = QPointF(0.0, 0.0)
        self._dragging = False

    def direction(self) -> tuple[float, float]:
        """返回当前归一化方向，范围约为 [-1, 1]。"""

        return self._direction.x(), self._direction.y()

    def reset(self) -> None:
        """把遥感杆回中。"""

        self._dragging = False
        self._set_direction(0.0, 0.0)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._set_from_position(event.pos())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        if self._dragging:
            self._set_from_position(event.pos())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._set_direction(0.0, 0.0)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        if not self._dragging:
            self._set_direction(0.0, 0.0)
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        size = min(self.width(), self.height())
        radius = size * 0.38
        knob_radius = max(10.0, size * 0.09)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        knob = QPointF(
            center.x() - self._direction.y() * radius,
            center.y() - self._direction.x() * radius,
        )

        painter.setPen(QPen(QColor("#666666"), 2))
        painter.setBrush(QColor("#202020"))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawLine(
            QPointF(center.x(), center.y() - radius),
            QPointF(center.x(), center.y() + radius),
        )
        painter.drawLine(
            QPointF(center.x() - radius, center.y()),
            QPointF(center.x() + radius, center.y()),
        )

        painter.setPen(QPen(QColor("#d8d8d8"), 2))
        painter.setBrush(QColor("#4f8cff"))
        painter.drawEllipse(knob, knob_radius, knob_radius)

    def _set_from_position(self, pos) -> None:
        size = min(self.width(), self.height())
        radius = max(1.0, size * 0.38)
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        dx_screen = (pos.x() - center_x) / radius
        dy_screen = (pos.y() - center_y) / radius
        x = -dy_screen
        y = -dx_screen
        norm = math.hypot(x, y)
        if norm > 1.0:
            x /= norm
            y /= norm
        self._set_direction(x, y)

    def _set_direction(self, x: float, y: float) -> None:
        self._direction = QPointF(float(x), float(y))
        self.direction_changed.emit(float(x), float(y))
        self.update()
