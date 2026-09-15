import cv2
import numpy as np


class InvisibilityEffect:
    """Capture a clean background and replace the changed foreground with it."""

    def __init__(self):
        self.background = None
        self.body_box = None

    def capture_background(self, frame_bgr):
        self.background = frame_bgr.copy()
        self.body_box = None

    def detect_foreground_box(self, frame_bgr):
        """Return a bounding box around the main person foreground."""
        if self.background is None or self.background.shape != frame_bgr.shape:
            return None
        diff = cv2.absdiff(frame_bgr, self.background)
        change = diff.max(axis=2)
        mask = cv2.threshold(change, 28, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 1200]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        frame_h, frame_w = frame_bgr.shape[:2]
        if w < frame_w * 0.08 or h < frame_h * 0.12:
            return None
        pad_x = max(10, int(w * 0.025))
        pad_y = max(10, int(h * 0.025))
        detected = (max(0, x-pad_x), max(0, y-pad_y), min(frame_w-1, x+w+pad_x), min(frame_h-1, y+h+pad_y))
        if self.body_box is None:
            self.body_box = tuple(float(v) for v in detected)
        else:
            # Low-pass the body box so it does not jump with clothing/camera noise.
            self.body_box = tuple(0.20 * d + 0.80 * old for d, old in zip(detected, self.body_box))
        return tuple(int(v) for v in self.body_box)

    def ready(self):
        return self.background is not None

    def apply(self, frame_bgr, invisible):
        if not invisible or self.background is None:
            return frame_bgr

        background = self.background
        if background.shape != frame_bgr.shape:
            background = cv2.resize(
                background,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Build a robust foreground mask from both brightness and color change.
        diff = cv2.absdiff(frame_bgr, background)
        color_diff = diff.max(axis=2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        change = np.maximum(color_diff, gray_diff)

        mask = cv2.threshold(change, 32, 255, cv2.THRESH_BINARY)[1]

        # Remove camera noise, bridge small gaps in the silhouette, then fill
        # the remaining foreground contours so clothing doesn't become a sieve.
        small_kernel = np.ones((3, 3), np.uint8)
        large_kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= 250:
                cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

        # Feather only the final edge, preserving a clean disappearance without
        # the hard cutout/halo produced by a raw binary mask.
        alpha = cv2.GaussianBlur(filled, (15, 15), 0).astype(np.float32) / 255.0
        alpha = alpha[:, :, None]

        result = (
            frame_bgr.astype(np.float32) * (1.0 - alpha)
            + background.astype(np.float32) * alpha
        )
        return np.clip(result, 0, 255).astype(np.uint8)
