import cv2
import numpy as np


class InvisibilityEffect:
    """Capture a clean background and replace the changed foreground with it."""

    def __init__(self):
        self.background = None

    def capture_background(self, frame_bgr):
        self.background = frame_bgr.copy()

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
