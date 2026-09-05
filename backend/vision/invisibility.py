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

        # Compare against the clean scene. The moving person becomes the mask.
        diff = cv2.absdiff(frame_bgr, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 28, 255, cv2.THRESH_BINARY)[1]

        # Clean small camera noise while keeping the body silhouette intact.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        # Slightly expand the mask so edges don't leave a ghost outline.
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, None]

        result = (
            frame_bgr.astype(np.float32) * (1.0 - alpha)
            + background.astype(np.float32) * alpha
        )
        return np.clip(result, 0, 255).astype(np.uint8)
