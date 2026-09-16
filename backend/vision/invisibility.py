import cv2
import numpy as np
import mediapipe as mp


class InvisibilityEffect:
    """Capture a clean background and replace the changed foreground with it."""

    def __init__(self):
        self.background = None
        self.body_box = None
        self._holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=True, smooth_segmentation=True, min_detection_confidence=0.60, min_tracking_confidence=0.65)

    def capture_background(self, frame_bgr):
        self.background = frame_bgr.copy()
        self.body_box = None

    def detect_foreground_box(self, frame_bgr):
        """Track the person using MediaPipe Holistic pose and segmentation."""
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        results = self._holistic.process(rgb)
        landmarks = results.pose_landmarks
        if landmarks is None:
            return None

        h, w = frame_bgr.shape[:2]
        points = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks.landmark
                  if lm.visibility >= 0.45 and 0 <= lm.x <= 1 and 0 <= lm.y <= 1]
        if len(points) < 6:
            return None

        xs, ys = zip(*points)
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)

        if results.segmentation_mask is not None:
            mask = (np.clip(results.segmentation_mask, 0, 1) > 0.55).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 0.015*w*h:
                    sx, sy, sw, sh = cv2.boundingRect(contour)
                    x1, y1 = min(x1, sx), min(y1, sy)
                    x2, y2 = max(x2, sx+sw), max(y2, sy+sh)

        pad_x, pad_y = max(12, int((x2-x1)*0.08)), max(12, int((y2-y1)*0.05))
        detected = (max(0,x1-pad_x), max(0,y1-pad_y), min(w-1,x2+pad_x), min(h-1,y2+pad_y))

        if self.body_box is None:
            self.body_box = tuple(float(v) for v in detected)
        else:
            self.body_box = tuple(0.16*d + 0.84*old for d, old in zip(detected, self.body_box))
        return tuple(int(v) for v in self.body_box)


    def close(self):
        self._holistic.close()

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
