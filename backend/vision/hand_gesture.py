import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


class HandGestureDetector:
    """Detect an open hand or closed fist using MediaPipe Hand Landmarker."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(self.model_path):
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _distance(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    @classmethod
    def _finger_is_extended(cls, landmarks, mcp, pip, tip):
        # A finger is extended when its tip is farther from the wrist than
        # its PIP joint. This works regardless of whether the hand is rotated.
        wrist = landmarks[0]
        tip_distance = cls._distance(landmarks[tip], wrist)
        pip_distance = cls._distance(landmarks[pip], wrist)
        mcp_distance = cls._distance(landmarks[mcp], wrist)
        return tip_distance > pip_distance * 1.08 and tip_distance > mcp_distance * 1.18

    def detect_hand_box(self, frame_bgr):
        """Return the pixel bounding box of the first detected hand, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        pad = max(12, int(min(w, h) * 0.025))
        return (max(0, min(xs)-pad), max(0, min(ys)-pad), min(w-1, max(xs)+pad), min(h-1, max(ys)+pad))

    def _detect_landmarks(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def detect_hand_state(self, frame_bgr):
        """Return (box, is_fist, is_l_shape, hand_found) from one detection."""
        landmarks = self._detect_landmarks(frame_bgr)
        if landmarks is None:
            return None, False, False, False

        h, w = frame_bgr.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        pad = max(12, int(min(w, h) * 0.025))
        box = (max(0, min(xs) - pad), max(0, min(ys) - pad),
               min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad))

        fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
        extended = [self._finger_is_extended(landmarks, mcp, pip, tip)
                    for mcp, pip, tip in fingers]

        wrist = landmarks[0]
        thumb_extended = (
            self._distance(landmarks[4], wrist)
            > self._distance(landmarks[3], wrist) * 1.08
        )

        extended_count = sum(extended)
        is_fist = extended_count <= 1 and not thumb_extended
        is_l_shape = (thumb_extended and extended[0]
                      and not extended[1] and not extended[2] and not extended[3])
        return box, is_fist, is_l_shape, True

    def is_fist(self, frame_bgr):
        _, is_fist, _, hand_found = self.detect_hand_state(frame_bgr)
        return is_fist, hand_found
    def close(self):
        self.detector.close()
