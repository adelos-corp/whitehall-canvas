import os
import urllib.request

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


class HandGestureDetector:
    """Detect whether the visible hand is open or closed using MediaPipe Tasks."""

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
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _finger_extended(landmarks, tip, pip):
        return landmarks[tip].y < landmarks[pip].y - 0.02

    def is_fist(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return False, False

        landmarks = result.hand_landmarks[0]
        extended = sum(
            self._finger_extended(landmarks, tip, pip)
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

        return extended <= 1, True

    def close(self):
        self.detector.close()
