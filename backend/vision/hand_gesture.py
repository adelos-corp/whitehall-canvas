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
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.65,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.locked_center = None
        self.lock_alpha = 0.22

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
        return result.hand_landmarks if result.hand_landmarks else []

    def detect_hand_state(self, frame_bgr, locked_center=None):
        """Return the state of the locked hand, ignoring other visible hands."""
        all_hands = self._detect_landmarks(frame_bgr)
        if not all_hands:
            return None, False, False, False, locked_center

        h, w = frame_bgr.shape[:2]

        def center(landmarks):
            return (
                sum(lm.x for lm in landmarks) / len(landmarks) * w,
                sum(lm.y for lm in landmarks) / len(landmarks) * h,
            )

        centers = [center(hand) for hand in all_hands]

        if locked_center is None:
            # Prefer the largest hand, reducing accidental locks on a distant
            # gesturing hand during the initial lock-on.
            areas = []
            for hand in all_hands:
                hx = [lm.x * w for lm in hand]
                hy = [lm.y * h for lm in hand]
                areas.append((max(hx)-min(hx)) * (max(hy)-min(hy)))
            index = max(range(len(areas)), key=areas.__getitem__)
        else:
            distances = [
                (cx - locked_center[0]) ** 2 + (cy - locked_center[1]) ** 2
                for cx, cy in centers
            ]
            index = min(range(len(distances)), key=distances.__getitem__)
            # Do not jump across the frame to a different person's/hand's hand.
            if distances[index] > (min(w, h) * 0.28) ** 2:
                return None, False, False, False, locked_center

        landmarks = all_hands[index]
        raw_center = centers[index]
        if locked_center is None:
            selected_center = raw_center
        else:
            selected_center = (
                locked_center[0] * (1.0 - self.lock_alpha) + raw_center[0] * self.lock_alpha,
                locked_center[1] * (1.0 - self.lock_alpha) + raw_center[1] * self.lock_alpha,
            )

        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        pad = max(12, int(min(w, h) * 0.025))
        cx, cy = selected_center
        raw_box = (max(0, min(xs) - pad), max(0, min(ys) - pad), min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad))
        if locked_center is not None and self.locked_center is not None:
            dx = cx - self.locked_center[0]
            dy = cy - self.locked_center[1]
            raw_box = (int(raw_box[0]-dx*.78), int(raw_box[1]-dy*.78), int(raw_box[2]-dx*.78), int(raw_box[3]-dy*.78))
        self.locked_center = selected_center
        box = tuple(max(0, min(v, limit)) for v, limit in zip(raw_box, (w-1, h-1, w-1, h-1)))

        fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
        extended = [
            self._finger_is_extended(landmarks, mcp, pip, tip)
            for mcp, pip, tip in fingers
        ]

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        thumb_extended = (
            self._distance(thumb_tip, wrist) > self._distance(thumb_ip, wrist) * 1.10
            and self._distance(thumb_tip, index_mcp) > self._distance(thumb_ip, index_mcp) * 1.12
        )

        index_extended = extended[0]
        other_fingers_curled = sum(extended[1:]) == 0
        is_fist = sum(extended) == 0 and not thumb_extended
        is_l_shape = thumb_extended and index_extended and other_fingers_curled

        return box, is_fist, is_l_shape, True, selected_center

    def is_fist(self, frame_bgr):
        _, is_fist, _, hand_found, _ = self.detect_hand_state(frame_bgr)
        return is_fist, hand_found

    def is_fist(self, frame_bgr):
        _, is_fist, _, hand_found = self.detect_hand_state(frame_bgr)
        return is_fist, hand_found
    def close(self):
        self.detector.close()
