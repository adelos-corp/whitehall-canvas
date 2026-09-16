import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


class HandGestureDetector:
    """Stable hand-state detector for Whitehall Canvas.

    Uses MediaPipe Hand Landmarker landmarks, but classifies gestures from
    joint geometry plus temporal hysteresis so single-frame landmark jitter
    cannot switch the canvas state.
    """

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
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.75,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.locked_center = None
        self.locked_handedness = None
        self.lock_alpha = 0.18

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(
            np.array([a.x, a.y, a.z], dtype=np.float32) -
            np.array([b.x, b.y, b.z], dtype=np.float32)
        ))

    @staticmethod
    def _angle(a, b, c):
        """Angle ABC in degrees."""
        va = np.array([a.x-b.x, a.y-b.y, a.z-b.z], dtype=np.float32)
        vc = np.array([c.x-b.x, c.y-b.y, c.z-b.z], dtype=np.float32)
        den = float(np.linalg.norm(va) * np.linalg.norm(vc))
        if den < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(va, vc) / den, -1.0, 1.0))))

    @classmethod
    def _finger_extended(cls, lm, mcp, pip, dip, tip):
        # Extension requires both joints to be reasonably straight and the tip
        # to be farther from the wrist than the intermediate joints.
        wrist = lm[0]
        straight = (
            cls._angle(lm[mcp], lm[pip], lm[dip]) > 150.0
            and cls._angle(lm[pip], lm[dip], lm[tip]) > 145.0
        )
        radial = (
            cls._distance(lm[tip], wrist) >
            cls._distance(lm[pip], wrist) * 1.10
        )
        return straight and radial

    @classmethod
    def _finger_curled(cls, lm, mcp, pip, dip, tip):
        wrist = lm[0]
        bent = (
            cls._angle(lm[mcp], lm[pip], lm[dip]) < 145.0
            or cls._angle(lm[pip], lm[dip], lm[tip]) < 140.0
        )
        close = cls._distance(lm[tip], wrist) < cls._distance(lm[mcp], wrist) * 1.35
        return bent and close

    @classmethod
    def _thumb_extended(cls, lm):
        # Thumb needs both a straight thumb and meaningful separation from the
        # index MCP. This rejects many half-closed hands that resemble an L.
        straight = cls._angle(lm[2], lm[3], lm[4]) > 145.0
        reach = cls._distance(lm[4], lm[1]) > cls._distance(lm[3], lm[1]) * 1.12
        spread = cls._distance(lm[4], lm[5]) > 0.32 * cls._distance(lm[0], lm[9])
        return straight and reach and spread

    @classmethod
    def _classify(cls, lm):
        index = cls._finger_extended(lm, 5, 6, 7, 8)
        middle = cls._finger_extended(lm, 9, 10, 11, 12)
        ring = cls._finger_extended(lm, 13, 14, 15, 16)
        pinky = cls._finger_extended(lm, 17, 18, 19, 20)
        curled = [
            cls._finger_curled(lm, 5, 6, 7, 8),
            cls._finger_curled(lm, 9, 10, 11, 12),
            cls._finger_curled(lm, 13, 14, 15, 16),
            cls._finger_curled(lm, 17, 18, 19, 20),
        ]
        thumb = cls._thumb_extended(lm)

        # L requires exactly index + thumb extended, with all other fingers
        # demonstrably curled. A loose "two fingers up" rule caused false exits.
        is_l = (
            thumb and index and
            all(curled[1:]) and
            not middle and not ring and not pinky and
            cls._distance(lm[4], lm[8]) > 0.45 * cls._distance(lm[0], lm[9])
        )

        # Fist requires every finger to be curled and the thumb to be folded.
        thumb_folded = (
            cls._distance(lm[4], lm[0]) < cls._distance(lm[2], lm[0]) * 1.55
            and cls._angle(lm[1], lm[2], lm[3]) < 150.0
        )
        is_fist = all(curled) and thumb_folded

        is_open = (
            thumb and index and middle and ring and pinky
        )
        return is_open, is_fist, is_l

    def _detect(self, frame_bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        return result.hand_landmarks, result.handedness

    def detect_hand_state(self, frame_bgr, locked_center=None):
        all_hands, handedness = self._detect(frame_bgr)
        if not all_hands:
            return None, False, False, False, locked_center

        h, w = frame_bgr.shape[:2]

        def center(lm):
            return (
                sum(p.x for p in lm) / len(lm) * w,
                sum(p.y for p in lm) / len(lm) * h,
            )

        centers = [center(hand) for hand in all_hands]

        if locked_center is None:
            # Lock onto the physically largest/closest hand.
            areas = []
            for hand in all_hands:
                xs = [p.x * w for p in hand]
                ys = [p.y * h for p in hand]
                areas.append((max(xs)-min(xs)) * (max(ys)-min(ys)))
            index = int(np.argmax(areas))
        else:
            distances = [
                (cx-locked_center[0]) ** 2 + (cy-locked_center[1]) ** 2
                for cx, cy in centers
            ]
            index = int(np.argmin(distances))
            max_jump = (min(w, h) * 0.22) ** 2
            if distances[index] > max_jump:
                return None, False, False, False, locked_center

            # If handedness is known, do not silently swap left/right hands.
            if self.locked_handedness is not None and handedness:
                label = handedness[index][0].category_name
                if label != self.locked_handedness:
                    candidates = [
                        i for i, hnd in enumerate(handedness)
                        if hnd and hnd[0].category_name == self.locked_handedness
                    ]
                    if candidates:
                        index = min(candidates, key=lambda i:
                            (centers[i][0]-locked_center[0])**2 +
                            (centers[i][1]-locked_center[1])**2)
                    else:
                        return None, False, False, False, locked_center

        lm = all_hands[index]
        raw_center = centers[index]
        if locked_center is None:
            selected_center = raw_center
            if handedness and handedness[index]:
                self.locked_handedness = handedness[index][0].category_name
        else:
            selected_center = (
                locked_center[0] * (1.0-self.lock_alpha) + raw_center[0] * self.lock_alpha,
                locked_center[1] * (1.0-self.lock_alpha) + raw_center[1] * self.lock_alpha,
            )

        self.locked_center = selected_center

        xs = [int(p.x*w) for p in lm]
        ys = [int(p.y*h) for p in lm]
        pad = max(14, int(min(w, h) * 0.03))
        box = (
            max(0, min(xs)-pad), max(0, min(ys)-pad),
            min(w-1, max(xs)+pad), min(h-1, max(ys)+pad)
        )

        is_open, is_fist, is_l = self._classify(lm)
        return box, is_fist, is_l, True, selected_center

    def detect_hand_box(self, frame_bgr):
        box, _, _, found, _ = self.detect_hand_state(frame_bgr, self.locked_center)
        return box if found else None

    def is_fist(self, frame_bgr):
        _, is_fist, _, found, _ = self.detect_hand_state(frame_bgr, self.locked_center)
        return is_fist, found

    def close(self):
        self.detector.close()
