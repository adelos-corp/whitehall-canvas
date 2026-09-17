import Foundation
import Vision
import cv2
import numpy as np


class HandGestureDetector:
    """Apple Vision hand-pose detector with deterministic gesture classification.

    Vision supplies the hand landmarks and confidence values. Whitehall Canvas
    keeps ownership of gesture semantics, so OPEN/FIST/L remain explicit and
    tunable instead of being delegated to a second opaque model.
    """

    JOINTS = {
        "wrist": Vision.VNHumanHandPoseObservationJointNameWrist,
        "thumb_cmc": Vision.VNHumanHandPoseObservationJointNameThumbCMC,
        "thumb_mp": Vision.VNHumanHandPoseObservationJointNameThumbMP,
        "thumb_ip": Vision.VNHumanHandPoseObservationJointNameThumbIP,
        "thumb_tip": Vision.VNHumanHandPoseObservationJointNameThumbTip,
        "index_mcp": Vision.VNHumanHandPoseObservationJointNameIndexMCP,
        "index_pip": Vision.VNHumanHandPoseObservationJointNameIndexPIP,
        "index_dip": Vision.VNHumanHandPoseObservationJointNameIndexDIP,
        "index_tip": Vision.VNHumanHandPoseObservationJointNameIndexTip,
        "middle_mcp": Vision.VNHumanHandPoseObservationJointNameMiddleMCP,
        "middle_pip": Vision.VNHumanHandPoseObservationJointNameMiddlePIP,
        "middle_dip": Vision.VNHumanHandPoseObservationJointNameMiddleDIP,
        "middle_tip": Vision.VNHumanHandPoseObservationJointNameMiddleTip,
        "ring_mcp": Vision.VNHumanHandPoseObservationJointNameRingMCP,
        "ring_pip": Vision.VNHumanHandPoseObservationJointNameRingPIP,
        "ring_dip": Vision.VNHumanHandPoseObservationJointNameRingDIP,
        "ring_tip": Vision.VNHumanHandPoseObservationJointNameRingTip,
        "little_mcp": Vision.VNHumanHandPoseObservationJointNameLittleMCP,
        "little_pip": Vision.VNHumanHandPoseObservationJointNameLittlePIP,
        "little_dip": Vision.VNHumanHandPoseObservationJointNameLittleDIP,
        "little_tip": Vision.VNHumanHandPoseObservationJointNameLittleTip,
    }

    def __init__(self):
        self.request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self.request.setMaximumHandCount_(2)
        self.locked_center = None
        self.locked_handedness = None
        self.lock_alpha = 0.18
        self.min_confidence = 0.35

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

    @staticmethod
    def _angle(a, b, c):
        va = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        vc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        den = float(np.linalg.norm(va) * np.linalg.norm(vc))
        if den < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(va, vc) / den, -1.0, 1.0))))

    @classmethod
    def _finger_extended(cls, lm, mcp, pip, dip, tip):
        wrist = lm["wrist"]
        straight = (
            cls._angle(lm[mcp], lm[pip], lm[dip]) > 150.0
            and cls._angle(lm[pip], lm[dip], lm[tip]) > 145.0
        )
        radial = cls._distance(lm[tip], wrist) > cls._distance(lm[pip], wrist) * 1.10
        return straight and radial

    @classmethod
    def _finger_curled(cls, lm, mcp, pip, dip, tip):
        wrist = lm["wrist"]
        bent = (
            cls._angle(lm[mcp], lm[pip], lm[dip]) < 145.0
            or cls._angle(lm[pip], lm[dip], lm[tip]) < 140.0
        )
        close = cls._distance(lm[tip], wrist) < cls._distance(lm[mcp], wrist) * 1.35
        return bent and close

    @classmethod
    def _classify(cls, lm):
        index = cls._finger_extended(lm, "index_mcp", "index_pip", "index_dip", "index_tip")
        middle = cls._finger_extended(lm, "middle_mcp", "middle_pip", "middle_dip", "middle_tip")
        ring = cls._finger_extended(lm, "ring_mcp", "ring_pip", "ring_dip", "ring_tip")
        little = cls._finger_extended(lm, "little_mcp", "little_pip", "little_dip", "little_tip")

        curled_index = cls._finger_curled(lm, "index_mcp", "index_pip", "index_dip", "index_tip")
        curled_middle = cls._finger_curled(lm, "middle_mcp", "middle_pip", "middle_dip", "middle_tip")
        curled_ring = cls._finger_curled(lm, "ring_mcp", "ring_pip", "ring_dip", "ring_tip")
        curled_little = cls._finger_curled(lm, "little_mcp", "little_pip", "little_dip", "little_tip")

        wrist = lm["wrist"]
        thumb_tip = lm["thumb_tip"]
        thumb_mp = lm["thumb_mp"]
        palm_size = cls._distance(wrist, lm["middle_mcp"])

        thumb_straight = cls._angle(lm["thumb_mp"], lm["thumb_ip"], lm["thumb_tip"]) > 145.0
        thumb_reach = cls._distance(thumb_tip, lm["thumb_cmc"]) > cls._distance(thumb_mp, lm["thumb_cmc"]) * 1.10
        thumb_spread = cls._distance(thumb_tip, lm["index_mcp"]) > palm_size * 0.32
        thumb_extended = thumb_straight and thumb_reach and thumb_spread

        thumb_to_palm = cls._distance(thumb_tip, lm["middle_mcp"])
        thumb_folded = thumb_to_palm < palm_size * 0.72 and cls._angle(lm["thumb_cmc"], lm["thumb_mp"], thumb_tip) < 145.0

        four_curled = all((curled_index, curled_middle, curled_ring, curled_little))
        four_not_extended = not any((index, middle, ring, little))
        is_fist = four_curled and four_not_extended and thumb_folded

        is_l = (
            thumb_extended
            and index
            and curled_middle
            and curled_ring
            and curled_little
            and not middle
            and not ring
            and not little
            and cls._distance(thumb_tip, lm["index_tip"]) > palm_size * 0.45
        )

        is_open = thumb_extended and index and middle and ring and little
        return is_open, is_fist, is_l

    def _request_points(self, frame_bgr):
        # Vision accepts image Data directly. JPEG is used only as the bridge
        # from OpenCV's NumPy frame to Apple's image-analysis API; no model or
        # external inference runtime is involved.
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return []
        data = Foundation.NSData.dataWithBytes_length_(encoded.tobytes(), int(encoded.nbytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        try:
            handler.performRequests_error_([self.request], None)
        except Exception:
            return []
        return self.request.results() or []

    def _observation_landmarks(self, observation):
        landmarks = {}
        confidences = []
        for name, joint in self.JOINTS.items():
            try:
                point, error = observation.recognizedPointForJointName_error_(joint, None)
            except Exception:
                return None
            if point is None:
                return None
            confidence = float(point.confidence())
            if confidence < self.min_confidence:
                return None
            location = point.location()
            # Vision coordinates use a lower-left origin; OpenCV uses a
            # top-left origin. Convert once at the API boundary.
            landmarks[name] = np.array([float(location.x), 1.0 - float(location.y)], dtype=np.float32)
            confidences.append(confidence)
        if not confidences or min(confidences) < self.min_confidence:
            return None
        return landmarks

    @staticmethod
    def _center(landmarks, width, height):
        points = np.asarray(list(landmarks.values()), dtype=np.float32)
        return float(points[:, 0].mean() * width), float(points[:, 1].mean() * height)

    @staticmethod
    def _box(landmarks, width, height):
        points = np.asarray(list(landmarks.values()), dtype=np.float32)
        xs = points[:, 0] * width
        ys = points[:, 1] * height
        pad = max(14, int(min(width, height) * 0.03))
        return (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(width - 1, int(xs.max()) + pad),
            min(height - 1, int(ys.max()) + pad),
        )

    def detect_hand_state(self, frame_bgr, locked_center=None):
        observations = self._request_points(frame_bgr)
        if not observations:
            return None, False, False, False, locked_center

        height, width = frame_bgr.shape[:2]
        candidates = []
        for observation in observations:
            landmarks = self._observation_landmarks(observation)
            if landmarks is None:
                continue
            box = self._box(landmarks, width, height)
            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            center = self._center(landmarks, width, height)
            try:
                chirality = int(observation.chirality())
            except Exception:
                chirality = None
            candidates.append((area, center, box, landmarks, chirality))

        if not candidates:
            return None, False, False, False, locked_center

        if locked_center is None:
            selected = max(candidates, key=lambda item: item[0])
            selected_center = selected[1]
            self.locked_center = selected_center
            self.locked_handedness = selected[4]
        else:
            def score(item):
                dx = item[1][0] - locked_center[0]
                dy = item[1][1] - locked_center[1]
                return dx * dx + dy * dy

            selected = min(candidates, key=score)
            max_jump = (min(width, height) * 0.22) ** 2
            if score(selected) > max_jump:
                return None, False, False, False, locked_center

            if self.locked_handedness is not None:
                matching = [item for item in candidates if item[4] == self.locked_handedness]
                if matching:
                    selected = min(matching, key=score)
                elif score(selected) > max_jump * 0.5:
                    return None, False, False, False, locked_center

            selected_center = (
                locked_center[0] * (1.0 - self.lock_alpha) + selected[1][0] * self.lock_alpha,
                locked_center[1] * (1.0 - self.lock_alpha) + selected[1][1] * self.lock_alpha,
            )
            self.locked_center = selected_center

        _, _, box, landmarks, _ = selected
        is_open, is_fist, is_l = self._classify(landmarks)
        return box, is_fist, is_l, True, selected_center

    def detect_hand_box(self, frame_bgr):
        box, _, _, found, _ = self.detect_hand_state(frame_bgr, self.locked_center)
        return box if found else None

    def is_fist(self, frame_bgr):
        _, is_fist, _, found, _ = self.detect_hand_state(frame_bgr, self.locked_center)
        return is_fist, found

    def close(self):
        self.request = None
