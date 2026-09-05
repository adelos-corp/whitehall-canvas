import cv2
from mediapipe.python.solutions import hands as mp_hands


class HandGestureDetector:
    """Detect whether the visible hand is open or closed (fist)."""

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    @staticmethod
    def _finger_extended(landmarks, tip, pip):
        # In the mirrored camera view, comparing y coordinates works for the
        # four non-thumb fingers. A small margin prevents jitter near a fist.
        return landmarks[tip].y < landmarks[pip].y - 0.02

    def is_fist(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return False, False

        landmarks = result.multi_hand_landmarks[0].landmark
        extended = sum(
            self._finger_extended(landmarks, tip, pip)
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

        # Treat 0 or 1 extended finger as a closed fist. This is intentionally
        # forgiving because the demo should not depend on perfect hand pose.
        return extended <= 1, True

    def close(self):
        self.hands.close()
