import cv2


class Camera:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None

    def start(self) -> None:
        # Explicitly use Apple's AVFoundation backend on macOS.
        self.capture = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_AVFOUNDATION,
        )

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("Could not open camera.")

        # Request a sensible camera mode for the live canvas.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 60)

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera has not been started.")

        success, frame = self.capture.read()

        if not success or frame is None:
            raise RuntimeError("Could not read frame.")

        return frame

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
