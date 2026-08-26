import cv2


class Camera:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera.")

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera has not been started.")

        success, frame = self.capture.read()

        if not success:
            raise RuntimeError("Could not read frame.")

        return frame

def stop(self) -> None:
    if self.capture is not None:
        self.capture.release()
        self.capture = None