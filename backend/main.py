import time

import cv2

from camera.camera import Camera


def main():
    camera = Camera()
    camera.start()

    window_name = "Whitehall Canvas"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = camera.read()
            frame = cv2.flip(frame, 1)

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            height, width = frame.shape[:2]

            cv2.putText(
                frame,
                f"Resolution: {width}x{height}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()