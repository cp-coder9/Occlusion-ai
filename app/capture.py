import cv2

class CameraCapture:
    def __init__(self, src=0, width=None, height=None):
        self.src = int(src) if isinstance(src, str) and src.isnumeric() else src
        self.cap = cv2.VideoCapture(self.src)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()
