# segmentation wrapper: tries YOLOv8 via ultralytics, falls back to simple color/edge heuristic
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

class Segmenter:
    def __init__(self, model_name='yolov8m-seg.pt', device='cpu', conf=0.35):
        self.device = device
        self.conf = conf
        self.model_name = model_name
        self.model = None
        if HAS_YOLO:
            try:
                self.model = YOLO(model_name)
                self.model.fuse()
            except Exception:
                self.model = None

    def predict_mask(self, frame):
        # If YOLO is available, use it (segmentation mode)
        if self.model is not None:
            try:
                results = self.model.predict(source=frame, conf=self.conf, imgsz=640, device=self.device, save=False, verbose=False)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for r in results:
                    if getattr(r, 'masks', None) is None:
                        continue
                    # ultralytics v8: r.masks.data is an array of masks
                    try:
                        mdata = r.masks.data
                        for m in mdata:
                            m = (m > 0.5).astype('uint8')
                            mask = np.maximum(mask, m)
                    except Exception:
                        # fallback polygons
                        polys = getattr(r.masks, 'xy', [])
                        for poly in polys:
                            pts = np.array(poly, dtype=np.int32)
                            cv2.fillPoly(mask, [pts], 1)
                mask = (mask * 255).astype('uint8')
                return mask
            except Exception as e:
                print('YOLO segmentation failed, falling back to heuristic:', e)

        # Fallback heuristic: green/leaf color segmentation in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 40, 20])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
