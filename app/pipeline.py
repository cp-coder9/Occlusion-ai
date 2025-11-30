import time
import cv2
from .capture import CameraCapture
from .segment import Segmenter
from .inpaint import opencv_inpaint
from .depth import DepthEstimator
from .tracker import SimpleTracker
from .renderer import overlay_predicted_objects, blend_inpainted

class Pipeline:
    def __init__(self, cfg):
        cam_src = cfg['camera'].get('url', 0) if cfg['camera'].get('type','rtsp') == 'rtsp' else cfg['camera'].get('url', 0)
        self.cam = CameraCapture(src=cam_src)
        self.segmenter = Segmenter(model_name=cfg['segmentation'].get('model','yolov8m-seg.pt'),
                                   device=cfg['processing'].get('device','cpu'),
                                   conf=cfg['segmentation'].get('conf_thresh',0.35))
        self.depth = DepthEstimator(device=cfg['processing'].get('device','cpu'))
        self.tracker = SimpleTracker(max_lost=10)
        self.cfg = cfg

    def run(self):
        fps_cap = self.cfg['processing'].get('fps_cap', 15)
        while True:
            t0 = time.time()
            frame = self.cam.read()
            if frame is None:
                break

            mask = self.segmenter.predict_mask(frame)
            inpainted = opencv_inpaint(frame, mask)

            # quick motion detection behind occluder
            fg = cv2.absdiff(cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            _, fgmask = cv2.threshold(fg, 30, 255, cv2.THRESH_BINARY)
            hidden_motion = cv2.bitwise_and(fgmask, mask)
            cnts, _ = cv2.findContours(hidden_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if w*h < 200: continue
                centroids.append((x + w//2, y + h//2))

            objects = self.tracker.update(centroids)
            cleaned = blend_inpainted(frame, inpainted, mask)
            out = overlay_predicted_objects(cleaned, objects)

            if self.cfg['output'].get('show_window', True):
                cv2.imshow('occlusion-removed', out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.cfg['output'].get('save_out', False):
                # implement saving frames / video
                pass

            dt = time.time() - t0
            if dt < 1.0 / fps_cap:
                time.sleep(1.0 / fps_cap - dt)

        self.cam.release()
        cv2.destroyAllWindows()
