import cv2
import numpy as np

def overlay_predicted_objects(frame, objects):
    out = frame.copy()
    for oid, c in objects.items():
        cv2.circle(out, (int(c[0]), int(c[1])), 8, (0,255,0), -1)
        cv2.putText(out, f"id:{oid}", (int(c[0])+10, int(c[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    return out

def blend_inpainted(frame, inpainted, mask):
    m = (mask > 127).astype('uint8')[:,:,None]
    out = frame.copy()
    out[m==1] = inpainted[m==1]
    return out
