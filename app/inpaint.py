# inpaint wrapper: provides quick OpenCV inpaint and hooks for advanced methods (LaMa / E2FGVI / Stable Diffusion)
import cv2
import numpy as np

def opencv_inpaint(frame, mask, radius=3, method='telea'):
    m = (mask > 127).astype('uint8')
    inpaintFlags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    res = cv2.inpaint(frame, m, radius, inpaintFlags)
    return res

# Advanced inpainting hooks:
def la_ma_inpaint(frame, mask, la_ma_path=None, device='cpu'):
    """Placeholder: use LaMa inpainting model.
    To enable:
    - clone the LaMa repo and install dependencies
    - download pretrained weights and set la_ma_path
    - implement loading and calling LaMa model here
    """
    raise NotImplementedError('LaMa inpaint integration not implemented. See README for instructions.')

def e2fgvi_inpaint_video(frames, masks, device='cuda'):
    """Placeholder: integrate E2FGVI video inpainting for temporal coherence.
    E2FGVI repo: https://github.com/MCG-NKU/E2FGVI
    """
    raise NotImplementedError('E2FGVI integration placeholder. Install E2FGVI and adapt this function to call it.')
