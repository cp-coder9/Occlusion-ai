# minimal smoke test to import modules
import app
from app.capture import CameraCapture
from app.segment import Segmenter
from app.inpaint import opencv_inpaint
print('Smoke test passed (imports ok)')