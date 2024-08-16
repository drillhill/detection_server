import cv2
import uuid
import time
from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np

# Initialize FastAPI app
app = FastAPI()

class TextRegionCropper:
    def __init__(self, det_model_dir='det_db_inference',rec_model_dir='en_PP-OCRv4_rec', show_log=False):
        self.ocr = PaddleOCR(det_model_dir=det_model_dir,rec_char_dict_path='./sidecode.txt',rec_model_dir=rec_model_dir, show_log=show_log,use_gpu=False)

    def crop_text_regions(self, img_path: str):
        # Record start time
        start_time = time.time()

        # Load the image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Image could not be read"}, 400

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(f"rotates.jpg", img)
        # Perform OCR to detect text regions
        ocr_results = self.ocr.ocr(img=img, rec=True)
        print(ocr_results)


# Instance of TextRegionCropper
cropper = TextRegionCropper()
cropper.crop_text_regions('/data/users/bao.nguyen/shipment-server-final/detection_server/5.png')