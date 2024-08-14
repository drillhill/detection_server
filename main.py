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
    def __init__(self, det_model_dir='det_db_inference', show_log=False):
        self.ocr = PaddleOCR(det_model_dir=det_model_dir, show_log=show_log,use_gpu=True)

    async def crop_text_regions(self, img_path: str):
        # Record start time
        start_time = time.time()

        # Load the image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Image could not be read"}, 400

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Perform OCR to detect text regions
        ocr_results = self.ocr.ocr(img=img_path, rec=False)

        # List to hold cropped images
        cropped_images = []

        # Iterate over detected text regions
        for region in ocr_results[0]:  # Assuming the first item is the list of text regions
            # Get the coordinates of the bounding box
            points = region

            # Convert points to integers
            points = [[int(coord) for coord in point] for point in points]

            # Crop the region from the image
            x_min = min([point[0] for point in points])
            x_max = max([point[0] for point in points])
            y_min = min([point[1] for point in points])
            y_max = max([point[1] for point in points])
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Append the cropped image to the list
            cropped_images.append((uuid.uuid4().hex, cropped_img))  # Use UUID as filename

        # Calculate processing time
        processing_time = time.time() - start_time

        return cropped_images, processing_time

# Instance of TextRegionCropper
cropper = TextRegionCropper()

# Endpoint to process image path
@app.post("/process_image")
async def process_image(image_path: str):
    cropped_images, processing_time = await cropper.crop_text_regions(image_path)

    if isinstance(cropped_images, dict):  # If there's an error in processing
        return cropped_images

    # Prepare response
    response = {
        "processing_time": f"{processing_time:.2f} seconds",
        "cropped_images": [f"{img_id}.jpg" for img_id, _ in cropped_images]
    }

    # Save cropped images
    for img_id, cropped_img in cropped_images:
        cv2.imwrite(f"{img_id}.jpg", cropped_img)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", workers=1)
