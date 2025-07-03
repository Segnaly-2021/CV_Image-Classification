import os
import cv2
import uvicorn  # noqa: F401
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Handwritten_digit_Identifier")

# Load model once at startup
try:
    model_path = os.path.abspath("./app/models/")
    model = tf.saved_model.load(model_path)
    pred_func = model.signatures['serving_default']
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def homepage():
    return (
        "Welcome to the home page!!! To test the app, go to  http://localhost:8080/docs"
    )

@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    # Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ("jpeg", "jpg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file format")
    
    try:
        # Read file once
        file_bytes = await file.read()
        bytes_to_nparray = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        image = cv2.imdecode(bytes_to_nparray, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_0_1 = image_tensor / 255.0
        input_image = tf.expand_dims(tf.expand_dims(image_0_1, axis=0), axis=3)
        
        # Single prediction
        prediction = pred_func(input_image)['output_0'].numpy()
        y_pred = np.argmax(prediction)
        conf_prob = np.max(prediction) * 100
        
        message = (
            f"The handwritten digit is predicted to be {y_pred} "
            f"with a confidence level of: {conf_prob:.2f}%"
        )
        return {"message": message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    

    
if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)