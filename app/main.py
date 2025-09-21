import os
import cv2
import uvicorn  # noqa: F401
import numpy as np
import tensorflow as tf
from ResBlock import ResidualBlock
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Handwritten Digit Identifier")

# Load model once at startup
try:
    model_path = os.path.abspath("./app/models/model.keras")
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"ResidualBlock": ResidualBlock}
    )
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
        content_bytes = await file.read()
        bytes_to_nparray = np.asarray(bytearray(content_bytes), dtype=np.uint8)
        image = cv2.imdecode(bytes_to_nparray, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess
        image_0_1 = image / 255.0
        input_image = tf.expand_dims(image_0_1, axis=0)
        
        # Single prediction
        predictions = model.predict(input_image)
        y_pred = np.argmax(predictions)
        conf_prob = np.max(predictions) * 100
        
        message = (
            f"The digit you provided is predicted to be {y_pred}, "
            f"with a confidence level of: {conf_prob:.2f}%"
        )
        return {"message": message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    

    
if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)