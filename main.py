from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import io
import os

app = FastAPI()

# Load your trained model
model = load_model('/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/process10.h5')

# Define class labels and image dimensions
img_height = 177
img_width = 177
class_labels = ['grape', 'apple', 'starfruit', 'orange', 'kiwi', 'mango', 'pineapple', 'banana', 'watermelon', 'strawberry']

# Path to index.html
index_file_path = os.path.join(os.path.dirname(__file__), 'index.html')

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        with open(index_file_path, "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

# Endpoint to classify image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    try:
        # Read file bytes
        img_bytes = await file.read()
        img = load_img(io.BytesIO(img_bytes), target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)
        probabilities = predictions[0]

        # Sort results
        sorted_indices = np.argsort(probabilities)[::-1]
        results = [
            {'label': class_labels[i], 'probability': float(probabilities[i] * 100)}
            for i in sorted_indices
        ]

        return JSONResponse(content={'results': results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files if needed
static_folder = os.path.dirname(index_file_path)
app.mount("/static", StaticFiles(directory=static_folder), name="static")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
