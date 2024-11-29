from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import io  # Needed to convert file into byte stream

app = Flask(__name__)

# Load your trained model
model = load_model('/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/process10.h5')

# Define class labels and image dimensions
img_height = 177
img_width = 177
class_labels = ['grape', 'apple', 'starfruit', 'orange', 'kiwi', 'mango', 'pineapple', 'banana', 'watermelon', 'strawberry']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    try:
        # Convert the file to a byte stream
        img_bytes = file.read()
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

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
