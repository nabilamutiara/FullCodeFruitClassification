import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


model = load_model('/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/process3.h5')


img_height = 177  
img_width = 177   
class_labels = ['anggur', 'apel', 'belimbing', 'jeruk', 'kiwi', 'mangga', 'nanas', 'pisang', 'semangka', 'stroberi']  


def classify_image(img_path):
    
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    
    probabilities = predictions[0]
    
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]

    for label, prob in zip(sorted_labels, sorted_probabilities):
        print(f'{label.upper()} {prob * 100:.2f}%')

image_path = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/validation/dataset_anggur/YDO408ACEHZJ.jpg'  # Replace with the actual image path
classify_image(image_path)
