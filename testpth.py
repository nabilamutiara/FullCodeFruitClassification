import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define your model architecture here (SimpleNet2D or any other model)
class SimpleNet2D(nn.Module):
    def __init__(self):
        super(SimpleNet2D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(128 * 20 * 20, 128),  # Corrected fully connected layer input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Path to the saved model
model_path = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/SimpleNet2D_model.pth'

# Transformations for input image
transform_test = transforms.Compose([
    transforms.Resize((177, 177)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to load the model
def load_model(model_path):
    # Choose the device to run the model (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model (this must match the model architecture used for training)
    model = SimpleNet2D()  # Make sure to match your model architecture here
    
    # Load the state_dict of the model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure loading the model on the correct device
    model.eval()  # Set the model to evaluation mode (no dropout or batchnorm update)
    
    return model.to(device), device

# Function to predict the image
def predict_image(model, device, img_path):
    # Open the image and apply transformations
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform_test(img).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference (without calculating gradients)
    with torch.no_grad():
        output = model(img_tensor)

    # Apply sigmoid to get the probability (between 0 and 1)
    probability = output.item()

    # Convert the probability to percentage
    percentage = probability * 100

    # Return the result as percentage
    return f"Persentase gambar ini adalah {percentage:.2f}% kemungkinan Apel"

# Main function for user input
def main():
    model, device = load_model(model_path)  # Load model and device
    
    while True:
        img_path = input("Masukkan path gambar untuk prediksi (atau ketik 'exit' untuk keluar): ")
        if img_path.lower() == 'exit':
            break
        
        if os.path.isfile(img_path):  # Ensure the image path is valid
            result = predict_image(model, device, img_path)
            print(result)
        else:
            print("Path gambar tidak valid. Silakan coba lagi.")

# Run the program
if __name__ == '__main__':
    main()
