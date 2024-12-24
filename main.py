from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import torchvision.transforms as transforms
import numpy as np
import io
import os
import torch.nn as nn

app = FastAPI()

class SimpleNet2D(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)  # Change this to 512
        self.bn3 = nn.BatchNorm2d(512)  # Update batch norm layer accordingly
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)  # Adjust this to match the output from conv3
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


num_classes = 10
# Initialize the model
# Initialize the model
model = SimpleNet2D(num_classes=num_classes)

# Save the model's state_dict after initialization
torch.save(model.state_dict(), "model2.pth")

model_path = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning copy/model2.pth'
# Correct code to load the model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)



model.eval()
# Define class labels and image dimensions
img_height = 177
img_width = 177
class_labels = ['grape', 'apple', 'starfruit', 'orange', 'kiwi', 'mango', 'pineapple', 'banana', 'watermelon', 'strawberry']

# Define the image transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

        # Transform the image to a tensor
        img_tensor = transform(img)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Make predictions using the PyTorch model
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        results = []
        for i in range(top5_prob.size(0)):
            label = class_labels[top5_catid[i]]
            prob = top5_prob[i].item() * 100  # Convert to percentage
            results.append({'label': label, 'probability': prob})

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
