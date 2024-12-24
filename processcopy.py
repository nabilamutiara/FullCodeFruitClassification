import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


# Direktori dataset
train_dir = 'datasetdeeplearning/training'
validation_dir = 'datasetdeeplearning/validation'
test_dir = 'datasetdeeplearning/testing'

# Hyperparameter
img_height, img_width = 177, 177
batch_size = 32

# Transformasi data
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = validation_transform

# Dataset dan DataLoader
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=validation_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definisi model
class SimpleNet2D(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet2D, self).__init__()
        
        # Lapisan konvolusional dan Batch Normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Max Pooling dan Global Average Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer dan Dropout
        self.fc1 = nn.Linear(512, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)  # Batch Normalization pada FC1
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)  # Batch Normalization pada FC2
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)  # Dropout untuk regularisasi

    def forward(self, x):
        # Konvolusi dan Batch Normalization dengan LeakyReLU
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten tensor untuk input ke fully connected layer

        # Fully Connected dan Batch Normalization
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)  # Dropout setelah FC1
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)), negative_slope=0.1)
        x = self.fc3(x)  # Output layer
        
        return x


# Model, loss, optimizer
num_classes = len(train_dataset.classes)
model = SimpleNet2D(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Pelatihan model
num_epochs = 50
train_acc_history = []
val_acc_history = []

def train_model():
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_acc_history.append(train_acc)

        # Validasi
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f}")

        scheduler.step()
    torch.save(model.state_dict(), 'model3.pth')
    print("Model saved as model3.pth")

train_model()

# Evaluasi model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Test Accuracy
test_accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion Matrix per Kelas
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix per Kelas:")
print(cm)

# Grafik akurasi
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

# Confusion Matrix sebagai gambar
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(ticks=np.arange(num_classes), labels=train_dataset.classes, rotation=45)
plt.yticks(ticks=np.arange(num_classes), labels=train_dataset.classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
