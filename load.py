import os
import pydicom
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = cv2.resize(img, (224, 224))
    img = img / np.max(img)
    return img

ms_images = []
healthy_images = []

ms_folder = "multiple_sclerosis/ST000001"

count_ms = 0
for folder in os.listdir(ms_folder):
    for file in os.listdir(f"{ms_folder}/{folder}"):
        if file.endswith(".dcm"):
            path = os.path.join(ms_folder, folder, file)
            img = load_dicom(path)
            count_ms += 1
            # output_path = os.path.join("output", folder, file.replace(".dcm", ".png"))
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # cv2.imwrite(output_path, img * 255)
            
healthy_folder = "healthy/ST000001"    

count_healthy = 0
for folder in os.listdir(healthy_folder):
    for file in os.listdir(f"{healthy_folder}/{folder}"):
        if file.endswith(".dcm"):
            path = os.path.join(healthy_folder, folder, file)
            img = load_dicom(path)
            count_healthy += 1     

labels = [1] * count_ms + [0] * count_healthy 
print(count_ms, count_healthy)
print(labels)


# # Convert labels to tensors
# y = torch.tensor(labels, dtype=torch.long)  # Assuming labels exist

# # Create dataset & dataloader
# dataset = TensorDataset(X_tensor, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Define a simple CNN
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16 * 112 * 112, 2)  # Adjust based on your dataset

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         return x

# # Training setup
# model = SimpleCNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(5):
#     for images, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

            
    








