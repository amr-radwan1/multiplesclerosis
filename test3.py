
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pydicom
import cv2
import matplotlib.pyplot as plt

class MSClassifier(nn.Module):
    def __init__(self):
        super(MSClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Use smaller backbone
        
        # Modify first layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final layer with stronger regularization
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Dropout(0.6),      # Increase dropout
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.resnet(x)
    


def load_image(path):
    if path.lower().endswith(('.dcm', '.dicom')):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        img = cv2.resize(img, (224, 224))
        img = img / np.max(img) 
        return img
    elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Load JPG/PNG image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  
        return img
    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_model_and_predict_folder(model_path, folder_path, device, threshold=0.5):
    # 1. Load the model
    model = MSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Iterate through all images in the folder
    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue  # Skip non-file entries
        
        try:
            # Load and preprocess the image
            img = load_image(file_path)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            img_tensor = normalize(img_tensor).to(device)
            
            # Make prediction with confidence score
            with torch.no_grad():
                output = model(img_tensor)
                probability = output.item()  # Get the raw probability
                prediction = 1 if probability >= threshold else 0
            
            # Append result for this image
            results.append({
                "filename": filename,
                "prediction": "MS" if prediction == 1 else "Healthy",
                "probability": probability,
                "confidence": probability if prediction == 1 else 1 - probability
            })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return results

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = load_model_and_predict_folder("best_ms_classifier.pth", "combination", device)
count_correct = 0
count_total = len(results)
for result in results:
    if "healthy" in result["filename"] and result["prediction"] == "Healthy":
        count_correct += 1
    elif "ms" in result["filename"] and result["prediction"] == "MS":
        count_correct += 1
    print(f"\nFile: {result['filename']}")
    print(f"Diagnosis: {result['prediction']}")
    print(f"MS Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.2%}")
print(f"Correct Predictions: {count_correct}/{count_total} ({(count_correct/count_total)*100:.2f}%)")

