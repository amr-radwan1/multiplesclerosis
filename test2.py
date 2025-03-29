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
        self.resnet = models.resnet50(pretrained=False)  
        
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.resnet(x)

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = cv2.resize(img, (224, 224))
    img = img / np.max(img)  # Normalize to [0,1]
    return img

def predict_single_image(model, image_path, device):
    # Load and preprocess image
    if image_path.endswith('.dcm'):
        img = load_dicom(image_path)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"MRI Scan: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    img_tensor = normalize(img_tensor)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probability = output.item()
    
    # Display result
    diagnosis = "Multiple Sclerosis" if probability > 0.5 else "Healthy"
    confidence = probability if probability > 0.5 else 1 - probability
    
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"Diagnosis: {diagnosis}")
    print(f"Confidence: {confidence:.2%}")
    print(f"MS Probability: {probability:.4f}")
    
    return probability

# Function to predict on multiple images in a folder
def predict_folder(model, folder_path, device):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm") or file.endswith(".jpg"):
                path = os.path.join(root, file)
                print(f"\nAnalyzing: {path}")
                probability = predict_single_image(model, path, device)
                results.append({
                    'file': path,
                    'probability': probability,
                    'diagnosis': "MS" if probability > 0.5 else "Healthy"
                })
    
    # Print summary
    print("\n===== PREDICTION SUMMARY =====")
    print(f"Total images analyzed: {len(results)}")
    ms_count = sum(1 for r in results if r['diagnosis'] == "MS")
    healthy_count = len(results) - ms_count
    print(f"MS detected: {ms_count} images")
    print(f"Healthy: {healthy_count} images")
    
    return results

# Main prediction function
def run_prediction():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model_path = 'best_ms_classifier.pth'  # Path to your saved model
    
    # Initialize the model
    model = MSClassifier().to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    predict_folder(model, "combination", device)  

if __name__ == "__main__":
    run_prediction()