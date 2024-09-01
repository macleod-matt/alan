'''
    Module for loading from custom CNN (in progress)
    Must pull from submodules to use this method 
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os

labels = os.listdir('classifier\\Data\\test')
num_classes = len(labels)  # Assuming each folder is a class
classifier_path = "classifier\models\card_classifier.pth"

class CardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # If 128x128 input was used during training
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output dynamically
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = CardClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(classifier_path))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust to match training setup
    transforms.ToTensor(),
])

# Step 3: Capture video from webcam
cap = cv2.VideoCapture(1)

while True:
    # Step 4: Read frame from the webcam
    ret, frame = cap.read()
    
    # Convert the image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Step 5: Preprocess the frame
    input_tensor = transform(pil_img)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Step 6: Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = labels[predicted.item()]  # Use the predicted index to get the label
        confidence_score = confidence.item() * 100  # Convert to percentage
    
    # Step 7: Display the prediction and confidence score
    text = f'Prediction: {label} ({confidence_score:.2f}%)'
    if confidence_score > 80:
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Card Classifier', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
