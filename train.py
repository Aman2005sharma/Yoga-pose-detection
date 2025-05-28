import cv2
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# Load SVM model
with open("yoga_pose_svm.pkl", "rb") as f:
    model_data = pickle.load(f)

# Setup ResNet50 feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval().to(device)

# Define preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.cpu().numpy().reshape(1, -1)

# Optional: Expected pose name for feedback
expected_pose = "Tree"  # Replace with your target pose or leave empty

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
else:
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict
        features = extract_features(frame)
        pred = model_data["model"].predict(features)[0]
        pose = model_data["class_names"][pred]

        # Add text overlay
        cv2.putText(frame, f"Pose: {pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Check if expected pose matches
        if expected_pose:
            if pose.lower() == expected_pose.lower():
                feedback = "Correct Pose"
                color = (0, 255, 0)
            else:
                feedback = "Incorrect Pose"
                color = (0, 0, 255)
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

        cv2.imshow("Yoga Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
