import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox



# Load the trained age detection model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('best_age_detection_model.pth'))
model.eval()



# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define transform for age prediction
age_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def map_age_to_group(age):
    if age < 10:
        return '1-9'
    elif age < 20:
        return '10-19'
    elif age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    elif age < 80:
        return '70-79'
    elif age >= 80:
        return '80+'

def predict_age_group(face_image):
    predicted_age = predict_age(face_image)
    age_group = map_age_to_group(predicted_age)
    return age_group
    
def predict_age(face_image):
    """
    Predict the age of a face.
    """
    face_image = age_transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        age_prediction = model(face_image).item()
    return age_prediction


# process_dataset processes all images of a given directory and plots the cumulative data
def process_dataset(dataset_directory):
    actual_age_groups = []
    predicted_age_groups = []

    # Get the actual age labels for the dataset
    actual_age_labels = get_actual_age_labels(dataset_directory)

    # Iterate over each image path and its corresponding actual age in the dataset
    for image_path, actual_age in actual_age_labels.items():
        image = cv2.imread(image_path)
        if image is not None:
            # The detect_faces function now returns predicted ages instead of face sizes
            _, _, predicted_ages = detect_faces(image)
            # For each detected face, match the predicted age group with the actual age group
            for predicted_age in predicted_ages:
                predicted_age_group = map_age_to_group(predicted_age)
                actual_age_group = map_age_to_group(actual_age)
                predicted_age_groups.append(predicted_age_group)
                actual_age_groups.append(actual_age_group)

    # At this point, actual_age_groups and predicted_age_groups should have the same length
    # Compute the confusion matrix and plot it
    compute_and_plot_confusion_matrix(actual_age_groups, predicted_age_groups)


def compute_and_plot_confusion_matrix(actual_age_groups, predicted_age_groups):
    age_groups = ['1-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    conf_matrix = confusion_matrix(actual_age_groups, predicted_age_groups, labels=age_groups)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=age_groups, yticklabels=age_groups)
    plt.xlabel('Predicted Age Group')
    plt.ylabel('Actual Age Group')
    plt.title('Age Group Confusion Matrix')
    plt.show()


def get_actual_age_labels(dataset_directory):
    actual_age_labels = {}
    # Iterate through each folder in the base directory
    for folder_name in os.listdir(dataset_directory):
        folder_path = os.path.join(dataset_directory, folder_name)
        # Check if it's a directory and the folder name is a number
        if os.path.isdir(folder_path) and folder_name.isdigit():
            # Convert folder name to integer age
            age = int(folder_name.lstrip("0"))  # Remove leading zeros
            # Append the age label for each image file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(folder_path, file_name)
                    actual_age_labels[file_path] = age
    return actual_age_labels



# detect_faces takes a cv2 image object and uses pretrained model to detect faces and draws a rectangle around it
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    predicted_ages = []  # List to store predicted ages
    for (x, y, w, h) in faces:
       # Extract each face
       face_image = image[y:y+h, x:x+w]
       face_image_pil = Image.fromarray(face_image)  # Convert to PIL image
       # Predict age
       predicted_age = predict_age(face_image_pil)
       predicted_ages.append(predicted_age)  # Append predicted age to the list
       # Draw rectangle and put text (age)
       cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
       cv2.putText(image, f'Age: {int(predicted_age)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image, faces, predicted_ages

# plot_face_data generates a histogram of age distribution
def plot_age_data(predicted_ages):
    if not predicted_ages:
        print("No ages detected in the dataset.")
        return

    # Convert ages to integer for plotting
    predicted_ages = [int(age) for age in predicted_ages]

    # Sort the ages and get unique counts for each age
    unique_ages, counts = np.unique(predicted_ages, return_counts=True)

    # Plot the histogram
    plt.figure(figsize=(20, 10))
    plt.bar(unique_ages, counts, width=0.8)

    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Ages in the Dataset')

    # Rotate the x-axis labels to fit them better
    plt.xticks(unique_ages, rotation=90)

    plt.tight_layout()
    plt.show()

# process_image takes a path to an image, processes it and show the result and the histogram
def process_image(file_path):
    image = cv2.imread(file_path)
    if image is not None:
        detected_image, _, _ = detect_faces(image)
        cv2.imshow('Faces with Age Predictions', detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image not found")


def augment_image(image):
    augmentation_type = np.random.choice(['flip_horizontal', 'flip_vertical', 'rotate'])

    if augmentation_type == 'flip_horizontal':
        # Flip the image horizontally
        return cv2.flip(image, 1)
    elif augmentation_type == 'flip_vertical':
        # Flip the image vertically
        return cv2.flip(image, 0)
    elif augmentation_type == 'rotate':
        # Rotate the image by a random angle
        angle = np.random.randint(-30, 30)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))

def process_image_with_augmentation(file_path):
    original_image = cv2.imread(file_path)
    if original_image is not None:
        for i in range(0, 10):
            # Augment the image
            augmented_image = augment_image(original_image)
        # Process the augmented image
            detected_image, faces, face_sizes = detect_faces(augmented_image)
            cv2.imshow('Augmented Image', detected_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Error: Image not found")


def process_video(file_path):
    video = cv2.VideoCapture(file_path)
    all_face_sizes = []  # List to accumulate face sizes for all frames

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            detected_frame, faces, face_sizes = detect_faces(frame)
            all_face_sizes.extend(face_sizes)  # Accumulate face sizes
            cv2.imshow('Faces', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    plot_face_data(all_face_sizes)


def live_video():
    video = cv2.VideoCapture(0)
    all_face_sizes = []  # List to accumulate face sizes for all frames

    while True:
        ret, frame = video.read()
        if ret:
            detected_frame, faces, face_sizes = detect_faces(frame)
            all_face_sizes.extend(face_sizes)  # Accumulate face sizes
            cv2.imshow('Faces', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    plot_face_data(all_face_sizes)


def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

def select_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_video(file_path)

def start_live_video():
    live_video()

# Create the main window
root = tk.Tk()
root.title("Face and Age Detection")

# Create buttons
btn_select_image = tk.Button(root, text="Select Image", command=select_image)
btn_select_image.pack(padx=10, pady=10)

btn_select_video = tk.Button(root, text="Select Video", command=select_video)
btn_select_video.pack(padx=10, pady=10)

btn_live_video = tk.Button(root, text="Start Live Video", command=start_live_video)
btn_live_video.pack(padx=10, pady=10)


if __name__  == "__main__":
    root.mainloop()
    # if len(sys.argv) != 3:
    #     print("Usage: python face_detection.py mode filepath")
    #     print("mode: image, video, live")
    #     sys.exit(1)

    # mode = sys.argv[1]
    # file_path = sys.argv[2]

    # if mode == 'aug_img':
    #     process_image_with_augmentation(file_path)
    # elif mode == 'image':
    #     process_image(file_path)
    # elif mode == 'video':
    #     process_video(file_path)
    # elif mode == 'live':
    #     live_video()
    # elif mode == 'dataset':
    #     process_dataset(file_path)
    # else:
    #     print("Invalid mode")