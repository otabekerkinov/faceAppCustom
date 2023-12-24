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
    all_face_sizes = []

    for filename in os.listdir(dataset_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Checking for image files
            file_path = os.path.join(dataset_directory, filename)
            image = cv2.imread(file_path)
            if image is not None:
                _, _, face_sizes = detect_faces(image)
                all_face_sizes.extend(face_sizes)  # Accumulate face sizes

    # After processing all images, plot the cumulative data
    plot_face_data(all_face_sizes)


# detect_faces takes a cv2 image object and uses pretrained model to detect faces and draws a rectangle around it
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    face_sizes = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_sizes.append(w*h)  # Adding face size (area) for EDA

    for (x, y, w, h) in faces:
        # Extract each face
        face_image = image[y:y+h, x:x+w]
        face_image_pil = Image.fromarray(face_image)  # Convert to PIL image

        # Predict age
        predicted_age = predict_age(face_image_pil)

        # Draw rectangle and put text (age)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'Age: {int(predicted_age)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    return image, faces, face_sizes

# plot_face_data generates a histogram of dataset distribution (frequency of face sizes)
def plot_face_data(face_sizes):
    if not face_sizes:
        print("No faces detected in the dataset.")
        return

    mean_size = np.mean(face_sizes)
    median_size = np.median(face_sizes)
    min_size = min(face_sizes)
    max_size = max(face_sizes)

    plt.hist(face_sizes, bins=30, color='skyblue', edgecolor='black')

    plt.axvline(mean_size, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_size:.2f}')
    plt.axvline(median_size, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_size:.2f}')

    plt.title('Distribution of Face Sizes in the Dataset')
    plt.xlabel('Face Area (pixels²)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

    # Print basic statistics
    print(f"Basic Statistics:\nMin Size: {min_size}\nMax Size: {max_size}\nMean Size: {mean_size:.2f}\nMedian Size: {median_size:.2f}")

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


if __name__  == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python face_detection.py mode filepath")
        print("mode: image, video, live")
        sys.exit(1)

    mode = sys.argv[1]
    file_path = sys.argv[2]

    if mode == 'aug_img':
        process_image_with_augmentation(file_path)
    elif mode == 'image':
        process_image(file_path)
    elif mode == 'video':
        process_video(file_path)
    elif mode == 'live':
        live_video()
    elif mode == 'dataset':
        process_dataset(file_path)
    else:
        print("Invalid mode")