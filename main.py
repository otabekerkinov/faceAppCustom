import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def process_dataset(dataset_directory):
    for filename in os.listdir(dataset_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(dataset_directory, filename)
            process_image(file_path)



def detect_faces(image, draw_rectangle=True):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    face_sizes = []
    for (x, y, w, h) in faces:
        if draw_rectangle:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_sizes.append(w*h)  # Adding face size (area) for EDA

    return image, faces, face_sizes

def plot_face_data(face_sizes):
    plt.hist(face_sizes, bins=10)
    plt.title('Distribution of Face Sizes')
    plt.xlabel('Face Area')
    plt.ylabel('Frequency')
    plt.show()

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is not None:
        detected_image, faces, face_sizes = detect_faces(image)
        plot_face_data(face_sizes)
        cv2.imshow('Faces', detected_image)
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
        # Augment the image
        augmented_image = augment_image(original_image)

        # Process the augmented image
        detected_image, faces, face_sizes = detect_faces(augmented_image)
        plot_face_data(face_sizes)
        cv2.imshow('Original Image', original_image)
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

    if mode == 'image':
        process_image(file_path)
        #process_image_with_augmentation(file_path)
    elif mode == 'video':
        process_video(file_path)
    elif mode == 'live':
        live_video()
    else:
        print("Invalid mode")