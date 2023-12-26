
import os
import cv2
import numpy as np
import random 

def augment_image(image):
    augmentation_functions = [
        'flip_horizontal', 'flip_vertical', 'rotate', 'scale', 'noise_injection', 'translate'
    ]
    
    # Randomly choose two unique augmentation functions
    selected_augmentations = random.sample(augmentation_functions, 2)

    for augmentation in selected_augmentations:
        if augmentation == 'flip_horizontal':
            image = cv2.flip(image, 1)
        elif augmentation == 'flip_vertical':
            image = cv2.flip(image, 0)
        elif augmentation == 'rotate':
            angle = np.random.randint(-30, 30)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))
        elif augmentation == 'scale':
            scale_factor = np.random.uniform(0.7, 1.3)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        elif augmentation == 'noise_injection':
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*image.shape) * noise_level * 255
            image = image + noise
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif augmentation == 'translate':
            max_translation = 0.1
            tx = max_translation * image.shape[1] * np.random.uniform(-1, 1)
            ty = max_translation * image.shape[0] * np.random.uniform(-1, 1)
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    return image

def process_images_in_folder(folder_path, postfix='_aug'):
    # Iterate over subfolders in the given folder path
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        # Check if it's indeed a directory
        if os.path.isdir(subdir_path):
            # Process each image in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Construct full file path
                    file_path = os.path.join(subdir_path, filename)
                    # Read the image
                    image = cv2.imread(file_path)
                    # Check if the image was correctly loaded
                    if image is not None:
                        # Augment the image
                        augmented_image = augment_image(image)
                        # Construct the new filename and save the augmented image
                        new_filename = f"{os.path.splitext(filename)[0]}{postfix}{os.path.splitext(filename)[1]}"
                        new_file_path = os.path.join(subdir_path, new_filename)
                        cv2.imwrite(new_file_path, augmented_image)
                    else:
                        print(f"Failed to load image: {file_path}")

if __name__ == "__main__":
    # Assume the script is called with the folder path as the first argument
    process_images_in_folder("./age_dataset")
