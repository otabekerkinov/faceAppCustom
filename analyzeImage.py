# from PIL import Image
# import matplotlib.pyplot as plt

# def plot_histogram(image_path):
#     # Open the image
#     image = Image.open(image_path)
    
#     # Convert the image to grayscale
#     grayscale_image = image.convert("L")
    
#     # Get pixel values
#     pixel_values = list(grayscale_image.getdata())
    
#     # Plot histogram
#     plt.hist(pixel_values, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
#     plt.title("Image Histogram")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

# # Replace 'your_image.jpg' with the path to your image file
# plot_histogram('images/face_age/035/035_1814.png')

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def calculate_average_histogram(folder_path):
    histogram_sum = None
    image_count = 0
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                grayscale_image = image.convert("L")
                
                # Sum histograms of each image
                if histogram_sum is None:
                    histogram_sum = np.array(grayscale_image.histogram())
                else:
                    histogram_sum += np.array(grayscale_image.histogram())
                image_count += 1
    
    # Calculate the average histogram
    if image_count > 0:
        average_histogram = histogram_sum / image_count
        return average_histogram
    else:
        return None

def plot_histogram(histogram, title="Average Image Histogram"):
    # Plot histogram
    plt.hist(range(256), bins=256, weights=histogram, color='gray', alpha=0.75)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Average Frequency")
    plt.show()

def main(folder_path):
    average_histogram = calculate_average_histogram(folder_path)
    if average_histogram is not None:
        plot_histogram(average_histogram)
    else:
        print("No images found in the folder.")

# Replace 'your_folder' with the path to your folder containing subfolders of images
main('images/face_age')
