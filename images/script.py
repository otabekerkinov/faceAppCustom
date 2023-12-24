import os
import matplotlib.pyplot as plt

def count_images_in_folders(base_path):
    age_counts = {}

    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"The provided path '{base_path}' does not exist.")
        return

    # Iterate through each folder in the base directory
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # Check if it's a directory and the folder name is a number
        if os.path.isdir(folder_path) and folder_name.isdigit():
            count = 0
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    count += 1
            # Convert folder name to integer age
            age = int(folder_name.lstrip("0"))  # Remove leading zeros
            age_counts[age] = count

    return age_counts

def plot_histogram(age_counts):
    # Sort the dictionary by age
    sorted_age_counts = dict(sorted(age_counts.items()))

    ages = list(sorted_age_counts.keys())
    counts = list(sorted_age_counts.values())

    # Increase the figure size to allow for more space for each bar
    plt.figure(figsize=(20, 10))  # You can adjust the figure size as needed

    # Plot the bars with a smaller width to create space between them
    plt.bar(ages, counts, width=0.5)  # Adjust the width as needed

    plt.xlabel('Age')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Age Group')

    # Rotate the x-axis labels to fit them better
    plt.xticks(ages, rotation=90)  # Rotate the labels vertically

    plt.tight_layout()  # Adjust the layout
    plt.show()

if __name__ == "__main__":
    base_directory_path = "./face_age"  # Replace with your directory path
    image_counts = count_images_in_folders(base_directory_path)
    plot_histogram(image_counts)
