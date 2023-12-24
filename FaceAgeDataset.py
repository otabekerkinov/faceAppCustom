from torch.utils.data import Dataset
from PIL import Image
import os

class FaceAgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the age folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Iterate over each age directory
        for age_dir in os.listdir(root_dir):
            if not os.path.isdir(os.path.join(root_dir, age_dir)):
                continue  # Skip files

            age = int(age_dir)  # Convert directory name to integer
            age_folder = os.path.join(root_dir, age_dir)

            # Iterate over each image in the directory
            for img_name in os.listdir(age_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(age_folder, img_name)
                    self.samples.append((img_path, age))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image file

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, age
