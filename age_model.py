import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from FaceAgeDataset import *
from torch.utils.data import random_split



# Load a pre-trained ResNet
model = models.resnet18(pretrained=True)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

# Replace the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, 1)


# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
face_age_dataset = FaceAgeDataset(root_dir='images/face_age', transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(face_age_dataset))  # 80% of the dataset for training
val_size = len(face_age_dataset) - train_size  # The rest for validation
train_dataset, val_dataset = random_split(face_age_dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss function
criterion = nn.MSELoss()

# Optimizer - Only optimize parameters that require gradients
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Define device: use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to the chosen device
model = model.to(device)

# Define the number of epochs
num_epochs = 10



# Training and validation loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, ages in train_loader:
        # Move data to device
        images = images.to(device)
        ages = ages.to(device).view(-1, 1).float()  # Ensure ages are float

        # Forward pass
        predicted_ages = model(images)
        
        # Compute loss
        loss = criterion(predicted_ages, ages)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradient tracking needed
        val_loss = 0
        for images, ages in val_loader:
            images = images.to(device)
            ages = ages.to(device).view(-1, 1).float()
            
            # Forward pass
            val_predictions = model(images)
            
            # Calculate the batch loss
            batch_loss = criterion(val_predictions, ages)
            val_loss += batch_loss.item()
        
        # Calculate average loss over the validation set
        val_loss /= len(val_loader)
    
    # Print training and validation loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

    # Check if this is the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_age_detection_model_more_layers_saved.pth')  # Save best model



