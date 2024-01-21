import torch
import torchvision.models as models
import torch.nn as nn



class CustomResNet(nn.Module):
    def __init__(self, original_model, dropout_rate=0.4):
        super(CustomResNet, self).__init__()
        # Everything up to the last layer of layer4
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
        # New dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # The last layer of layer4
        self.layer4 = list(original_model.children())[-2]

        # The original fully connected layer is replaced with a new one
        self.fc = nn.Linear(original_model.fc.in_features, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)  # Flatten the features out
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        return x


class CustomGoogLeNet(nn.Module):
    def __init__(self, original_model, num_classes=1):
        super(CustomGoogLeNet, self).__init__()
        # Use all layers except the last one
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # New fully connected layer
        self.fc = nn.Linear(1024, num_classes)  # GoogLeNet uses 1024 features before the final layer

    def forward(self, x):
        x = self.features(x)
        # GoogLeNet specific flattening
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

