import torch.nn as nn
from torchsummary import summary
from torchvision import models

class DenseNet201Model(nn.Module):
    def __init__(self, num_labels):
        super(DenseNet201Model, self).__init__()

        # Load pre-trained DenseNet-201 model
        self.densenet201 = models.densenet201(pretrained=True)

        # Freeze all parameters in the pre-trained model
        for param in self.densenet201.parameters():
            param.requires_grad = False

        # Replace the final layer of the classifier
        in_features = self.densenet201.classifier.in_features

        self.densenet201.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.densenet201(x)

# if __name__ == "__main__":
#     # Create an instance of the model
#     model = DenseNet201Model(num_labels=8)

#     # Use torchsummary to print the model summary
#     summary(model, input_size=(3, 256, 256))