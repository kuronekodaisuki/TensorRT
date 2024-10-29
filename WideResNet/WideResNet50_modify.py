import torch
from torchvision import models

# Define a custom model that returns layer2 and layer3 outputs
class WideResNet50_2_Modified(torch.nn.Module):
    def __init__(self):
        super(WideResNet50_2_Modified, self).__init__()
        # Load pre-trained WideResNet50_2 model
        self.wide_resnet = models.wide_resnet50_2(pretrained=True)
       
        # Extract layers up to the desired points (conv1, bn1, layer1, etc.)
        self.conv1 = self.wide_resnet.conv1
        self.bn1 = self.wide_resnet.bn1
        self.relu = self.wide_resnet.relu
        self.maxpool = self.wide_resnet.maxpool
       
        # The residual blocks
        self.layer1 = self.wide_resnet.layer1
        self.layer2 = self.wide_resnet.layer2
        self.layer3 = self.wide_resnet.layer3

    def forward(self, x):
        # Pass the input through the initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        # Pass through layer1
        x = self.layer1(x)
       
        # Extract output from layer2
        layer2_output = self.layer2(x)
       
        # Extract output from layer3
        layer3_output = self.layer3(layer2_output)
       
        # Return the outputs of both layers
        return layer2_output, layer3_output

# Instantiate the modified model
model = WideResNet50_2_Modified()

# Test the modified model with a dummy input
input_tensor = torch.randn(1, 3, 224, 224)  # Example input: batch_size=1, channels=3, height=224, width=224
layer2_out, layer3_out = model(input_tensor)

print(f"Layer 2 output shape: {layer2_out.shape}")
print(f"Layer 3 output shape: {layer3_out.shape}")
