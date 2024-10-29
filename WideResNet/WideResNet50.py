import torch
from torchvision import models
model = models.wide_resnet50_2()
input_tensor = torch.randn(1, 3, 1024, 1024)
output = model(input_tensor)
print(output.shape)
#model.export(format="onnx")
