import torch
import torchvision.models as models

# Load the WideResNet50_2 pre-trained model
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)

# Set the model to evaluation mode (important for certain layers like dropout)
model.eval()

# Create a dummy input tensor of the correct size (batch_size, channels, height, width)
# For example, (1, 3, 256, 256) where 1 is the batch size, 3 is the number of color channels (RGB), and 256x256 is the image size
dummy_input = torch.randn(1, 3, 256, 256)  # Adjust the size if necessary

# Define the path where you want to save the ONNX model
onnx_file_path = "wide_resnet50_2.onnx"

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_file_path, 
                  export_params=True,         # Store the trained parameters in the model
                  opset_version=11,           # ONNX version, 11 is commonly used
                  do_constant_folding=True,   # Optimize constant folding (optional)
                  input_names=["input"],      # Name for the input tensor
                  output_names=["output"],    # Name for the output tensor
                  dynamic_axes={              # Specify axes that can be dynamic (e.g., batch size)
                      'input': {0: 'batch_size'},   # Batch size is dynamic
                      'output': {0: 'batch_size'}
                  })

print("Model successfully exported to ONNX format!")
