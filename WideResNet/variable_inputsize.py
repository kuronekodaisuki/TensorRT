import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def Classify(args):
    filename = args.filename
    kernel_size = args.kernel
    resize = args.size

    print(filename, kernel_size, resize)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    model = models.wide_resnet50_2(weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)

    model.conv1 = nn.Conv2d(in_channels=3,
                        out_channels=model.conv1.out_channels,
                        kernel_size=(kernel_size, kernel_size),
                        stride=2,
                        padding=3,
                        bias=False)

    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.open(filename))
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)

    #print(output[0])

    probabilities = torch.nn.functional.softmax(output[0], dim = 0)

    prob, cat = torch.topk(probabilities, 10)
    for i in range(prob.size(0)):
        print(categories[cat[i]], prob[i].item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='../images/IMG_1133.jpg')
    parser.add_argument('-s', '--size', default=256, type=int)
    parser.add_argument('-k', '--kernel', default=7, type =int)
    Classify(parser.parse_args())
