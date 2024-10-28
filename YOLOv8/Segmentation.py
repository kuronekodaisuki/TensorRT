import argparse
import os
from ultralytics import YOLO

def Inference(config):
    file = os.path.join(config.path, config.filename)
    print(file)
    model = YOLO('yolov8m-seg.pt')
    results = model(file)
    for r in results:
        print(r.masks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='E_7BUVjUYAQ3UHx.jpg')
    parser.add_argument('-p', '--path', default='../images')
    Inference(parser.parse_args())



