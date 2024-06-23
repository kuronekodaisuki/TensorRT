#include "YOLOv8.h"

const char* MODEL = "../models/yolov8n-seg.onnx";
const int MODEL_WIDTH = 640;
const int MODEL_HEIGHT = 640;

int main(int argc, char* argv[])
{
	YOLOv8 yolo;
	yolo.Initialize(MODEL, MODEL_WIDTH, MODEL_HEIGHT);
}