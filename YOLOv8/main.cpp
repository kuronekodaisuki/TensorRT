#include <opencv2/opencv.hpp>
#include "YOLOv8.h"
#include "../include/Object.h"

const char* MODEL = "../models/yolov8s.onnx";
const char* ENGINE = "../models/yolov8s.engine";
const int MODEL_WIDTH = 640;
const int MODEL_HEIGHT = 640;

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
bool FileExists(const char* path) { return fs::exists(path); }
#else
#include <sys/stat.h>
bool FileExists(const char* name)
{
	struct stat   buffer;
	return (stat(name, &buffer) == 0);
}
#endif


int main(int argc, char* argv[])
{
	YOLOv8 yolo;
	if (FileExists(ENGINE))
	{
		yolo.LoadEngine(ENGINE, MODEL_WIDTH, MODEL_HEIGHT);
	}
	else 
	{
		if (yolo.Initialize(MODEL, MODEL_WIDTH, MODEL_HEIGHT))
		{
			yolo.SaveEngine(ENGINE);
		}
	}

	cv::Mat image;
	if (2 <= argc)
	{
		image = cv::imread(argv[1]);
		std::vector<Object> objects = yolo.Detect(image);
		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].Draw(image);
		}
		cv::imwrite("out.png", image);
	}
	else
	{
		cv::VideoCapture camera;
		if (camera.open(0))
		{
			for (bool loop = true; loop && camera.read(image);)
			{
				std::vector<Object> objects = yolo.Detect(image);
				for (int i = 0; i < objects.size(); i++)
				{
					objects[i].Draw(image);
				}
				cv::imshow("out", image);

				switch (cv::waitKey(1))
				{
				case 'q':
					loop = false;
					break;

				case ' ':
					cv::imwrite("out.png", image);
					break;
				}
			}
		}
	}

}