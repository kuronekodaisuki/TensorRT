// YOLOX.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <opencv2/opencv.hpp>
#include "YOLOX.h"
#include "../include/Object.h"

const char* MODEL = "../models/yolox_m_736x1280.onnx";
const char* ENGINE = "../models/yolox_m_736x1280.trt";
const int MODEL_WIDTH = 1280;
const int MODEL_HEIGHT = 736;


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
	cv::Mat image;
	YOLOX yolo;
	if (FileExists(ENGINE))
	{
		yolo.LoadEngine(ENGINE, MODEL_WIDTH, MODEL_HEIGHT);
	}
	else
	{
		if (yolo.LoadModel(MODEL, MODEL_WIDTH, MODEL_HEIGHT))
		{
			yolo.SaveEngine(ENGINE);
		}
	}

	if (2 <= argc)
	{
		puts(argv[1]);
		image = cv::imread(argv[1]);
		std::vector<Object> objects = yolo.Detect(image);

		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].Draw(image);
		}
		cv::imshow("out", image);

		switch (cv::waitKey(0))
		{
		case ' ':
			cv::imwrite("out.png", image);
			break;
		}
	}

}

