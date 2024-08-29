// Retinanet.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <opencv2/opencv.hpp>
#include "Retinanet_resnet50_fpn.h"

const char* MODEL = "../models/retinanet_resnet50_fpn.onnx";
const char* ENGINE = "../models/retinanet_resnet50_fpn.trt";
const int MODEL_WIDTH = 650;
const int MODEL_HEIGHT = 400;

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
	Retinanet_resnet50_fpn resnet;
	if (FileExists(ENGINE))
	{
		resnet.LoadEngine(ENGINE, MODEL_WIDTH, MODEL_HEIGHT);
	}
	else
	{
		if (resnet.Initialize(MODEL, MODEL_WIDTH, MODEL_HEIGHT))
		{
			resnet.SaveEngine(ENGINE);
		}
	}
}

