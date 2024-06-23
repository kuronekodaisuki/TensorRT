// YOLOv8.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//


#include "YOLOv8.h"

YOLOv8::YOLOv8()
{

}

YOLOv8::~YOLOv8()
{

}

bool YOLOv8::Initialize(const char* model_path, int model_width, int model_height)
{
	if (LoadModel(model_path, model_width, model_height))
	{
		return true;
	}
	else
	{
		printf("Failed to load %s\n", model_path);
		return false;
	}
}

std::vector<Object> YOLOv8::Detect(cv::Mat image)
{
	preProcess(image);

	doInference();

	return postProcess();
}

void YOLOv8::preProcess(cv::Mat& image)
{
	blobFromImage(image);
}

std::vector<Object> YOLOv8::postProcess()
{
	std::vector<Object> objects;


	return objects;
}