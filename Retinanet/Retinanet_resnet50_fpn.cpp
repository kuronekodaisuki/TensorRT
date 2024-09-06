#include "Retinanet_resnet50_fpn.h"

Retinanet_resnet50_fpn::Retinanet_resnet50_fpn(const char* input, const char* output, int batch_size) : TensorRT(input, output)
{

}

Retinanet_resnet50_fpn::~Retinanet_resnet50_fpn()
{

}

bool Retinanet_resnet50_fpn::Initialize(const char* model_path, int model_width, int model_height)
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

std::vector<Object> Retinanet_resnet50_fpn::Detect(cv::Mat image)
{
	std::vector<Object> objects;

	return objects;
}

void Retinanet_resnet50_fpn::preProcess(cv::Mat& image)
{

}

std::vector<Object> Retinanet_resnet50_fpn::postProcess(float scaleX, float scaleY)
{
	std::vector<Object> objects;

	return objects;
}