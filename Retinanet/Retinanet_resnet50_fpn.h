#pragma once
#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"

class Retinanet_resnet50_fpn : public TensorRT
{
public:
	Retinanet_resnet50_fpn(const char* input = "images", const char* output = "3479", int batch_size = 1);
	~Retinanet_resnet50_fpn();

	bool Initialize(const char* model_path, int model_width, int model_height);
	std::vector<Object> Detect(cv::Mat image);

protected:
	void preProcess(cv::Mat& image);
	std::vector<Object> postProcess(float scaleX, float scaleY);
};

