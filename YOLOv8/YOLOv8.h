#pragma once
#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"


class YOLOv8 : public TensorRT
{
public:
	YOLOv8();
	~YOLOv8();

	bool Initialize(const char* model_path, int model_width, int model_height);
	std::vector<Object> Detect(cv::Mat image);

protected:
	void preProcess(cv::Mat& image);
	std::vector<Object> postProcess();

private:
	std::vector<Object> _proposals;
	std::vector<Object> _objects;
};

