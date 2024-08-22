#pragma once
#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"

#define NUM_CLASSES 80
#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.25f

class YOLOX : public TensorRT
{
public:
	bool Initialize(const char* model_path, int model_width, int model_height);
	std::vector<Object> Detect(cv::Mat image);

protected:
	void preProcess(cv::Mat& image);
	std::vector<Object> postProcess(float scaleX, float scaleY);

	void generate_proposals(float scaleX, float scaleY, float prob_threshold);
	std::vector<int> nms(float nms_threshold);

private:
	float _nms_threshold = NMS_THRESH;
	float _bbox_confidential_threshold = BBOX_CONF_THRESH;
	uint _numClasses = NUM_CLASSES;

	std::vector<Object> _proposals;
	std::vector<Object> _objects;
};

