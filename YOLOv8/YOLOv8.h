#pragma once
#ifndef YOLOv8_INCLUDED

#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"


#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.25f

typedef struct
{
	float cx;
	float cy;
	float w;
	float h;
	float scores[80];
} CHANNEL;

class YOLOv8 : public TensorRT
{
public:
	YOLOv8();
	~YOLOv8();

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
	uint _numClasses;

	std::vector<Object> _proposals;
	std::vector<Object> _objects;
};

#define YOLOv8_INCLUDED
#endif