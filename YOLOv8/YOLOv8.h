#pragma once
#ifndef YOLOv8_INCLUDED

#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"


#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.3f

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

private:
	struct GridAndStride
	{
		int grid0;
		int grid1;
		int stride;

		GridAndStride(int g0, int g1, int s) :grid0(g0), grid1(g1), stride(s) {}
	};

	float _nms_threshold = NMS_THRESH;
	float _bbox_confidential_threshold = BBOX_CONF_THRESH;
	uint _numClasses;
	//float _scaleX;
	//float _scaleY;

	std::vector<Object> _proposals;
	std::vector<Object> _objects;

	std::vector<GridAndStride> _grid_strides;

	std::vector<GridAndStride> generate_grids_and_stride();
	void generate_proposals(float prob_threshold);
	std::vector<int> nms(float nms_threshold);

};

#define YOLOv8_INCLUDED
#endif