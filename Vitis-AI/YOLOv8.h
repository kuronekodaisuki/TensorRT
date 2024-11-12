#pragma once
#include <opencv2/core.hpp>

#include "../include/TensorRT.h"
#include "../include/Object.h"

#define NUM_CLASSES 80
#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.25f


namespace vitis
{
	struct Box
	{
		int label;
		int box[4];
	};

	struct YOLOv8Result
	{
		std::vector<Box> bboxes;
	};
 
	class YOLOv8 : public TensorRT
	{
	public:
		YOLOv8();
		~YOLOv8();

		bool create(const std::string& model_name, uint width = 0, uint height = 0, uint channels = 3);

		YOLOv8Result run(const cv::Mat& image);
		//std::vector<YOLOv8Result> run(std::vector<cv::Mat>& images);

	protected:
		void preProcess(const cv::Mat& image);
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
}
