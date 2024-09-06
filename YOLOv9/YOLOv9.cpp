#include "YOLOv9.hpp"

YOLOv9::YOLOv9(): TensorRT()
{
}

YOLOv9::~YOLOv9(): ~TensorRT()
{
}

void YOLOv9::preProcess(cv::Mat& image)
{
    blobFromImage(image);
}

void YOLOv9::postProcess(const int image_w, const int image_h, float scaleX, float scaleY)
{
}

/// <summary>
/// Inference
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
std::vector<Object> YOLOv9::Detect(cv::Mat image)
{
    preProcess(image);
    
    doInference();

    postProcess(_width, _height, _width / (float)image.cols, _height / (float)image.rows);

    return _objects;
}