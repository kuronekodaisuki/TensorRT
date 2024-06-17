#ifndef YOLOv9_INCLUDED
#include "../include/TensorRT.h"

class YOLOv9: public TensorRT
{
public:
    YOLOv9(/* args */);
    ~YOLOv9();

    /// <summary>
    /// Detect objects
    /// </summary>
    /// <param name="image"></param>
    /// <returns>Detected objects</returns>
    std::vector<Object> Detect(cv::Mat image);

protected:
    void preProcess(cv::Mat& image);
    void postProcess(const int image_w, const int image_h, float scaleX, float scaleY);
private:
    /* data */
};

#define YOLOv9_INCLUDED
#endif