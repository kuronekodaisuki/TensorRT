#pragma once
#ifdef WIN32
#pragma warning(disable: 4819)
#endif
#include <opencv2/opencv.hpp>


class Object
{
public:
    cv::Rect_<float> rect;
    int label;
    float prob;

    bool operator<(const Object& right) const {
        return prob > right.prob;
    }

    bool Send(std::ostream& stream);

    //static void Set(char* names[]);
private:
    //static const char* _names[];
};