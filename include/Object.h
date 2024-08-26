#pragma once
#ifndef OBJECT_INCLUDED

#include <opencv2/opencv.hpp>

#include "80categories.h"

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

    void Draw(cv::Mat& image, int offsetX = 0, int offsetY = 0)
    {
        cv::Scalar color = cv::Scalar((uint)(color_list[label][0] * 255), (uint)(color_list[label][1] * 255), (uint)(color_list[label][2] * 255));
        float c_mean = (float)cv::mean(color)[0];

        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::Rect bbox((int)rect.x + offsetX, (int)rect.y + offsetY, (int)rect.width, (int)rect.height);
        cv::rectangle(image, bbox, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[label], prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = (int)bbox.x;
        int y = (int)bbox.y;
        //int y = obj.bbox.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
};
#define OBJECT_INCLUDED
#endif