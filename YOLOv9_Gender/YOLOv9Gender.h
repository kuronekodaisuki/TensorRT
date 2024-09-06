#pragma once
#ifndef YOLOv9_INCLUDED
#define YOLOv9_INCLUDED

#include <opencv2/core.hpp>

#include "../include/TensorRT.h"

#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.25f
#define NUM_CLASSES 3

static cv::Scalar colors[NUM_CLASSES] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 255) };
static const char* labels[NUM_CLASSES] = {"Male", "Female", "Unknown"};

class Object
{
public:
    cv::Rect_<float> rect;
    int label;
    float prob;

    bool operator<(const Object& right) const {
        return prob > right.prob;
    }

    void Draw(cv::Mat& image)
    {
        cv::Scalar color = colors[label];
        float c_mean = (float)cv::mean(color)[0];

        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", labels[label], prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = (int)rect.x;
        int y = (int)rect.y;
        //int y = obj.rect.y - label_size.height - baseLine;
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

typedef struct
{
    float cx;
    float cy;
    float w;
    float h;
    float scores[NUM_CLASSES];
} CHANNEL;

class YOLOv9Gender : public TensorRT
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

#endif