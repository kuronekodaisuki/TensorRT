// Vitis-AI.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/opencv.hpp>
#include "YOLOv8.h"

const std::string MODEL = "../models/yolov8s.engine";


int main(int argc, char* argv[])
{
    vitis::YOLOv8 yolo;
    if (yolo.create(MODEL))
    {
        cv::Mat image;
        if (2 <= argc)
        {
            image = cv::imread(argv[1]);
            vitis::YOLOv8Result results = yolo.run(image);
            for (const auto& result : results.bboxes)
            {
                int label = result.label;
                auto& box = result.box;
                cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), cv::Scalar());
                cv::imshow("result", image);
                cv::waitKey(0);
            }
        }
        puts(MODEL.c_str());
    }
}

