// Vitis-AI.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/opencv.hpp>
#include "YOLOv8.h"

const std::string MODEL = "../models/yolov8s.engine";


int main(int argc, char* argv[])
{
    std::unique_ptr<vitis::YOLOv8> yolo = vitis::YOLOv8::create(MODEL);
}

