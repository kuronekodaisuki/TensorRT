// demo.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/opencv.hpp>
#include "YOLOv9.hpp"

#define MODEL_FILE "../models/YOLOv9_s.onnx"
#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

int main(int argc, char* argv[])
{
    cv::Mat image;
    cv::VideoCapture capture;
    YOLOv9 yolo;

    switch (argc)
    {
    case 3:
        yolo.LoadModel(argv[1], MODEL_WIDTH, MODEL_HEIGHT);
        
        break;
    default:
        puts("Usage: YOLOv9 <model> <input image>");
        return 0;
    }


}
