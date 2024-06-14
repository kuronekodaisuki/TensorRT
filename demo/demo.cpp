// demo.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/opencv.hpp>
#include "../include/YOLOX.h"

int main(int argc, char* argv[])
{
    cv::Mat image;
    cv::VideoCapture capture;
    YOLOX yolox;

    switch (argc)
    {
    case 3:
        capture.open(argv[2]);
        yolox.LoadEngine(argv[1], 640, 640);
    }
}

