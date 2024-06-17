// demo.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/opencv.hpp>
#include "../include/YOLOX.h"

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
bool FileExists(const char* path) { return fs::exists(path); }
#else
#include <sys/stat.h>
bool FileExists(const char *name)
{
  struct stat   buffer;
  return (stat (name, &buffer) == 0);
}
#endif

#define ENGINE_FILE "../models/yolox_s.engine"
#define MODEL_FILE "../models/yolox_s.onnx"
#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

int main(int argc, char* argv[])
{
    cv::Mat image;
    cv::VideoCapture capture;

    YOLOX yolo;

    switch (argc)
    {
    case 3:
        if (!FileExists(ENGINE_FILE))
        {
            printf("Convert ONNX to Engine: %s\n", MODEL_FILE);
            if (yolo.LoadModel(argv[1], MODEL_WIDTH, MODEL_HEIGHT))
            {
                yolo.SaveEngine(ENGINE_FILE);
            }
        }
        else
        {
            yolo.LoadEngine(ENGINE_FILE, MODEL_WIDTH, MODEL_HEIGHT);
        }
        break;
    default:
        puts("Usage: YOLOv9 <model> <input image>");
        return 0;
    }

    if (capture.open(argv[2]))
    {
        for (bool loop = true; loop && capture.read(image); )
        {
            yolo.Detect(image);
            switch (cv::waitKey(1))
            {
            case 'q':
                loop = false;
                break;

            case ' ':
                break;
            }
        }
    }
}

