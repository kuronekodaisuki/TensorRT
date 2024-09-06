// YOLOX.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <opencv2/opencv.hpp>
#ifdef _MSC_VER
// Windows
#include <Windows.h>
/// return a timestamp with sub-second precision
/** QueryPerformanceCounter and clock_gettime have an undefined starting point (null/zero)
	and can wrap around, i.e. be nulled again. **/
double seconds()
{
	static LARGE_INTEGER frequency;
	if (frequency.QuadPart == 0)
		::QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER now;
	::QueryPerformanceCounter(&now);
	return now.QuadPart / double(frequency.QuadPart);
}
#else
// Linux

#include <time.h>
double seconds()
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec + now.tv_nsec / 1000000000.0;
}
#endif

#include "YOLOX.h"
#include "../include/Object.h"

const char* MODEL = "../models/yolox_m_1088x1920.onnx";
const char* ENGINE = "../models/yolox_m_188x1920.trt";
const int MODEL_WIDTH = 1920;
const int MODEL_HEIGHT = 1088;


#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
bool FileExists(const char* path) { return fs::exists(path); }
#else
#include <sys/stat.h>
bool FileExists(const char* name)
{
	struct stat   buffer;
	return (stat(name, &buffer) == 0);
}
#endif

int main(int argc, char* argv[])
{
	cv::Mat image;
	YOLOX yolo;
	if (FileExists(ENGINE))
	{
		yolo.LoadEngine(ENGINE, MODEL_WIDTH, MODEL_HEIGHT);
	}
	else
	{
		if (yolo.LoadModel(MODEL, MODEL_WIDTH, MODEL_HEIGHT))
		{
			yolo.SaveEngine(ENGINE);
		}
	}

	if (2 <= argc)
	{
		puts(argv[1]);
		image = cv::imread(argv[1]);
		double start, stop;

		start = seconds();
		std::vector<Object> objects = yolo.Detect(image);
		stop = seconds();

		printf("Finish %f s to perform YOLOX\n", stop - start);

		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].Draw(image);
		}
		cv::imshow("out", image);

		switch (cv::waitKey(0))
		{
		case ' ':
			cv::imwrite("out.png", image);
			break;
		}
	}

}

