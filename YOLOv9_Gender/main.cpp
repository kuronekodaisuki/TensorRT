﻿// YOLOv9_Gender.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <opencv2/opencv.hpp>
#include "YOLOv9Gender.h"

const char* MODEL = "../models/yolov9_s_gender_0200_1x3x480x640.onnx";
const char* ENGINE = "../models/yolov9_s_gender_0200_1x3x480x640.engine";
const int MODEL_WIDTH = 640;
const int MODEL_HEIGHT = 480;

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
	YOLOv9Gender yolo;
	if (FileExists(ENGINE))
	{
		yolo.LoadEngine(ENGINE, MODEL_WIDTH, MODEL_HEIGHT);
	}
	else
	{
		if (yolo.Initialize(MODEL, MODEL_WIDTH, MODEL_HEIGHT))
		{
			yolo.SaveEngine(ENGINE);
		}
	}

	cv::Mat image;
	cv::VideoCapture camera;
	if (2 <= argc)
	{
		if (camera.open(argv[1]))
		{
			//camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
			for (bool loop = true; loop;)
			{
				try // for prevent rtsp decode error
				{
					if (camera.read(image))
					{
						std::vector<Object> objects = yolo.Detect(image);
						for (int i = 0; i < objects.size(); i++)
						{
							objects[i].Draw(image);
						}
						cv::imshow("out", image);

						switch (cv::waitKey(1))
						{
						case 'q':
							loop = false;
							break;

						case ' ':
							cv::imwrite("out.png", image);
							break;
						}
					}
				}
				catch (const std::exception& e)
				{
					std::cerr << e.what() << '\n';
				}
			}
		}
	}
	else
	{
		if (camera.open(0))
		{
			//camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
			for (bool loop = true; loop && camera.read(image);)
			{
				std::vector<Object> objects = yolo.Detect(image);
				for (int i = 0; i < objects.size(); i++)
				{
					objects[i].Draw(image);
				}
				cv::imshow("out", image);

				switch (cv::waitKey(1))
				{
				case 'q':
					loop = false;
					break;

				case ' ':
					cv::imwrite("out.png", image);
					break;
				}
			}
		}
	}

}