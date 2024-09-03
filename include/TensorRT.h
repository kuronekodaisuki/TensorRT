#pragma once
#ifndef __TENSORRT_H__
#define __TENSORRT_H__

#ifdef WIN32
#ifdef EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif

#ifdef WIN32
#pragma warning(disable: 4819 4251)
#endif
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


namespace nvinfer1
{
    class IBuilder;
    class IBlobNameToTensor;
    class ICudaEngine;
    class IExecutionContext;
    class INetworkDefinition;
    class IRuntime;
}

namespace nvonnxparser
{
    class IParser;
}


class API TensorRT
{
public:
    TensorRT(const char* input, const char* output);
    ~TensorRT();

    enum PRECISION {
        INT8,
        FP16,
    };

    bool ConvertModel(const char* filepath, uint width, uint height, uint channels, PRECISION precision);

    /// <summary>
    /// Load model
    /// </summary>
    /// <param name="filepath">filepath</param>
    /// <param name="width">width of model</param>
    /// <param name="height">height of model</param>
    /// <param name="channels">channels of model</param>
    /// <param name="prescision"></param>
    /// <returns>false if failed to load</returns>
    virtual bool LoadModel(const char* filepath, uint width, uint height, uint channels = 3, PRECISION precision = FP16);

    virtual bool LoadEngine(const char* filepath, uint width, uint height, uint channels = 3);
    
    void SaveEngine(const char* filepath);

    //void ShowResized(const char* title);

    cv::Size GetScaledSize() {
        return cv::Size(_width, _height);
    }

protected:
    virtual void blobFromImage(cv::Mat& image, bool bgr2rgb = true);
    virtual void imageFromBlob(cv::Mat& image, bool rgb2bgr = true);

    /// <summary>
    /// Inference
    /// </summary>
    virtual void doInference();

    bool LoadONNX(const char* filepath, uint width, uint height, uint channels, PRECISION precision);
    //bool LoadUff(const char* filepath, uint width, uint height, uint channels);

    uint _width;
    uint _height;
    uint _channels;
    int _input;
    int _output;
    const char* _input_name;
    const char* _output_name;
    float* _input_buffer;
    float* _output_buffer;
    cv::Mat _resized;
    std::vector<uint> _output_shape;

protected:
    bool _modelLoaded;
    int _output_size;

    virtual void AllocateBuffers();
    virtual void FreeBuffers();

    nvinfer1::IBlobNameToTensor* _blogToTensor = nullptr;
    nvinfer1::IBuilder* _builder = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    nvinfer1::INetworkDefinition* _network = nullptr;
    nvinfer1::IRuntime* _runtime = nullptr;  
};
#endif // __TENSORRT_H__
