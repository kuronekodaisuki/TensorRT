///
/// TensorRT inference class 
///

#include <fstream>
#include <iostream>
#include <sstream>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "TensorRT.h"
#include "logging.h"

using namespace nvinfer1;


#define DEVICE 0  // GPU id

Logger logger;

/// <summary>
/// Constructor
/// </summary>
TensorRT::TensorRT(const char* input, const char* output) : _input_name(input), _output_name(output), _input(0), _output(1), _input_buffer(nullptr), _output_buffer(nullptr)
{
    _modelLoaded = false;
    cudaSetDevice(0);
}

/// <summary>
/// Destructor
/// </summary>
TensorRT::~TensorRT()
{
    if (_modelLoaded)
    {
        FreeBuffers();
        delete _context;
        delete _engine;
        delete _runtime;
    }
}

bool TensorRT::ConvertModel(const char* filepath, uint width, uint height, uint channels, PRECISION precision)
{
    if (LoadModel(filepath, width, height, channels, precision))
    {
        std::string savepath(filepath);
        size_t period = savepath.find_last_of('.');
        savepath.replace(period, savepath.length() - period, ".engine");
        
        SaveEngine(savepath.c_str());
        return true;
    }
    return false;   
}

void TensorRT::AllocateBuffers()
{
    if (_input_name != nullptr)
    {
        _input = _engine->getBindingIndex(_input_name);
    }
    else
    {
        _input_name = _engine->getBindingName(_input);
    }
    if (_output_name != nullptr)
    {
        _output = _engine->getBindingIndex(_output_name);
    }
    else
    {
        _output_name = _engine->getBindingName(_output);
    }

#ifdef _DEBUG
    for (int i = 0; i < _engine->getNbIOTensors(); i++)
    {
        const char* name = _engine->getBindingName(i);
        const char* desc = _engine->getBindingFormatDesc(i);
        Dims dimension = _engine->getTensorShape(name);
        const char* type = "";
        switch (_engine->getTensorDataType(name))
        {
        case DataType::kFLOAT:
            type = "FP32";
            break;
        case DataType::kHALF:
            type = "FP16";
            break;
        case DataType::kINT8:
            type = "INT8";
            break;
        case DataType::kINT32:
            type = "INT32";
            break;
        case DataType::kUINT8:
            type = "UINT8";
            break;
        case DataType::kBOOL:
            type = "BOOL";
            break;
        }
        TensorFormat format = _engine->getBindingFormat(i);
        printf("Binding[%d], %s %s[%d]\n", i, name, type, dimension.nbDims);
        for (int j = 0; j < dimension.nbDims; j++)
        {
            printf("\t%d\n", dimension.d[j]);
        }
    }
#endif

    auto dimensions = _engine->getBindingDimensions(1);
#ifdef _DEBUG
    printf("%d dimensions\n", dimensions.nbDims);
#endif
    _output_size = 1;
    for (int j = 0; j < dimensions.nbDims; j++)
    {
        _output_size *= dimensions.d[j];
        _output_shape.push_back(dimensions.d[j]);
#ifdef _DEBUG
        printf("%d size:%d\n", j, dimensions.d[j]);
#endif
    }
    _output_buffer = new float[_output_size];
    _input_buffer = new float[_width * _height * _channels];
    _resized.create(_height, _width, CV_8UC3);

}

void TensorRT::FreeBuffers()
{
    _resized.release();
    delete[] _input_buffer;
    delete[] _output_buffer;
}

bool TensorRT::LoadModel(const char* filepath, uint width, uint height, uint channels, PRECISION precision)
{
    _width = width;
    _height = height;
    _channels = channels;

    std::ifstream file(filepath, std::ios::binary);
    if (file.good()) 
    {
        file.close();
        try
        {
            int verbosity = 4;
            NetworkDefinitionCreationFlags flags = 1;
            // Create Instances
            _builder = createInferBuilder(logger);
            _runtime = createInferRuntime(logger);
            _network = _builder->createNetworkV2(flags);

#ifdef _DEBUG
            printf("Loading %s\n", filepath);
#endif
            nvonnxparser::IParser* parser = nvonnxparser::createParser(*_network, logger);

            if (parser->parseFromFile(filepath, verbosity))
            {
                IBuilderConfig* config = _builder->createBuilderConfig();
                switch (precision)
                {
                case INT8:
                    config->setFlag(BuilderFlag::kINT8);
                    puts("INT8");
                    break;
                case FP16:
                    config->setFlag(BuilderFlag::kFP16);
                    puts("FP16");
                    break;
                }

                _engine = _builder->buildEngineWithConfig(*_network, *config);
                //_engine = _builder->buildSerializedNetwork(*_network, *config);
                _context = _engine->createExecutionContext();
                _builder->setMaxBatchSize(2);

                AllocateBuffers();
                return _modelLoaded = true;
            }
        }
        catch (std::exception e)
        {
            return false;
        }
    }
    return false;
}

static size_t filesize(const char* filepath)
{
    size_t size = 0;
    std::fstream motd(filepath, std::ios::binary | std::ios::in | std::ios::ate);
    if (motd) 
    {
        size = motd.tellg();
    }
    return size;
}

bool TensorRT::LoadEngine(const char* filepath, uint width, uint height, uint channels)
{
    _width = width;
    _height = height;
    _channels = channels;

    size_t size = filesize(filepath);
    
    // If failed to access engine, Throw exception

    char* buffer = new char[size];

    std::ifstream file(filepath, std::ios::binary);
    file.read(buffer, size);
    _runtime = createInferRuntime(logger);
    _engine = _runtime->deserializeCudaEngine(buffer, size);
    delete[] buffer;

    if (_engine != nullptr)
    {
        auto dimensions = _engine->getBindingDimensions(1);

        AllocateBuffers();

        _context = _engine->createExecutionContext();

        return _modelLoaded = true;
    }
    else
        return false;
}

void TensorRT::SaveEngine(const char* filepath)
{
    IHostMemory* memory = _engine->serialize();
    std::ofstream file(filepath,  std::ios::binary);
    file.write((const char*)memory->data(), memory->size());
    file.close();
}

bool TensorRT::LoadONNX(const char* filepath, uint width, uint height, uint channels, PRECISION precision)
{
    int verbosity = 4;
    NetworkDefinitionCreationFlags flags = 1;

    _width = width;
    _height = height;
    _channels = channels;

    // Create Instances
    _builder = createInferBuilder(logger);
    _runtime = createInferRuntime(logger);
    _network = _builder->createNetworkV2(flags);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*_network, logger);
    if (parser->parseFromFile(filepath, verbosity))
    {
        IBuilderConfig* config = _builder->createBuilderConfig();
        switch (precision)
        {
        case INT8:
            config->setFlag(BuilderFlag::kINT8);
            break;
        case FP16:
            config->setFlag(BuilderFlag::kFP16);
            break;
        }

        _engine = _builder->buildEngineWithConfig(*_network, *config);
        _context = _engine->createExecutionContext();

        AllocateBuffers();

        return _modelLoaded = true;
    }
    return false;
}

/// <summary>
/// Convert image to tensor(1, channels, width, height)
/// </summary>
/// <param name="image"></param>
void TensorRT::blobFromImage(const cv::Mat& image, bool bgr2rgb)
{
    cv::resize(image, _resized, cv::Size(_width, _height));
    if (bgr2rgb)
    {
        cv::cvtColor(_resized, _resized, cv::COLOR_BGR2RGB);
    }

    for (uint c = 0; c < _channels; c++)
    {
        for (uint h = 0; h < _height; h++)
        {
            for (uint w = 0; w < _width; w++)
            {
                _input_buffer[c * _width * _height + h * _width + w] = (float)_resized.at<cv::Vec3b>(h, w)[c] / 255;
            }
        }
    }
}

/*
void TensorRT::ShowResized(const char* title)
{
	cv::imshow(title, _resized);
}
*/

/// <summary>
/// Convert image from tensor(1, channels, width, height)
/// </summary>
/// <param name="image"></param>
/// <param name="rgb2bgr"></param>
void TensorRT::imageFromBlob(cv::Mat& image, bool rgb2bgr)
{
    for (uint c = 0; c < _channels; c++)
    {
        for (uint h = 0; h < _height; h++)
        {
            for (uint w = 0; w < _width; w++)
            {
                image.at<cv::Vec3b>(h, w)[c] = cv::saturate_cast<uchar>(_output_buffer[c * _width * _height + h * _width + w] * 255);
            }
        }
    }
    if (rgb2bgr)
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
}

/// <summary>
/// Inference
/// </summary>
void TensorRT::doInference()
{
    assert(_engine->getNbBindings() >= 2);
    void* buffers[2];

    assert(_engine->getTensorDataType(_input_name) == DataType::kFLOAT);
    assert(_engine->getTensorDataType(_output_name) == DataType::kFLOAT);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[_input], _channels * _height * _width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[_output], _output_size * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[_input], _input_buffer, _channels * _height * _width * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    (_context->enqueueV2(buffers, stream, nullptr));

    CHECK(cudaMemcpyAsync(_output_buffer, buffers[_output], _output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);

    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
}

