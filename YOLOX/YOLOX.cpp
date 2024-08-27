#include "YOLOX.h"


#define INPUT_BLOB_NAME "images"
#define OUTPUT_BLOB_NAME "output"

YOLOX::YOLOX(int batch_size): TensorRT(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, batch_size)
{
}

YOLOX::~YOLOX()
{

}

bool YOLOX::LoadModel(const char* filepath, uint width, uint height, uint channels, PRECISION precision)
{
    if (TensorRT::LoadModel(filepath, width, height, channels, precision))
    {
        _grid_strides = generate_grids_and_stride();
        return true;
    }
    else
        return false;
}

bool YOLOX::LoadEngine(const char* filepath, uint width, uint height, uint channels)
{
    if (TensorRT::LoadEngine(filepath, width, height, channels))
    {
        _grid_strides = generate_grids_and_stride();
        return true;
    }
    else
        return false;
}

/// <summary>
/// Inference
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
std::vector<Object> YOLOX::Detect(cv::Mat image)
{
    blobFromImage(image);

    doInference();

    return _objects = postProcess(_width, _height, _width / (float)image.cols, _height / (float)image.rows, _output_buffer);
}

std::vector<Object> YOLOX::DetectBatch(cv::Mat image, int batch_size, bool bgr2rgb)
{
    //setMaxBatchSize(batch_size);
    float nX = (float)image.cols / _width;
    float nY = (float)image.rows / _height;
    if (nX <= 1 && nY <= 1)
    {
        return Detect(image);
    }
    else
    {
        std::vector<cv::Rect> rois;
        rois.push_back(cv::Rect(0, 0, _width, _height));
        rois.push_back(cv::Rect(image.cols - _width, 0, _width, _height));
        rois.push_back(cv::Rect(0, image.rows - _height, _width, _height));
        rois.push_back(cv::Rect(image.cols - _width, image.rows - _height, _width, _height));

        float* input = _input_buffer;
        for (int batch = 0; batch < batch_size; batch++)
        {
            cv::Mat roi(image, rois[batch]);
            if (bgr2rgb)
            {
                cv::cvtColor(roi, _resized, cv::COLOR_BGR2RGB);
            }
            else
            {
                roi.copyTo(_resized);
            }

            for (uint c = 0; c < _channels; c++)
            {
                for (uint h = 0; h < _height; h++)
                {
                    for (uint w = 0; w < _width; w++)
                    {
                        input[h * _width + w] = (float)_resized.at<cv::Vec3b>(h, w)[c];
                    }
                }
                input += _width * _height;
            }
        }

        doInference();

        float* output = _output_buffer;
        for (int batch = 0; batch < batch_size; batch++)
        {
            char buffer[60];
            std::vector<Object> objects = postProcess(_width, _height, 1, 1, output);
            printf("batch:%d %ld objects\n", batch, objects.size());
            for (int i = 0; i < objects.size(); i++)
            {
                objects[i].Draw(image, rois[batch].x, rois[batch].y);
            }
            sprintf(buffer, "batch%d.png", batch);
            cv::imwrite(buffer, image);
            output += _output_size;
        }

        return _objects;
    }
}

/// <summary>
/// Convert image to tensor(1, channels, width, height)
/// </summary>
/// <param name="image"></param>
void YOLOX::blobFromImage(cv::Mat& image, bool bgr2rgb)
{
    cv::resize(image, _resized, cv::Size(_width, _height));
    if (bgr2rgb)
    {
        cv::cvtColor(_resized, _resized, cv::COLOR_BGR2RGB);
    }

    float* input = _input_buffer;
    for (uint c = 0; c < _channels; c++)
    {
        for (uint h = 0; h < _height; h++)
        {
            for (uint w = 0; w < _width; w++)
            {
                input[h * _width + w] = (float)_resized.at<cv::Vec3b>(h, w)[c];
            }
        }
        input += _width * _height;
    }
}

/// <summary>
/// Decode inference results
/// </summary>
/// <param name="prob"></param>
/// <param name="scale"></param>
/// <param name="image_w">Width of input image</param>
/// <param name="image_h">Height of input image</param>
std::vector<Object> YOLOX::postProcess(const int width, const int height, float scaleX, float scaleY, float* output)
{
    std::vector<Object> objects;
    _proposals.clear();
    generate_yolox_proposals(_bbox_confidential_threshold, output);

    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }

    std::vector<int> picked;
    nms_sorted_bboxes(_proposals, picked, _nms_threshold);

    size_t count = picked.size();

    objects.resize(count);
    for (size_t i = 0; i < count; i++)
    {
        objects[i] = _proposals[picked[i]];

        objects[i].rect.x /= scaleX;
        objects[i].rect.y /= scaleY;
        objects[i].rect.width /= scaleX;
        objects[i].rect.height /= scaleY;
    }
    return objects;
}

void YOLOX::generate_yolox_proposals(float prob_threshold, float* output)
{
    const size_t num_anchors = _grid_strides.size();

    for (size_t anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = _grid_strides[anchor_idx].grid0;
        const int grid1 = _grid_strides[anchor_idx].grid1;
        const int stride = _grid_strides[anchor_idx].stride;

        const size_t offset = anchor_idx * (_numClasses + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (output[offset + 0] + grid0) * stride;
        float y_center = (output[offset + 1] + grid1) * stride;
        float w = exp(output[offset + 2]) * stride;
        float h = exp(output[offset + 3]) * stride;
        float box_objectness = output[offset + 4];

        float x0 = x_center - w / 2;
        float y0 = y_center - h / 2;

        cv::Rect_<float> bound(x0, y0, w, h);

        for (size_t class_idx = 0; class_idx < _numClasses; class_idx++)
        {
            float box_cls_score = output[offset + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (prob_threshold < box_prob)
            {
                Object obj;
                obj.rect = bound;
                obj.label = (int)class_idx;
                obj.prob = box_prob;

                _proposals.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

std::vector<YOLOX::GridAndStride> YOLOX::generate_grids_and_stride()
{
    std::vector<int> strides = { 8, 16, 32 };

    std::vector<GridAndStride> grid_strides;
    for (auto stride : strides)
    {
        int num_grid_y = _height / stride;
        int num_grid_x = _width / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back(GridAndStride(g0, g1, stride));
            }
        }
    }
    _numClasses = _output_shape[2] - 5;

    return grid_strides;
}

void YOLOX::nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (size_t i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (nms_threshold < inter_area / union_area)
                keep = 0;
        }

        if (keep)
            picked.push_back((int)i);
    }
}

/// <summary>
/// Set thresholds
/// </summary>
/// <param name="bbox_conf_thres"></param>
/// <param name="nms_thres"></param>
void YOLOX::SetThresholds(float bbox_conf_thres, float nms_thres)
{
    _bbox_confidential_threshold = bbox_conf_thres;
    _nms_threshold = nms_thres;
}

