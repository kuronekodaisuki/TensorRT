///
/// YOLOX
///

#pragma once
#ifndef __YOLOX_H__
#define __YOLOX_H__

#include "../include/TensorRT.h"
#include "../include/Object.h"

#define NMS_THRESH 0.45f
#define BBOX_CONF_THRESH 0.3f

class API YOLOX : public TensorRT
{
public:
    /// <summary>
    /// Constructor
    /// </summary>
    YOLOX(int batch_size = 1);
    ~YOLOX();

    /// <summary>
    /// Load model function overrided for supplimental operation
    /// </summary>
    /// <param name="filepath"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="channels"></param>
    /// <param name="precision"></param>
    /// <returns></returns>
    bool LoadModel(const char* filepath, uint width, uint height, uint channels = 3, PRECISION precision = FP16);

    /// <summary>
    /// Load engine(GPU specified model) function overrided for suplimental operation
    /// </summary>
    /// <param name="filepath"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="channels"></param>
    /// <returns></returns>
    bool LoadEngine(const char* filepath, uint width, uint height, uint channels = 3);

    /// <summary>
    /// Set threshold for NMS and conficence
    /// </summary>
    /// <param name="nms_thres"></param>
    /// <param name="bbox_conf_thres"></param>
    void SetThresholds(float bbox_conf_thres = BBOX_CONF_THRESH, float nms_thres = NMS_THRESH);

    /// <summary>
    /// Detect objects
    /// </summary>
    /// <param name="image"></param>
    /// <returns>Detected objects</returns>
    std::vector<Object> Detect(cv::Mat image);

    std::vector<Object> DetectBatch(cv::Mat image, int maxBatchSize = 2, bool bgr2rgb = true);

protected:
    void blobFromImage(cv::Mat& image, bool bgr2rgb = true);
    std::vector<Object> postProcess(const int image_w, const int image_h, float scaleX, float scaleY, float* output);
    void generate_yolox_proposals(float prob_threshold, float* output);

private:
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;

        GridAndStride(int g0, int g1, int s) :grid0(g0), grid1(g1), stride(s) {}
    };

    float _nms_threshold = NMS_THRESH;
    float _bbox_confidential_threshold = BBOX_CONF_THRESH;
    uint _numClasses;

    std::vector<Object> _proposals;
    std::vector<Object> _objects;

    std::vector<GridAndStride> _grid_strides;
    std::vector<GridAndStride> generate_grids_and_stride();

    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
};
#endif // __YOLOX_H__
