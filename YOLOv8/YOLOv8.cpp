// YOLOv8.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//


#include "YOLOv8.h"

YOLOv8::YOLOv8()
{

}

YOLOv8::~YOLOv8()
{

}

bool YOLOv8::Initialize(const char* model_path, int model_width, int model_height)
{
	if (LoadModel(model_path, model_width, model_height))
	{
		return true;
	}
	else
	{
		printf("Failed to load %s\n", model_path);
		return false;
	}
}

std::vector<Object> YOLOv8::Detect(cv::Mat image)
{
	preProcess(image);

	doInference();

	return postProcess(_width / (float)image.cols, _height / (float)image.rows);
}

void YOLOv8::preProcess(cv::Mat& image)
{
	blobFromImage(image);
}

std::vector<Object> YOLOv8::postProcess(float scaleX, float scaleY)
{
    _proposals.clear();
    generate_proposals(scaleX, scaleY, _bbox_confidential_threshold);

    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }

    std::vector<int> picked = nms(_nms_threshold);

    size_t count = picked.size();

    _objects.resize(count);
    for (size_t i = 0; i < count; i++)
    {
        _objects[i] = _proposals[picked[i]];
    }
    printf("%ld %ld\n", _proposals.size(), _objects.size());
	return _objects;
}

void YOLOv8::generate_proposals(float scaleX, float scaleY, float prob_threshold)
{
    int channels = _output_shape[1];
    int anchors = _output_shape[2];
    cv::Mat output = cv::Mat(channels, anchors, CV_32F, _output);
    output = output.t();

    for (int i = 0; i < anchors; i++)
    {
        CHANNEL* channel = (CHANNEL*)output.row(i).ptr<float>();
        float* maxScorePtr = std::max_element(channel->scores, channel->scores + _numClasses);
        if (_bbox_confidential_threshold < *maxScorePtr)
        {
            float left = (channel->cx - channel->w / 2) / scaleX;
            float top = (channel->cy - channel->h / 2) / scaleY;
            Object object = { {left, top, channel->w / scaleX, channel->h / scaleY}, maxScorePtr - channel->scores, *maxScorePtr};
            _proposals.push_back(object);
        }
    }

    /*
    const size_t num_anchors = _grid_strides.size();

    for (size_t anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = _grid_strides[anchor_idx].grid0;
        const int grid1 = _grid_strides[anchor_idx].grid1;
        const int stride = _grid_strides[anchor_idx].stride;

        const size_t offset = anchor_idx * (_numClasses + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (_output[offset + 0] + grid0) * stride;
        float y_center = (_output[offset + 1] + grid1) * stride;
        float w = exp(_output[offset + 2]) * stride;
        float h = exp(_output[offset + 3]) * stride;
        float box_objectness = _output[offset + 4];

        float x0 = x_center - w / 2;
        float y0 = y_center - h / 2;

        cv::Rect_<float> bound(x0, y0, w, h);

        for (size_t class_idx = 0; class_idx < _numClasses; class_idx++)
        {
            float box_cls_score = _output[offset + 5 + class_idx];
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
    */
}

std::vector<int> YOLOv8::nms(float nms_threshold)
{
    std::vector<int> picked;

    const size_t n = _proposals.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        areas[i] = _proposals[i].rect.area();
    }

    for (size_t i = 0; i < n; i++)
    {
        const Object& a = _proposals[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = _proposals[picked[j]];

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
    return picked;
}

std::vector<YOLOv8::GridAndStride> YOLOv8::generate_grids_and_stride()
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
