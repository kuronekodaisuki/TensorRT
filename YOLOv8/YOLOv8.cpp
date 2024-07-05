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
#ifdef _DEBUG
    printf("%ld %ld\n", _proposals.size(), _objects.size());
#endif
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
            Object object = { {left, top, channel->w / scaleX, channel->h / scaleY}, (int)(maxScorePtr - channel->scores), *maxScorePtr};
            _proposals.push_back(object);
        }
    }
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

