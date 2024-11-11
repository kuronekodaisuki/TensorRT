///
///
///
#include "YOLOv8.h"

typedef struct
{
    float cx;
    float cy;
    float w;
    float h;
    float scores[NUM_CLASSES];
} CHANNEL;

namespace vitis
{
    std::unique_ptr<YOLOv8> create(const std::string& model_name, bool need_preprocess)
    {
        YOLOv8* yolo = new YOLOv8();
        yolo->LoadEngine(model_name.c_str(), 0, 0);

        return (std::unique_ptr<YOLOv8>)yolo;
    }

    YOLOv8::YOLOv8() : TensorRT("images", "output0")
    {

    }

    YOLOv8::~YOLOv8()
    {

    }
    
	YOLOv8Result YOLOv8::run(const cv::Mat& image)
	{
        YOLOv8Result results;
        std::chrono::high_resolution_clock clock;
        std::chrono::steady_clock::time_point start, stop;

        char buffer[80];
        preProcess(image);

        start = clock.now();
        doInference();
        stop = clock.now();

        sprintf(buffer, "%.2f ms", std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.0e-6);
        cv::putText(image, buffer, cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        //puts("doInferece");
        std::vector<Object> objects = postProcess(_width / (float)image.cols, _height / (float)image.rows);
        for (const Object& object : objects)
        {
            Box result = { object.label };
            result.box[0] = (int)object.rect.x;
            result.box[1] = (int)object.rect.y;
            result.box[2] = (int)(object.rect.width + object.rect.x);
            result.box[3] = (int)(object.rect.height + object.rect.y);
            results.bboxes.push_back(result);
        }
        return results;
	}

    void YOLOv8::preProcess(const cv::Mat& image)
    {
        blobFromImage(image);
    }

    std::vector<Object> YOLOv8::postProcess(float scaleX, float scaleY)
    {
        //puts("postProcess");
        _proposals.clear();
        generate_proposals(scaleX, scaleY, _bbox_confidential_threshold);

        if (2 <= _proposals.size())
        {
            std::sort(_proposals.begin(), _proposals.end());

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
        }

        return _objects;
    }

    void YOLOv8::generate_proposals(float scaleX, float scaleY, float prob_threshold)
    {
        int channels = _output_shape[1];
        int anchors = _output_shape[2];
        //printf("Channels:%d Anchors:%d\n", channels, anchors);
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
                Object object = { {left, top, channel->w / scaleX, channel->h / scaleY}, (int)(maxScorePtr - channel->scores), *maxScorePtr };
                _proposals.push_back(object);
            }
        }
        //printf("%d proposals\n", _proposals.size());
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
}