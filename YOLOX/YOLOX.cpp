#include "YOLOX.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

// MSVC defines this in winsock2.h!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec = (long)((time - EPOCH) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#endif
static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

typedef struct
{
    float cx;
    float cy;
    float w;
    float h;
    float prob;
    float scores[NUM_CLASSES];
} CHANNEL;

bool YOLOX::Initialize(const char* model_path, int model_width, int model_height)
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

std::vector<Object> YOLOX::Detect(cv::Mat image)
{
    timeval start, stop;
    char buffer[80];
    preProcess(image);

    gettimeofday(&start, NULL);
    doInference("images", "output");
    gettimeofday(&stop, NULL);
    sprintf(buffer, "%.2f ms", (__get_us(stop) - __get_us(start)) / 1000);
    cv::putText(image, buffer, cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    //puts("doInferece");
    return postProcess(_width / (float)image.cols, _height / (float)image.rows);
}

void YOLOX::preProcess(cv::Mat& image)
{
    blobFromImage(image);
}

std::vector<Object> YOLOX::postProcess(float scaleX, float scaleY)
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

void YOLOX::generate_proposals(float scaleX, float scaleY, float prob_threshold)
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

std::vector<int> YOLOX::nms(float nms_threshold)
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

