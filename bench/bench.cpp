#include <vector>
#include <benchmark/benchmark.h>
#include "../src/draw.h"
#include "../test/testTools.hpp"
#include <string.h>

#define GPU 1
#define CPU 0

constexpr int width = 1200;
constexpr int height = 800;

void pipelineCPU(rgba** ref, rgba** image, int width, int height)
{
    grayScale(image, width, height);
    gaussianBlur(image, width, height);
    dilation(image, height, width, 5);
    erosion(image, height, width, 1);
    basic_threshold(image, height, width, 100);
}

void pipelineGPU(rgba* ref, rgba* image, int width, int height)
{
    grayScaleGPU(image, width, height);
    gaussianBlurGPU(image, width, height);
    imageDiffGPU(image, ref, width, height);
    dilationGPU(image, height, width, 5);
    erosionGPU(image, width, height, 5);
    basicThresholdGPU(image, height, width, 100);
}

void BM_Rendering_cpu_1_image(benchmark::State& st)
{
    char img_ref[] = "images/001.jpg";
    char img_test[] = "images/059.jpg";
    vector<char*> data = {img_test};
    for (auto _ : st)
        findBoundingBoxes(img_ref, data.size(), data.data(), CPU);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu_1_image(benchmark::State& st)
{
    char img_ref[] = "images/001.jpg";
    char img_test[] = "images/059.jpg";
    vector<char*> data = {img_test};
    for (auto _ : st)
        findBoundingBoxes(img_ref, data.size(), data.data(), GPU);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

// benchmark with different image sizes
void BM_Pipeline_cpu_800x600(benchmark::State& st)
{
    int width = 800, height = 600;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);

    for (auto _ : st)
        pipelineCPU(ref, image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

// GPU
void BM_Pipeline_gpu_800x600(benchmark::State& st)
{
    int width = 800, height = 600;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    for (auto _ : st)
        pipelineGPU(refGPU, imageGPU, width, height);
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

// benchmark with different image sizes
void BM_Pipeline_cpu_1200x800(benchmark::State& st)
{
    int width = 1200, height = 800;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);

    for (auto _ : st)
        pipelineCPU(ref, image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

// GPU
void BM_Pipeline_gpu_1200x800(benchmark::State& st)
{
    int width = 1200, height = 800;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    for (auto _ : st)
        pipelineGPU(refGPU, imageGPU, width, height);
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

// benchmark with different image sizes
void BM_Pipeline_cpu_1600x1200(benchmark::State& st)
{
    int width = 1600, height = 1200;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);

    for (auto _ : st)
        pipelineCPU(ref, image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

// GPU
void BM_Pipeline_gpu_1600x1200(benchmark::State& st)
{
    int width = 1600, height = 1200;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    for (auto _ : st)
        pipelineGPU(refGPU, imageGPU, width, height);
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

// benchmark with different image sizes
void BM_Pipeline_cpu_1920x1080(benchmark::State& st)
{
    int width = 1920, height = 1080;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);

    for (auto _ : st)
        pipelineCPU(ref, image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

// GPU
void BM_Pipeline_gpu_1920x1080(benchmark::State& st)
{
    int width = 1920, height = 1080;
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    for (auto _ : st)
        pipelineGPU(refGPU, imageGPU, width, height);
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}


BENCHMARK(BM_Rendering_cpu_1_image)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu_1_image)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_cpu_800x600)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu_800x600)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_cpu_1200x800)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu_1200x800)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_cpu_1600x1200)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu_1600x1200)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_cpu_1920x1080)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu_1920x1080)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();