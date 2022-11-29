#include <vector>
#include <benchmark/benchmark.h>
#include "../src/draw.h"
#include "../test/testTools.hpp"

#define GPU 1
#define CPU 0

constexpr int width = 1200;
constexpr int height = 800;

// test each pipeline step
void BM_GrayScale_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    for (auto _ : st)
        grayScale(image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

void BM_GaussianBlur_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    grayScale(image, width, height);
    for (auto _ : st)
        gaussianBlur(image, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

void BM_Dilation_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    for (auto _ : st)
        dilation(image, height, width, 10);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

void BM_Erosion_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    for (auto _ : st)
        erosion(image, height, width, 10);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

void BM_Threshold_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    for (auto _ : st)
        basic_threshold(image, height, width, 100);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

// component labeling
void BM_ComponentLabeling_cpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    
    grayScale(image, width, height);
    gaussianBlur(image, width, height);
    dilation(image, height, width, 5);
    erosion(image, height, width, 1);
    basic_threshold(image, height, width, 70);

    for (auto _ : st)
    {
        set<int> labels;
        connectCompenent(image, height, width, labels);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
}

void BM_GrayScale_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);
    for (auto _ : st)
        grayScaleGPU(imageGPU, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

void BM_GaussianBlur_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);
    grayScaleGPU(imageGPU, width, height);
    for (auto _ : st)
        gaussianBlurGPU(imageGPU, width, height);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

void BM_Dilation_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);
    for (auto _ : st)
        dilationGPU(imageGPU, width, height, 10);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

void BM_Erosion_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);
    for (auto _ : st)
        erosionGPU(imageGPU, height, width, 10);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

void BM_Threshold_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);
    for (auto _ : st)
        basicThresholdGPU(imageGPU, height, width, 100);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}

// component labeling
void BM_ComponentLabeling_gpu(benchmark::State& st)
{
    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);


    grayScaleGPU(imageGPU, width, height);
    gaussianBlurGPU(imageGPU, width, height);
    imageDiffGPU(imageGPU, refGPU, width, height);
    dilationGPU(imageGPU, height, width, 5);
    erosionGPU(imageGPU, width, height, 5);
    basicThresholdGPU(imageGPU, height, width, 70);
    for (auto _ : st)
    {
        set<size_t> labels;
        connectCompenentGPU(imageGPU, height, width, labels);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
    unloadImage(image, height);
    unloadImage(ref, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
    unloadImageGPU(refGPU);
}


BENCHMARK(BM_GrayScale_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_GaussianBlur_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Dilation_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Erosion_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Threshold_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_ComponentLabeling_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_GrayScale_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_GaussianBlur_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Dilation_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Erosion_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Threshold_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_ComponentLabeling_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
