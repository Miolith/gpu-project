#include <vector>
#include <benchmark/benchmark.h>
#include "../src/draw.h"
#include "../test/testTools.hpp"

#define GPU 1
#define CPU 0

constexpr int width = 1200;
constexpr int height = 800;

void BM_Rendering_cpu(benchmark::State& st)
{
    char img_ref[] = "images/001.jpg";
    char img_test[] = "images/059.jpg";
    vector<char*> data = {img_test};
    for (auto _ : st)
        findBoundingBoxes(img_ref, data.size(), data.data(), CPU);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State& st)
{
    char img_ref[] = "images/001.jpg";
    char img_test[] = "images/059.jpg";
    vector<char*> data = {img_test};
    for (auto _ : st)
        findBoundingBoxes(img_ref, data.size(), data.data(), GPU);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();