#include <vector>
#include <benchmark/benchmark.h>
#include "../src/draw.h"
#include "../test/testTools.hpp"

#define GPU 1
#define CPU 0

constexpr int width = 1200;
constexpr int height = 800;


void findBoxGPUBench(benchmark::State& st)
{
    st.PauseTiming();
    vector<int> func_call_count = {0,0,0,0,0,0,0,0,0,0,0};

    char img_ref[] = "images/001.jpg";
    char img_test[] = "images/059.jpg";
    vector<char*> data = {img_test};
    char *reference = img_ref;
    int count = data.size();
    char **images = data.data();
    boxMap boxes;
    int h, w;

    rgba* refGPU;
    refGPU = loadReferenceGPU(reference, &w, &h);
    rgba* ref = refGPU;
    st.ResumeTiming();
    for (auto _ : st)
    {
        for(int i = 0; i < count; i++)
        {
                boxList box;
                int width, height;

                rgba *img = loadImageGPU(images[i], &width, &height);
                func_call_count[0] += 1;

                grayScaleGPU(img, width, height);
                func_call_count[1] += 1;

                gaussianBlurGPU(img, width, height);
                func_call_count[2] += 1;

                imageDiffGPU(ref, img, width, height);
                func_call_count[3] += 1;

                // CLOSING
                dilationGPU(img, width, height, 20);
                func_call_count[4] += 1;

                erosionGPU(img, width, height, 20);
                func_call_count[5] += 1;

                // OPENING
                erosionGPU(img, width, height, 50);
                func_call_count[5] += 1;

                dilationGPU(img, width, height, 50);
                func_call_count[4] += 1;

                basicThresholdGPU(img, height, width, 70);
                func_call_count[6] += 1;

                set<size_t> label_list;
                vector<vector<size_t>> labels = connectCompenentGPU(img, height, width, label_list);
                func_call_count[7] += 1;
                show_componentsGPU(img, labels, width, height, label_list);
                func_call_count[8] += 1;

                for(auto boxe: component_box_detectionGPU(labels, width, height, label_list))
                {
                    box.push_back(boxe);
                    func_call_count[9] += 1;
                }
                unloadImageGPU(img);
                func_call_count[10] += 1;
                unloadImageGPU(ref);
                func_call_count[10] += 1;

                boxes[images[i]] = box;
        }
    }


    st.counters["\nloadImageGPU"] = benchmark::Counter(func_call_count[0], benchmark::Counter::kIsRate);
    st.counters["\ngrayScaleGPU"] = benchmark::Counter(func_call_count[1], benchmark::Counter::kIsRate);
    st.counters["\ngaussianBlurGPU"] = benchmark::Counter(func_call_count[2], benchmark::Counter::kIsRate);
    st.counters["\nimageDiffGPU"] = benchmark::Counter(func_call_count[3], benchmark::Counter::kIsRate);
    st.counters["\ndilationGPU"] = benchmark::Counter(func_call_count[4], benchmark::Counter::kIsRate);
    st.counters["\nerosionGPU"] = benchmark::Counter(func_call_count[5], benchmark::Counter::kIsRate);
    st.counters["\nbasicThresholdGPU"] = benchmark::Counter(func_call_count[6], benchmark::Counter::kIsRate);
    st.counters["\nconnectCompenentGPU"] = benchmark::Counter(func_call_count[7], benchmark::Counter::kIsRate);
    st.counters["\nshow_componentsGPU"] = benchmark::Counter(func_call_count[8], benchmark::Counter::kIsRate);
    st.counters["\ncomponent_box_detectionGPU"] = benchmark::Counter(func_call_count[9], benchmark::Counter::kIsRate);
    st.counters["\nunloadImageGPU"] = benchmark::Counter(func_call_count[10], benchmark::Counter::kIsRate);
}


BENCHMARK(findBoxGPUBench)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
