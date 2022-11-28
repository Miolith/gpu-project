#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

#include "CImg.h"
#include "color.hh"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

void show_components(rgba **img, vector<vector<int>> comp, int width,
                     int height, set<int> &labelSet)
{
    int slice = 300 / (labelSet.size());
    HSL rainbow(100, 1, 0.5);
    for (auto &i : labelSet)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rainbow.H = (20 + slice * i) % 360;
                if (comp[y][x] == i)
                {
                    img[y][x].red = HSLToRGB(rainbow).R;
                    img[y][x].blue = HSLToRGB(rainbow).B;
                    img[y][x].green = HSLToRGB(rainbow).G;
                }
            }
        }
    }
}

void show_componentsGPU(rgba *img, vector<vector<size_t>> labelTable, int width,
                     int height, set<size_t> &labelSet)
{
    int slice = 300 / (labelSet.size());
    HSL rainbow(100, 1, 0.5);
    for (auto &i : labelSet)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rainbow.H = (20 + slice * i) % 360;
                if (labelTable[y][x] == i)
                {
                    img[y * width + x].red = HSLToRGB(rainbow).R;
                    img[y * width + x].blue = HSLToRGB(rainbow).B;
                    img[y * width + x].green = HSLToRGB(rainbow).G;
                }
            }
        }
    }
}
std::vector<std::vector<int>>
component_box_detection(vector<vector<int>> components, int width, int height,
                        set<int> &labelSet)
{
    std::vector<vector<int>> results;
    for (auto i : labelSet)
    {
        int Xmin = width, Xmax = 0, Ymin = height, Ymax = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (components[y][x] == i)
                {
                    Xmin = min(x, Xmin);
                    Ymin = min(y, Ymin);
                    Xmax = max(x, Xmax);
                    Ymax = max(y, Ymax);
                }
            }
        }
        std::vector<int> out({ Xmin, Ymin, Xmax - Xmin, Ymax - Ymin });
        results.push_back(out);
    }
    return results;
}

std::vector<std::vector<int>>
component_box_detectionGPU(vector<vector<size_t>> components, int width, int height,
                        set<size_t> &labelSet)
{
    std::vector<vector<int>> results;
    for (auto &i : labelSet)
    {
        int Xmin = width, Xmax = 0, Ymin = height, Ymax = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (components[y][x] == i)
                {
                    Xmin = min(x, Xmin);
                    Ymin = min(y, Ymin);
                    Xmax = max(x, Xmax);
                    Ymax = max(y, Ymax);
                }
            }
        }
        std::vector<int> out({ Xmin, Ymin, Xmax - Xmin, Ymax - Ymin });
        results.push_back(out);
    }
    return results;
}


