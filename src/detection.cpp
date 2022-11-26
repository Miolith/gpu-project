#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "color.hh"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

std::vector<int> basic_box_detection(rgba **image, int width, int height)
{
    int Xmin = 255, Xmax = 0, Ymin = 255, Ymax = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[y][x].red == 255)
            {
                if (Xmin > x)
                    Xmin = x;
                if (Ymin > y)
                    Ymin = y;
                if (Xmax < x)
                    Xmax = x;
                if (Ymax < y)
                    Ymax = y;
            }
        }
    }
    return std::vector<int>({ Xmin, Ymin, Xmax - Xmin, Ymax - Ymin });
}

void show_components(rgba **img, vector<vector<int>> comp, int width,
                     int height, int components_nb)
{
    int slice = 360 / (components_nb);
    HSL rainbow(100, 1, 0.5);
    for (int i = 1; i <= components_nb; i++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rainbow.H = slice * i;
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

std::vector<std::vector<int>>
component_box_detection(vector<vector<int>> components, int width, int height,
                        int components_nb)
{
    std::vector<vector<int>> results;
    for (int i = 1; i <= components_nb; i++)
    {
        int Xmin = width, Xmax = 0, Ymin = height, Ymax = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (components[y][x] == i)
                {
                    if (Xmin > x)
                        Xmin = x;
                    if (Ymin > y)
                        Ymin = y;
                    if (Xmax < x)
                        Xmax = x;
                    if (Ymax < y)
                        Ymax = y;
                }
            }
        }
        std::vector<int> out({ Xmin, Ymin, Xmax - Xmin, Ymax - Ymin });
        results.push_back(out);
    }
    return results;
}
