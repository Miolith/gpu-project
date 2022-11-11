#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

std::vector<int> basic_box_detection(rgba **image, int width, int height)
{
    int Xmin = 255, Xmax = 0, Ymin = 255, Ymax = 0;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            if(image[y][x].red == 255)
            {
                if(Xmin > x)
                    Xmin = x;
                if(Ymin > x)
                    Ymin = y;
                if(Xmax < x)
                    Xmax = x;
                if(Ymax < y)
                    Ymax = y;
            }
        }
    }
    return std::vector<int>({Xmin, Ymin,Xmax-Xmin, Ymax-Ymin});
}
