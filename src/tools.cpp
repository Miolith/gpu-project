#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"

using namespace std;
using namespace cimg_library;

bool **getCircleTable(int radius)
{
    int new_y = radius / 2;
    int new_x = radius / 2;

    int squaredRadius = (radius / 2) * (radius / 2);
    bool **circleTable = new bool *[radius + 1];
    for (int i = 0; i <= radius; i++)
    {
        circleTable[i] = new bool[radius + 1];
    }
    for (int yoffset = -new_y; yoffset <= new_y; yoffset++)
    {
        for (int xoffset = -new_x; xoffset <= new_x; xoffset++)
        {
            circleTable[yoffset + new_y][xoffset + new_x] =
                (pow(xoffset, 2) + pow(yoffset, 2)) > squaredRadius;
        }
    }

    return circleTable;
}

void freeCircleTable(bool **table, int radius)
{
    for (int i = 0; i <= radius; i++)
    {
        delete[] table[i];
    }
    delete[] table;
}
