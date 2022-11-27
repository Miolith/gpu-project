#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"

using namespace std;
using namespace cimg_library;

bool **getCircleTable(int diameter)
{
    diameter = diameter + 1;
    int radius = diameter / 2;
    bool **circleTable = new bool*[diameter];
    int r2 = radius * radius;
    
    for (int i = 0; i < diameter; i++)
        circleTable[i] = new bool[diameter];
    
    for (int i = 0; i < diameter; i++)
    {
        for (int j = 0; j < diameter; j++)
        {
            int x = i - radius;
            int y = j - radius;
            circleTable[i][j] = (x * x + y * y > r2);
        }
    }
    return circleTable;
}

bool *getCircleTableGPU(int diameter)
{
    diameter = diameter + 1;
    int radius = diameter / 2;
    bool *circleTable = new bool[diameter * diameter];
    int r2 = radius * radius;
    
    for (int i = 0; i <= diameter; i++)
    {
        for (int j = 0; j <= diameter; j++)
        {
            int x = i - radius;
            int y = j - radius;
            circleTable[i * diameter + j] = (x * x + y * y > r2);
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
