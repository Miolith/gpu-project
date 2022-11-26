#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

#include "CImg.h"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

/*
 * Absolute substraction between imageRef and imageOther
 */
void imageDiff(rgba **imageRef, rgba **imageOther, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            imageOther[i][j].red =
                abs(imageOther[i][j].red - imageRef[i][j].red);
            imageOther[i][j].blue =
                abs(imageOther[i][j].blue - imageRef[i][j].blue);
            imageOther[i][j].green =
                abs(imageOther[i][j].green - imageRef[i][j].green);
        }
    }
}

/*
 * Apply GaussianBlurr filter on the image
 */
void gaussianBlur(rgba **image, int width, int height)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
        for(int j = 0; j < width; j++)
        {
            newImage[i][j].red = image[i][j].red;
            newImage[i][j].green = image[i][j].green;
            newImage[i][j].blue = image[i][j].blue;
            newImage[i][j].alpha = image[i][j].alpha;
        }
    }

    // gaussian filter
    float filter[3][3] = { { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
                           { 2.0 / 16, 4.0 / 16, 2.0 / 16 },
                           { 1.0 / 16, 2.0 / 16, 1.0 / 16 } };

    // apply filter to each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float red = 0.0;
            float green = 0.0;
            float blue = 0.0;
            float alpha = 0.0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (i + k >= 0 && i + k < height && j + l >= 0
                        && j + l < width)
                    {
                        red += newImage[i + k][j + l].red * filter[k + 1][l + 1];
                        green +=
                            newImage[i + k][j + l].green * filter[k + 1][l + 1];
                        blue += newImage[i + k][j + l].blue * filter[k + 1][l + 1];
                        alpha +=
                            newImage[i + k][j + l].alpha * filter[k + 1][l + 1];
                    }
                }
            }
            image[i][j].red = red;
            image[i][j].green = green;
            image[i][j].blue = blue;
            image[i][j].alpha = alpha;
        }
    }

    // delete newImage
    unloadImage(newImage, height);
}

void grayScale(rgba **image, int width, int height)
{
    // convert the image to grayscale
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uint8_t gray =
                (image[i][j].red + image[i][j].green + image[i][j].blue) / 3;
            image[i][j].red = gray;
            image[i][j].green = gray;
            image[i][j].blue = gray;
        }
    }
}



void dilation(rgba **image, int height, int width, int precision)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
        for (int j = 0; j < width; j++)
        {
            newImage[i][j].red = image[i][j].red;
        }
    }

    bool **circleTable = getCircleTable(2 * precision);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t maxi = 0;
            for (int yoffset = -precision; yoffset <= precision; yoffset++)
            {
                for (int xoffset = -precision; xoffset <= precision; xoffset++)
                {
                    int new_y = y + yoffset;
                    int new_x = x + xoffset;
                    if (new_y < 0 || new_y >= height || new_x < 0
                        || new_x >= width
                        || circleTable[yoffset + precision]
                                      [xoffset + precision])
                        continue;

                    maxi = max(newImage[new_y][new_x].red, maxi);
                }
            }
            image[y][x].red = maxi;
            image[y][x].green = maxi;
            image[y][x].blue = maxi;

        }
    }
    // delete newImage
    unloadImage(newImage, height);
    freeCircleTable(circleTable, 2 * precision);
}

void erosion(rgba **image, int height, int width, int precision)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
        for (int j = 0; j < width; j++)
        {
            newImage[i][j].red = image[i][j].red;
        }
    }

    bool **circleTable = getCircleTable(2 * precision);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t mini = 255;
            for (int yoffset = -precision; yoffset <= precision; yoffset++)
            {
                for (int xoffset = -precision; xoffset <= precision; xoffset++)
                {
                    int new_y = y + yoffset;
                    int new_x = x + xoffset;
                    if (new_y < 0 || new_y >= height || new_x < 0
                        || new_x >= width
                        || circleTable[yoffset + precision]
                                      [xoffset + precision])
                        continue;

                    mini = min(newImage[new_y][new_x].red, mini);
                }
            }
            image[y][x].red = mini;
            image[y][x].green = mini;
            image[y][x].blue = mini;
        }
    }

    // delete newImage
    unloadImage(newImage, height);
    freeCircleTable(circleTable, 2 * precision);
}

void basic_threshold(rgba **image, int height, int width, uint8_t threshold)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if(image[i][j].red >= threshold)
            {
                image[i][j].red = 255;
                image[i][j].green = 255;
                image[i][j].blue = 255;
                image[i][j].alpha = 255;
            }
            else
            {
                image[i][j].red = 0;
                image[i][j].green = 0;
                image[i][j].blue = 0;
                image[i][j].alpha = 255;
            }
        }
    }
}

class DisjSet {
    map<int, int> parent, rank;
 
public:
   
    DisjSet(int n)
    {
        makeSet(n);
    }
 
    void makeSet(int n)
    {
        parent[n] = n;
    }
 
    int find(int x)
    {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
 
        return parent[x];
    }
 
    void Union(int x, int y)
    {
        int xset = find(x);
        int yset = find(y);
 
        if (xset == yset)
            return;
 
        if (rank[xset] < rank[yset])
            parent[xset] = yset;
        
        else if (rank[xset] > rank[yset])
            parent[yset] = xset;
        
 
        else
        {
            parent[yset] = xset;
            rank[xset] = rank[xset] + 1;
        }
    }
};

// connected component labeling Two Pass Algorithm
vector<vector<int>> connectCompenent(rgba** img, int height, int width, int& Labelnum)
{
    vector<vector<int>> label = vector<vector<int>>(height, vector<int>(width, 0));

    DisjSet disjoint_set(0);

    int nextLabel = 1;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {

            if (img[i][j].red == 255)
            {
                set<int> label_set;

                for (int n_i = i - 1; n_i <= i + 1; n_i++)
                {
                    for (int n_j = j - 1; n_j <= j + 1; n_j++)
                    {
                        if (n_j < 0 || n_i < 0 || n_i >= height || n_j >= width || (n_i == i && n_j == j))
                            continue;

                        if (label[n_i][n_j] != 0)
                        {
                            label_set.insert(label[n_i][n_j]);
                        }
                    }
                }

                if (label_set.empty())
                {
                    disjoint_set.makeSet(nextLabel);
                    label[i][j] = nextLabel;
                    nextLabel++;
                }
                else
                {
                    label[i][j] = *min_element(label_set.begin(), label_set.end());
                    for (auto it = label_set.begin(); it != label_set.end(); it++)
                        for (auto it2 = label_set.begin(); it2 != label_set.end(); it2++)
                            disjoint_set.Union(*it, *it2);
                }
            }
        }
    }

    Labelnum = 0;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (label[i][j] != 0)
            {
                label[i][j] = disjoint_set.find(label[i][j]);
                Labelnum = max(Labelnum, label[i][j]);
            }

    return label;
}
