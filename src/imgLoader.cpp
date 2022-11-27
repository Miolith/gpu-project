#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"

using namespace std;
using namespace cimg_library;

rgba **loadImage(const string &filename, int *width, int *height)
{
    // load the image from path into the rgba array
    CImg<unsigned char> src(filename.c_str());
    *width = src.width();
    *height = src.height();

    rgba **image = new rgba *[*height];

    for (int i = 0; i < *height; i++)
    {
        image[i] = new rgba[*width];
        for (int j = 0; j < *width; j++)
        {
            uint8_t *fields[4] = { &image[i][j].red, &image[i][j].green,
                                   &image[i][j].blue, &image[i][j].alpha };
            image[i][j].alpha = 255;
            for (int spec = 0; spec < src.spectrum(); spec++)
            {
                *(fields[spec]) = src(j, i, 0, spec);
            }
        }
    }
    return image;
}


rgba *loadImageGPU(const string &filename, int *width, int *height)
{
    // load the image from path into the rgba array
    CImg<unsigned char> src(filename.c_str());
    *width = src.width();
    *height = src.height();

    rgba* image = new rgba[*height * *width];

    for (int i = 0; i < *height; i++)
    {
        for (int j = 0; j < *width; j++)
        {
            uint8_t *fields[4] = { &image[i * *width + j].red, &image[i * *width + j].green,
                                   &image[i * *width + j].blue, &image[i * *width + j].alpha };
            image[i * *width + j].alpha = 255;
            for (int spec = 0; spec < src.spectrum(); spec++)
            {
                *(fields[spec]) = src(j, i, 0, spec);
            }
        }
    }
    return image;
}

void unloadImage(rgba **image, int height)
{
    for (int i = 0; i < height; i++)
    {
        delete[] image[i];
    }
    delete[] image;
}


void unloadImageGPU(rgba *image)
{
    delete[] image;
}

// save image to file
void saveImage(const string &filename, rgba **image, int width, int height)
{
    CImg<unsigned char> dest(width, height, 1, 4);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uint8_t *fields[4] = { &image[i][j].red, &image[i][j].green,
                                   &image[i][j].blue, &image[i][j].alpha };
            for (int spec = 0; spec < dest.spectrum(); spec++)
            {
                dest(j, i, 0, spec) = *(fields[spec]);
            }
        }
    }
    dest.save(filename.c_str());
}

void saveImageGPU(const string &filename, rgba *image, int width, int height)
{
    CImg<unsigned char> dest(width, height, 1, 4);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uint8_t *fields[4] = { &image[i * width + j].red, &image[i * width + j].green,
                                   &image[i * width + j].blue, &image[i * width + j].alpha };
            for (int spec = 0; spec < dest.spectrum(); spec++)
            {
                dest(j, i, 0, spec) = *(fields[spec]);
            }
        }
    }
    dest.save(filename.c_str());
}

rgba* flattenImageGPU(rgba** image, int width, int height)
{
    rgba* flatImage = new rgba[height * width];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            flatImage[i * width + j] = image[i][j];
        }
    }

    return flatImage;
}