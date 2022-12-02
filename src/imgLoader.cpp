#include <algorithm>
#include <assert.h>
#include <iostream>

#include "draw.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

rgba **loadImage(const string &filename, int *width, int *height)
{
    int x, y, n;
    unsigned char *data = stbi_load(filename.c_str(), &x, &y, &n, 4);
    if (data == NULL)
    {
        cout << "Error: " << stbi_failure_reason() << endl;
        exit(1);
    }
    *width = x;
    *height = y;
    rgba **img = new rgba*[y];

    for (int i = 0; i < y; i++)
    {
        img[i] = new rgba[x];
        for (int j = 0; j < x; j++)
        {
            img[i][j].red = data[(i * x + j) * 4];
            img[i][j].green = data[(i * x + j) * 4 + 1];
            img[i][j].blue = data[(i * x + j) * 4 + 2];
            img[i][j].alpha = data[(i * x + j) * 4 + 3];
        }
    }
    stbi_image_free(data);
    return img;
}


rgba *loadImageGPU(const string &filename, int *width, int *height)
{
    // load the image from path into the rgba array
    int x, y, n;
    unsigned char *data = stbi_load(filename.c_str(), &x, &y, &n, 4);
    if (data == NULL)
    {
        cout << "Error: " << stbi_failure_reason() << endl;
        exit(1);
    }
    *width = x;
    *height = y;
    rgba *img = new rgba[x * y];

    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < x; j++)
        {
            img[i * x + j].red = data[(i * x + j) * 4];
            img[i * x + j].green = data[(i * x + j) * 4 + 1];
            img[i * x + j].blue = data[(i * x + j) * 4 + 2];
            img[i * x + j].alpha = data[(i * x + j) * 4 + 3];
        }
    }
    stbi_image_free(data);
    return img;
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
    unsigned char *data = new unsigned char[width * height * 4];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            data[(i * width + j) * 4] = image[i][j].red;
            data[(i * width + j) * 4 + 1] = image[i][j].green;
            data[(i * width + j) * 4 + 2] = image[i][j].blue;
            data[(i * width + j) * 4 + 3] = image[i][j].alpha;
        }
    }
    stbi_write_png(filename.c_str(), width, height, 4, data, width * 4);
    delete[] data;
}

void saveImageGPU(const string &filename, rgba *image, int width, int height)
{
    unsigned char *data = new unsigned char[width * height * 4];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            data[(i * width + j) * 4] = image[i * width + j].red;
            data[(i * width + j) * 4 + 1] = image[i * width + j].green;
            data[(i * width + j) * 4 + 2] = image[i * width + j].blue;
            data[(i * width + j) * 4 + 3] = image[i * width + j].alpha;
        }
    }
    stbi_write_png(filename.c_str(), width, height, 4, data, width * 4);
    delete[] data;
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