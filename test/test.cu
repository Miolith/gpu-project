#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <cstdlib>

#include "CImg.h"
#include "../src/draw.h"

#define GPU 1
#define CPU 0

using namespace std;

rgba** createRefImage(int width, int height)
{
    rgba** image = new rgba*[height];
    for (int i = 0; i < height; i++)
    {
        image[i] = new rgba[width];
        // fill image[i] with random values
        for (int j = 0; j < width; j++)
        {
            image[i][j].red = rand() % 256;
            image[i][j].green = rand() % 256;
            image[i][j].blue = rand() % 256;
            image[i][j].alpha = 255;
        }
    }
    return image;
}

rgba** copyImage(rgba** image, int width, int height)
{
    rgba** copy = new rgba*[height];
    for (int i = 0; i < height; i++)
    {
        copy[i] = new rgba[width];
        for (int j = 0; j < width; j++)
        {
            copy[i][j] = image[i][j];
        }
    }
    return copy;
}

// add one rectangle at the center of the image
rgba** createTestImage(rgba** ref, int width, int height)
{
    // copy ref into image
    rgba** image = new rgba*[height];
    for (int i = 0; i < height; i++)
    {
        image[i] = new rgba[width];
        for (int j = 0; j < width; j++)
        {
            image[i][j] = ref[i][j];
        }
    }

    int x = width / 2;
    int y = height / 2;
    int w = 100;
    int h = 100;
    int r = 255;
    int g = 0;
    int b = 0;
    int a = 255;

    for (int i = y - h / 2; i < y + h / 2; i++)
    {
        for (int j = x - w / 2; j < x + w / 2; j++)
        {
            image[i][j].red = r;
            image[i][j].green = g;
            image[i][j].blue = b;
            image[i][j].alpha = a;
        }
    }
    return image;
}

bool compareImages(rgba** image1, rgba** image2, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (!((image1[i][j].red == image2[i][j].red) &&
            (image1[i][j].green == image2[i][j].green) &&
            (image1[i][j].blue == image2[i][j].blue) &&
            (image1[i][j].alpha == image2[i][j].alpha)))
            {
                cerr << "Images differ at (" << i << ", " << j << ")" << endl;
                return false;
            }
        }
    }
    return true;
}

void testGrayScale(rgba** image, rgba** imageCopy, int width, int height)
{
    cerr << "Testing grayscale..." << endl;
    // convert image to grayscale
    grayScale(image, height, width);
    grayScaleGPU(imageCopy, height, width);

    // compare
    if (compareImages(image, imageCopy, width, height))
    {
        cerr << "grayScale() passed" << endl;
    }
    else
    {
        cerr << "grayScale() failed" << endl;
    }
}

void testGaussianBlur(rgba** image, rgba** imageCopy, int width, int height)
{
    cerr << "Testing gaussian blur..." << endl;
    // blur image
    gaussianBlur(image, height, width);
    GaussianBlurGPU(imageCopy, height, width);

    // compare
    if (compareImages(image, imageCopy, width, height))
    {
        cerr << "gaussianBlur() passed" << endl;
    }
    else
    {
        cerr << "gaussianBlur() failed" << endl;
    }
}

int main()
{
    int width = 1000;
    int height = 1000;

    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);

    testGrayScale(image, imageCopy, width, height);
    testGaussianBlur(image, imageCopy, width, height);

    unloadImage(ref, height);
    unloadImage(image, height);
    unloadImage(imageCopy, height);
}