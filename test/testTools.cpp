#include "testTools.hpp"
#include <iostream>
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

    for (int i = y - h / 2; i < y + h / 2; i++)
    {
        for (int j = x - w / 2; j < x + w / 2; j++)
        {
            image[i][j].red = r;
            image[i][j].green = g;
            image[i][j].blue = b;
        }
    }

    // rectangle at the upper left corner
    x = 50;
    y = 50;
    w = 100;
    h = 100;
    r = 0;
    g = 255;
    b = 0;
    for (int i = y - h / 2; i < y + h / 2; i++)
    {
        for (int j = x - w / 2; j < x + w / 2; j++)
        {
            image[i][j].red = r;
            image[i][j].green = g;
            image[i][j].blue = b;
        }
    }
    return image;
}

bool compareImages(rgba** image1, rgba* image2, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (!((image1[i][j].red == image2[i * width + j].red) &&
                  (image1[i][j].green == image2[i * width + j].green) &&
                  (image1[i][j].blue == image2[i * width + j].blue) &&
                  (image1[i][j].alpha == image2[i * width + j].alpha)))
            {
                cerr << "Images differ at (" << i << ", " << j << ")" << endl;

                cerr << "image1: " << (int)image1[i][j].red << " "
                     << (int)image1[i][j].green << " "
                     << (int)image1[i][j].blue << " "
                     << (int)image1[i][j].alpha << endl;

                cerr << "image2: " << (int)image2[i * width + j].red << " " 
                     << (int)image2[i * width + j].green << " "
                     << (int)image2[i * width + j].blue << " "
                     << (int)image2[i * width + j].alpha << endl;

                return false;
            }
        }
    }
    return true;
}

