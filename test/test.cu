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

void testGrayScale(rgba** image, rgba* imageGPU, int width, int height)
{
    cerr << "Testing GRAY SCALE..." << endl;
    cerr << "Applying CPU version grayScale()" << endl;
    grayScale(image, height, width);
    cerr << "Applying GPU version grayScaleGPU()" << endl;
    grayScaleGPU(imageGPU, height, width);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "grayScale() passed" << endl;
    else
        cerr << "grayScale() failed" << endl;
}

void testGaussianBlur(rgba** image, rgba* imageGPU, int width, int height)
{
    cerr << "Testing GAUSSIAN BLUR..." << endl;
    
    cerr << "Applying CPU version gaussianBlur()" << endl;
    gaussianBlur(image, height, width);
    cerr << "Applying GPU version gaussianBlurGPU()" << endl;
    gaussianBlurGPU(imageGPU, height, width);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "gaussianBlur() passed" << endl;
    else
        cerr << "gaussianBlur() failed" << endl;
}

void testImageDiff(rgba** ref, rgba** image, rgba* refGPU, rgba* imageGPU, int width, int height)
{
    cerr << "Testing IMAGE DIFF..." << endl;
    cerr << "Applying CPU version imageDiff()" << endl;
    imageDiff(ref, image, height, width);
    cerr << "Applying GPU version imageDiffGPU()" << endl;
    imageDiffGPU(refGPU, imageGPU, height, width);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "imageDiff() passed" << endl;
    else
        cerr << "imageDiff() failed" << endl;
}

void testDilation(rgba** image, rgba* imageGPU, int precision, int width, int height)
{
    cerr << "Testing DILATION..." << endl;
    cerr << "Applying CPU version dilation()" << endl;
    dilation(image, height, width, precision);
    cerr << "Applying GPU version dilationGPU()" << endl;
    dilationGPU(imageGPU, height, width, precision);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "dilation() passed" << endl;
    else
        cerr << "dilation() failed" << endl;
}

int main()
{
    int width = 1000;
    int height = 1000;

    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    saveImage("test_image.png", image, width, height);

    testGrayScale(image, imageGPU, width, height);

    grayScale(ref, height, width);
    grayScaleGPU(refGPU, height, width);

    saveImage("test_image_gray.png", image, width, height);
    saveImageGPU("test_image_gray_gpu.png", imageGPU, width, height);

    testGaussianBlur(image, imageGPU, width, height);

    saveImage("test_image_blur.png", image, width, height);
    saveImageGPU("test_image_blur_gpu.png", imageGPU, width, height);

    gaussianBlur(ref, height, width);
    gaussianBlurGPU(refGPU, height, width);

    testImageDiff(ref, image, refGPU, imageGPU, width, height);

    saveImage("test_image_diff.png", image, width, height);
    saveImageGPU("test_image_diff_gpu.png", imageGPU, width, height);
    
    testDilation(image, imageGPU, 10, width, height);

    saveImage("test_image_dilation.png", image, width, height);
    saveImageGPU("test_image_dilation_gpu.png", imageGPU, width, height);
    
    unloadImage(ref, height);
    unloadImage(image, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
}