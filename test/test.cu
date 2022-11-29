#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <cstdlib>

#include "CImg.h"
#include "../src/draw.h"
#include "testTools.hpp"

#define GPU 1
#define CPU 0

using namespace std;

void testGrayScale(rgba** image, rgba* imageGPU, int width, int height)
{
    cerr << "Testing GRAY SCALE..." << endl;
    cerr << "Applying CPU version grayScale()" << endl;
    grayScale(image, width, height);
    cerr << "Applying GPU version grayScaleGPU()" << endl;
    grayScaleGPU(imageGPU, width, height);

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
    gaussianBlur(image, width, height);
    cerr << "Applying GPU version gaussianBlurGPU()" << endl;
    gaussianBlurGPU(imageGPU, width, height);

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
    imageDiff(ref, image, width, height);
    cerr << "Applying GPU version imageDiffGPU()" << endl;
    imageDiffGPU(refGPU, imageGPU, width, height);

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
    dilationGPU(imageGPU, width, height, precision);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "dilation() passed" << endl;
    else
        cerr << "dilation() failed" << endl;
}

void testErosion(rgba** image, rgba* imageGPU, int precision, int width, int height)
{
    cerr << "Testing EROSION..." << endl;
    cerr << "Applying CPU version erosion()" << endl;
    erosion(image, height, width, precision);
    cerr << "Applying GPU version erosionGPU()" << endl;
    erosionGPU(imageGPU, width, height, precision);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "erosion() passed" << endl;
    else
        cerr << "erosion() failed" << endl;
}

void testBasicThreshold(rgba** image, rgba* imageGPU, uint8_t threshold, int width, int height)
{
    cerr << "Testing BASIC THRESHOLD..." << endl;
    cerr << "Applying CPU version basicThreshold()" << endl;
    basic_threshold(image, height, width, threshold);
    cerr << "Applying GPU version basicThresholdGPU()" << endl;
    basicThresholdGPU(imageGPU, height, width, threshold);

    // compare
    if (compareImages(image, imageGPU, width, height))
        cerr << "basicThreshold() passed" << endl;
    else
        cerr << "basicThreshold() failed" << endl;
}

void testLabelComponents(rgba** image, rgba* imageGPU, int width, int height)
{
    set<int> labelSet = set<int>();
    set<size_t> labelSetGPU = set<size_t>();

    cerr << "Testing LABEL COMPONENTS..." << endl;
    cerr << "Applying CPU version labelComponents()" << endl;
    connectCompenent(image, height, width, labelSet);
    cerr << "Applying GPU version labelComponentsGPU()" << endl;
    connectCompenentGPU(imageGPU, height, width, labelSetGPU);

    // print sets
    cerr << "CPU labels: ";
    for (set<int>::iterator it = labelSet.begin(); it != labelSet.end(); it++)
        cerr << *it << " ";
    cerr << endl;

    cerr << "GPU labels: ";
    for (set<size_t>::iterator it = labelSetGPU.begin(); it != labelSetGPU.end(); it++)
        cerr << *it << " ";
    cerr << endl;

    // compare
    if (labelSet.size() == labelSetGPU.size())
        cerr << "labelComponents() passed" << endl;
    else
        cerr << "labelComponents() failed" << endl;
}

int main()
{
    int width = 1200;
    int height = 900;

    rgba** ref = createRefImage(width, height);
    rgba** image = createTestImage(ref, width, height);
    rgba** imageCopy = copyImage(image, width, height);
    rgba* imageGPU = flattenImageGPU(imageCopy, width, height);
    rgba* refGPU = flattenImageGPU(ref, width, height);

    saveImage("test_image.png", image, width, height);

    testGrayScale(image, imageGPU, width, height);

    grayScale(ref, width, height);
    grayScaleGPU(refGPU, width, height);

    saveImage("test_image_gray.png", image, width, height);
    saveImageGPU("test_image_gray_gpu.png", imageGPU, width, height);

    testGaussianBlur(image, imageGPU, width, height);

    saveImage("test_image_blur.png", image, width, height);
    saveImageGPU("test_image_blur_gpu.png", imageGPU, width, height);

    gaussianBlur(ref, width, height);
    gaussianBlurGPU(refGPU, width, height);

    testImageDiff(ref, image, refGPU, imageGPU, width, height);

    saveImage("test_image_diff.png", image, width, height);
    saveImageGPU("test_image_diff_gpu.png", imageGPU, width, height);
    
    testDilation(image, imageGPU, 20, width, height);

    saveImage("test_image_dilation.png", image, width, height);
    saveImageGPU("test_image_dilation_gpu.png", imageGPU, width, height);
    
    testErosion(image, imageGPU, 20, width, height);

    saveImage("test_image_erosion.png", image, width, height);
    saveImageGPU("test_image_erosion_gpu.png", imageGPU, width, height);

    testBasicThreshold(image, imageGPU, 20, width, height);

    saveImage("test_image_threshold.png", image, width, height);
    saveImageGPU("test_image_threshold_gpu.png", imageGPU, width, height);

    testLabelComponents(image, imageGPU, width, height);

    unloadImage(ref, height);
    unloadImage(image, height);
    unloadImage(imageCopy, height);
    unloadImageGPU(imageGPU);
}
