#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <cstdlib>
#include "draw.h"

#define SAVE_IMAGE(filename, img) if (DEBUG) saveImage(filename, img, width, height);
#define SAVE_IMAGE_GPU(filename, img) if (DEBUG) saveImageGPU(filename, img, width, height);
using namespace std;

// Function to find bounding box for a given image
boxList findBox(rgba** ref, int w, int h, char *image)
{
    boxList box;

    int width, height;

    rgba **img = loadImage(image, &width, &height);

    assert(w == width && h == height);

    grayScale(img, width, height);
    SAVE_IMAGE("patate_grayscale.png", img);
    gaussianBlur(img, width, height);
    SAVE_IMAGE("patate_blur.png", img);

    imageDiff(ref, img, width, height);
    SAVE_IMAGE("patate_diff.png", img);

    // CLOSING
    dilation(img, height, width, 20);
    erosion(img, height, width, 20);
    SAVE_IMAGE("patate_closing.png", img);

    // OPENING
    erosion(img, height, width, 50);
    dilation(img, height, width, 50);
    SAVE_IMAGE("patate_opening.png", img);
    
    basic_threshold(img, height, width, 70);
    SAVE_IMAGE("patate_afterthresh.png", img);
    
    set<int> label_list;
    vector<vector<int>> labels = connectCompenent(img, height, width, label_list);
    
    show_components(img, labels, width, height, label_list);
    SAVE_IMAGE("patate_color.png", img);

    for(auto boxe: component_box_detection(labels, width, height, label_list))
    {
        box.push_back(boxe);
    }
    unloadImage(img, height);

    return box;
}

boxList findBoxGPU(rgba* ref, int w, int h, char *image)
{
    boxList box;
    int width, height;

    rgba *img = loadImageGPU(image, &width, &height);
    
    assert(w == width && h == height);

    grayScaleGPU(img, width, height);
    SAVE_IMAGE_GPU("patate_gray.png", img);
    gaussianBlurGPU(img, width, height);
    SAVE_IMAGE_GPU("patate_blur.png", img);

    imageDiffGPU(ref, img, width, height);
    SAVE_IMAGE_GPU("patate_diff.png", img);

    // CLOSING
    dilationGPU(img, width, height, 20);
    erosionGPU(img, width, height, 20);
    SAVE_IMAGE_GPU("patate_closing.png", img);

    // OPENING
    erosionGPU(img, width, height, 50);
    dilationGPU(img, width, height, 50);
    SAVE_IMAGE_GPU("patate_opening.png", img);
    
    basicThresholdGPU(img, height, width, 70);
    SAVE_IMAGE_GPU("patate_afterthresh.png", img);

    set<size_t> label_list;
    vector<vector<size_t>> labels = connectCompenentGPU(img, height, width, label_list);
    show_componentsGPU(img, labels, width, height, label_list);
    SAVE_IMAGE_GPU("patate_color.png", img);

    for(auto boxe: component_box_detectionGPU(labels, width, height, label_list))
    {
        box.push_back(boxe);
    }
    unloadImageGPU(img);

    return box;
}

rgba** loadReference(char *filename, int *width, int *height)
{
    rgba **image = loadImage(filename, width, height);
    grayScale(image, *width, *height);
    gaussianBlur(image, *width, *height);
    return image;
}

rgba* loadReferenceGPU(char *filename, int *width, int *height)
{
    rgba *image = loadImageGPU(filename, width, height);
    grayScaleGPU(image, *width, *height);
    gaussianBlurGPU(image, *width, *height);
    return image;
}

// Function to find bounding boxes for each image using the reference image
boxMap findBoundingBoxes(char *reference, int count, char **images, int device)
{
    boxMap boxes;
    int h, w;

    rgba** ref;
    rgba* refGPU;
    
    if(device == CPU)
        ref = loadReference(reference, &w, &h);
    
    if(device == GPU)
        refGPU = loadReferenceGPU(reference, &w, &h);
    
    for (int i = 0; i < count; i++)
    {
        if (device == GPU)
        {
            boxes[images[i]] = findBoxGPU(refGPU, w, h, images[i]);
        }
        else
        {
            boxes[images[i]] = findBox(ref, w, h, images[i]);
        }
    }
    if(device == CPU)
        unloadImage(ref, h);
    else
        unloadImageGPU(refGPU);
    return boxes;
}

// function to print the bounding boxes in json format
void printBoundingBoxes(boxMap boxes)
{
    // print the bounding boxes in json format
    cout << "{";
    for (auto it = boxes.begin(); it != boxes.end(); it++)
    {
        cout << "\"" << it->first << "\": [";
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            cout << "[" << (*it2)[0] << "," << (*it2)[1] << "," << (*it2)[2]
                 << "," << (*it2)[3] << "]";
            if (it2 != it->second.end() - 1)
                cout << ",";
        }
        cout << "]";
        if (it != prev(boxes.end()))
            cout << ",";
    }
    cout << "}\n";
}
