#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>
#include <cstdlib>

#include "CImg.h"
#include "draw.h"

#define GPU 1
#define CPU 0

using namespace std;
using namespace cimg_library;

// Function to find bounding box for a given image
boxList findBox(rgba** ref, int w, int h, char *image)
{
    boxList box;

    int width, height;

    rgba **img = loadImage(image, &width, &height);
    assert(w == width && h == height);

    grayScale(img, width, height);
    gaussianBlur(img, width, height);

    imageDiff(ref, img, width, height);
    saveImage("patate_diff.png", img, width, height);

    // Fermeture
    dilation(img, height, width, 20);
    erosion(img, height, width, 20);
    saveImage("patate_closing.png", img, width, height);

    // Ouverture
    erosion(img, height, width, 50);
    dilation(img, height, width, 50);
    saveImage("patate_opening.png", img, width, height);
    
    basic_threshold(img, height, width, 40);
    
    saveImage("patate_afterthresh.png", img, width, height);
    set<int> label_list;
    vector<vector<int>> labels = connectCompenent(img, height, width, label_list);
    
    show_components(img, labels, width, height, label_list);
    saveImage("patate_color.png", img, width, height);

    for(auto boxe: component_box_detection(labels, width, height, label_list))
    {
        box.push_back(boxe);
    }
    unloadImage(img, height);
    unloadImage(ref, h);

    return box;
}

boxList findBoxGPU(rgba** ref, int w, int h, char *image)
{
    boxList box;

    int width, height;

    rgba **img = loadImage(image, &width, &height);
    assert(w == width && h == height);

    // call cuda kernel grayscale
    grayScaleGPU(img, width, height);
    GaussianBlurGPU(img, width, height);

    imageDiff(ref,img, width, height);
    saveImage("patate_diff.png", img, width, height);

    // Fermeture
    dilation(img, height, width, 20);
    erosion(img, height, width, 20);
    saveImage("patate_closing.png", img, width, height);

    // Ouverture
    erosion(img, height, width, 50);
    dilation(img, height, width, 50);
    saveImage("patate_opening.png", img, width, height);
    
    basic_threshold(img, height, width, 40);
    
    saveImage("patate_afterthresh.png", img, width, height);
    set<int> label_list;
    vector<vector<int>> labels = connectCompenent(img, height, width, label_list);
    show_components(img, labels, width, height, label_list);
    saveImage("patate_color.png", img, width, height);

    for(auto boxe: component_box_detection(labels, width, height, label_list))
    {
        box.push_back(boxe);
    }
    unloadImage(img, height);
    unloadImage(ref, h);

    return box;
}

rgba** loadReference(char *filename, int *width, int *height)
{
    rgba **image = loadImage(filename, width, height);
    grayScale(image, *width, *height);
    gaussianBlur(image, *width, *height);
    return image;
}

// Function to find bounding boxes for each image using the reference image
boxMap findBoundingBoxes(char *reference, int count, char **images, int device)
{
    boxMap boxes;
    int h, w;
    rgba** ref = loadReference(reference, &w, &h);
    for (int i = 0; i < count; i++)
    {
        if (device == GPU)
        {
            boxes[images[i]] = findBoxGPU(ref, w, h, images[i]);
        }
        else
        {
            boxes[images[i]] = findBox(ref, w, h, images[i]);
        }
    }
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

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Usage :\n" << "DEVICE=(CPU|GPU) "
             << argv[0] << " backgroundImage ImagePath [ImagePath]+" << endl
             << "\n";
        return 0;
    }
    // mybin refencec.png image1.png ...

    char* device_name = getenv("DEVICE");
    int device = CPU;


    if (device_name != NULL && strcmp(device_name, "GPU") == 0)
    {
        cerr << "Using GPU" << endl;
        device = GPU;
    }
    else
    {
        cerr << "Using CPU" << endl;
    }

    boxMap boxes = findBoundingBoxes(argv[1], argc - 2, argv + 2, device);

    printBoundingBoxes(boxes);
    return 0;
}
