#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"

using namespace std;
using namespace cimg_library;

// Function to find bounding box for a given image
boxList findBox(char *reference, char *image)
{
    boxList box;

    int width;
    int height;
    int w, h;

    rgba **img = loadImage(image, &width, &height);
    rgba **ref = loadImage(reference, &w, &h);
    grayScale(img, width, height);
    gaussianBlur(ref, w, h); // TODO: put them out of findBox for opti
    assert(w == width && h == height);
    imageDiff(ref, img, width, height);
    // Fermeture
    dilation(img, height, width, 20);
    erosion(img, height, width, 20);
    // Ouverture
    erosion(img, height, width, 50);
    dilation(img, height, width, 50);
    saveImage("patate.png", img, width, height);

    box.push_back({ 0, 0, 100, 100 });
    box.push_back({ 0, 0, 100, 100 });

    unloadImage(img, height);
    unloadImage(ref, h);

    return box;
}

// Function to find bounding boxes for each image using the reference image
boxMap findBoundingBoxes(char *reference, int count, char **images)
{
    boxMap boxes;
    for (int i = 0; i < count; i++)
    {
        boxes[images[i]] = findBox(reference, images[i]);
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
        cout << "Usage :\n"
             << argv[0] << " backgroundImage ImagePath [ImagePath]+" << endl
             << "\n";
        return 0;
    }
    // mybin refencec.png image1.png ...
    boxMap boxes = findBoundingBoxes(argv[1], argc - 2, argv + 2);

    printBoundingBoxes(boxes);
    return 0;
}
