#include<vector>
#include<map>
#include<string>
#include<iostream>
#include "CImg.h"

using namespace std;
using namespace cimg_library;
using boxList = vector<vector<int>>;
using boxMap = map<string, boxList>;

struct rgba {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

rgba** loadImage(const string& filename, int* width, int* height)
{
    // load the image from path into the rgba array
    CImg<unsigned char> src(filename.c_str());
    *width = src.width();
    *height = src.height();

    rgba** image = new rgba*[*height];

    for (int i = 0; i < *height; i++) {
        image[i] = new rgba[*width];
        for (int j = 0; j < *width; j++) {
            image[i][j].red = src(j, i, 0, 0);
            image[i][j].green = src(j, i, 0, 1);
            image[i][j].blue = src(j, i, 0, 2);
            image[i][j].alpha = src(j, i, 0, 3);
        }
    }
    return image;
}

void unloadImage(rgba** image, int height)
{
    // delete the image array
    for (int i = 0; i < height; i++) {
        delete[] image[i];
    }
    delete[] image;
}

// Function to find bounding box for a given image
boxList findBox(char* reference, char* image)
{
    boxList box;

    int width;
    int height;

    rgba** img = loadImage(image, &width, &height);
    box.push_back({ 0, 0, 100, 100 });
    box.push_back({ 0, 0, 100, 100 });
    
    unloadImage(img, height);

    return box;
}

// Function to find bounding boxes for each image using the reference image
boxMap findBoundingBoxes(char* reference, int count, char** images)
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
            cout << "[" << (*it2)[0] << "," << (*it2)[1] << "," << (*it2)[2] << "," << (*it2)[3] << "]";
            if (it2 != it->second.end() - 1)
                cout << ",";
        }
        cout << "]";
        if (it != prev(boxes.end()))
            cout << ",";
    }
    cout << "}";
}

int main(int argc, char** argv) {
    // mybin refencec.png image1.png ...
    boxMap boxes = findBoundingBoxes(argv[1], argc - 2, argv + 2);
    
    printBoundingBoxes(boxes);
}
