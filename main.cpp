#include <iostream>

#include "draw.h"
#include "CImg.h"

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

void unloadImage(rgba **image, int height)
{
    // delete the image array
    for (int i = 0; i < height; i++)
    {
        delete[] image[i];
    }
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

// smooth the image with gaussian filter
void gaussianBlur(rgba **image, int width, int height)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
    }

    // gaussian filter
    float filter[3][3] = { { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
                           { 2.0 / 16, 4.0 / 16, 2.0 / 16 },
                           { 1.0 / 16, 2.0 / 16, 1.0 / 16 } };

    // apply filter to each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float red = 0.0;
            float green = 0.0;
            float blue = 0.0;
            float alpha = 0.0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (i + k >= 0 && i + k < height && j + l >= 0
                        && j + l < width)
                    {
                        red += image[i + k][j + l].red * filter[k + 1][l + 1];
                        green +=
                            image[i + k][j + l].green * filter[k + 1][l + 1];
                        blue += image[i + k][j + l].blue * filter[k + 1][l + 1];
                        alpha +=
                            image[i + k][j + l].alpha * filter[k + 1][l + 1];
                    }
                }
            }
            newImage[i][j].red = red;
            newImage[i][j].green = green;
            newImage[i][j].blue = blue;
            newImage[i][j].alpha = alpha;
        }
    }

    // copy the result back to the original image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j].red = newImage[i][j].red;
            image[i][j].green = newImage[i][j].green;
            image[i][j].blue = newImage[i][j].blue;
            image[i][j].alpha = newImage[i][j].alpha;
        }
    }

    // delete newImage
    unloadImage(newImage, height);
}

void grayScale(rgba **image, int width, int height)
{
    // convert the image to grayscale
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uint8_t gray =
                (image[i][j].red + image[i][j].green + image[i][j].blue) / 3;
            image[i][j].red = gray;
            image[i][j].green = gray;
            image[i][j].blue = gray;
        }
    }
}

// Function to find bounding box for a given image
boxList findBox(char *reference, char *image)
{
    boxList box;

    int width;
    int height;

    rgba **img = loadImage(image, &width, &height);
    grayScale(img, width, height);
    gaussianBlur(img, width, height);

    saveImage("patate.png", img, width, height);

    box.push_back({ 0, 0, 100, 100 });
    box.push_back({ 0, 0, 100, 100 });

    unloadImage(img, height);

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
    cout << "}";
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Usage :\n"
             << argv[0] << " backgroundImage ImagePath [ImagePath]+" << endl;
        return 0;
    }
    // mybin refencec.png image1.png ...
    boxMap boxes = findBoundingBoxes(argv[1], argc - 2, argv + 2);

    printBoundingBoxes(boxes);
    return 0;
}
