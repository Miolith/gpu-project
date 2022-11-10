#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CImg.h"
#include "draw.h"

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

// difference between two images
void imageDiff(rgba **imageRef, rgba **imageOther, int width, int height)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            imageOther[i][j].red =
                abs(imageOther[i][j].red - imageRef[i][j].red);
            imageOther[i][j].blue =
                abs(imageOther[i][j].blue - imageRef[i][j].blue);
            imageOther[i][j].green =
                abs(imageOther[i][j].green - imageRef[i][j].green);
        }
    }
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

bool **getCircleTable(int radius)
{
    int new_y = radius / 2;
    int new_x = radius / 2;

    int squaredRadius = (radius / 2) * (radius / 2);
    bool **circleTable = new bool *[radius + 1];
    for (int i = 0; i <= radius; i++)
    {
        circleTable[i] = new bool[radius + 1];
    }
    for (int yoffset = -new_y; yoffset <= new_y; yoffset++)
    {
        for (int xoffset = -new_x; xoffset <= new_x; xoffset++)
        {
            circleTable[yoffset + new_y][xoffset + new_x] =
                (pow(xoffset, 2) + pow(yoffset, 2)) > squaredRadius;
        }
    }

    return circleTable;
}

void dilation(rgba **image, int height, int width, int precision)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
    }

    bool **circleTable = getCircleTable(2 * precision);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t maxi = 0;
            for (int yoffset = -precision; yoffset <= precision; yoffset++)
            {
                for (int xoffset = -precision; xoffset <= precision; xoffset++)
                {
                    int new_y = y + yoffset;
                    int new_x = x + xoffset;
                    if (new_y < 0 || new_y >= height || new_x < 0
                        || new_x >= width
                        || circleTable[yoffset + precision]
                                      [xoffset + precision])
                        continue;

                    maxi = max(image[new_y][new_x].red, maxi);
                }
            }
            newImage[y][x].red = maxi;
        }
    }
    // copy the result back to the original image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j].red = newImage[i][j].red;
            image[i][j].green = newImage[i][j].red;
            image[i][j].blue = newImage[i][j].red;
            // image[i][j].alpha = newImage[i][j].alpha;
        }
    }

    // delete newImage
    unloadImage(newImage, height);
}

void erosion(rgba **image, int height, int width, int precision)
{
    // create a new image to store the result
    rgba **newImage = new rgba *[height];
    for (int i = 0; i < height; i++)
    {
        newImage[i] = new rgba[width];
    }

    bool **circleTable = getCircleTable(2 * precision);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t mini = 255;
            for (int yoffset = -precision; yoffset <= precision; yoffset++)
            {
                for (int xoffset = -precision; xoffset <= precision; xoffset++)
                {
                    int new_y = y + yoffset;
                    int new_x = x + xoffset;
                    if (new_y < 0 || new_y >= height || new_x < 0
                        || new_x >= width
                        || circleTable[yoffset + precision]
                                      [xoffset + precision])
                        continue;

                    mini = min(image[new_y][new_x].red, mini);
                }
            }
            newImage[y][x].red = mini;
        }
    }
    // copy the result back to the original image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j].red = newImage[i][j].red;
            image[i][j].green = newImage[i][j].red;
            image[i][j].blue = newImage[i][j].red;
            // image[i][j].alpha = newImage[i][j].alpha;
        }
    }

    // delete newImage
    unloadImage(newImage, height);
}

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
    grayScale(ref, w, h); // TODO: put them out of findBox for opti
    gaussianBlur(img, width, height);
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
