#pragma once

#define DEBUG false

#include <map>
#include <string>
#include <vector>
#include <set>

#define GPU 1
#define CPU 0

struct rgba
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

using boxList = std::vector<std::vector<int>>;
using boxMap = std::map<std::string, boxList>;

using namespace std;

boxList findBox(rgba** ref, int w, int h, char *image);
boxList findBoxGPU(rgba* ref, int w, int h, char *image);
rgba** loadReference(char *filename, int *width, int *height);
rgba* loadReferenceGPU(char *filename, int *width, int *height);
boxMap findBoundingBoxes(char *reference, int count, char **images, int device);
void printBoundingBoxes(boxMap boxes);


void imageDiff(rgba **imageRef, rgba **imageOther, int width, int height);

void gaussianBlur(rgba **image, int width, int height);

void grayScale(rgba **image, int width, int height);

void dilation(rgba **image, int height, int width, int precision);

void erosion(rgba **image, int height, int width, int precision);

rgba **loadImage(const string &filename, int *width, int *height);

void unloadImage(rgba **image, int height);

void saveImage(const string &filename, rgba **image, int width, int height);

void basic_threshold(rgba **image, int height, int width, uint8_t threshold);

vector<vector<int>> connectCompenent(rgba** img, int height, int width, set<int> &labelSet);

std::vector<std::vector<int>>
component_box_detection(vector<vector<int>> components, int width, int height,
                        set<int> &labelSet);

void show_components(rgba** img, vector<vector<int>> comp, int width, int height, set<int> &labelSet);


// GPU functions

rgba *loadImageGPU(const string &filename, int *width, int *height);
void unloadImageGPU(rgba *image);
void saveImageGPU(const string &filename, rgba *image, int width, int height);
rgba* flattenImageGPU(rgba** image, int width, int height);

void grayScaleGPU(rgba *image, int width, int height);

void gaussianBlurGPU(rgba *image, int width, int height);

void dilationGPU(rgba **image, int width, int height, int precision);

void imageDiffGPU(rgba *image, rgba *ref, int width, int height);
void dilationGPU(rgba *image, int width, int height, int precision);

bool *getCircleTableGPU(int diameter);

void erosionGPU(rgba *image, int width, int height, int precision);
void basicThresholdGPU(rgba *image, int height, int width, uint8_t threshold);

vector<vector<size_t>> connectCompenentGPU(rgba* img, int height, int width, set<size_t> &labelSet);


void show_componentsGPU(rgba *img, vector<vector<size_t>> labelTable, int width,
                     int height, set<size_t> &labelSet);

vector<vector<int>>
component_box_detectionGPU(vector<vector<size_t>> components, int width, int height,
                        set<size_t> &labelSet);