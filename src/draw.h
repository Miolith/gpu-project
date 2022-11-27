#pragma once

#include <map>
#include <string>
#include <vector>
#include <set>


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
using namespace cimg_library;


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

void grayScaleGPU(rgba **image, int width, int height);

void GaussianBlurGPU(rgba **image, int width, int height);
