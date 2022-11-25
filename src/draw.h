#ifndef DRAW_H
#define DRAW_H

#include <map>
#include <string>
#include <vector>


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

std::vector<int> basic_box_detection(rgba **image, int width, int height);

vector<vector<int>> connectCompenent(rgba** img, int height, int width);

#endif

