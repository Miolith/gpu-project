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

#endif

