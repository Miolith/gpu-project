#include<vector>
#include<map>
#include<string>
#include<iostream>

using namespace std;
using boxList = vector<vector<int>>;
using boxMap = map<string, boxList>;

// Function to find bounding box for a given image
boxList findBox(char* reference, char* image)
{
    boxList box;
    box.push_back({ 0, 0, 100, 100 });
    box.push_back({ 0, 0, 100, 100 });
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
