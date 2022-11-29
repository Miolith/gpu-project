#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>
#include <cstdlib>

#include "draw.h"

using namespace std;

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
