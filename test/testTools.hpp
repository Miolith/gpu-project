#include "../src/draw.h"

rgba** createRefImage(int width, int height);

rgba** copyImage(rgba** image, int width, int height);

rgba** createTestImage(rgba** ref, int width, int height);

bool compareImages(rgba** image1, rgba* image2, int width, int height);