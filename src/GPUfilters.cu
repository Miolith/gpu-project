#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

#include "CImg.h"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

__global__ void grayScaleKernel(rgba *image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int index = y * width + x;
        uint8_t gray =
            (image[index].red + image[index].green + image[index].blue) / 3;
        image[index].red = gray;
        image[index].green = gray;
        image[index].blue = gray;
    }
}
// call gray scale kernel
void grayScaleGPU(rgba **image, int width, int height)
{
    rgba *d_image;
    int size = width * sizeof(rgba);
    cudaMalloc(&d_image, size * height);
    for(int y = 0; y < height; y++)
    {
        cudaMemcpy(d_image +y*width, image[y], size, cudaMemcpyHostToDevice);
    }
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grayScaleKernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
    for (int y = 0; y < height; y++)
    {
        cudaMemcpy(image[y], d_image + y*width, size, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_image);
}
