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

__constant__ float gfilter[3][3] = { { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
                               { 2.0 / 16, 4.0 / 16, 2.0 / 16 },
                               { 1.0 / 16, 2.0 / 16, 1.0 / 16 } };


__global__ void gaussianBlurKernel(rgba *image, rgba **refImg, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int index = y * width + x;
        float red = 0.0;
        float green = 0.0;
        float blue = 0.0;
        float alpha = 0.0;
        for (int k = -1; k <= 1; k++)
        {
            for (int l = -1; l <= 1; l++)
            {
                if (y + k >= 0 && y + k < height && x + l >= 0
                    && x + l < width)
                {
                    red +=
                        refImg[y + k][x + l].red * gfilter[k + 1][l + 1];
                    green +=
                        refImg[y + k][x + l].green * gfilter[k + 1][l + 1];
                    blue +=
                        refImg[y + k][x + l].blue * gfilter[k + 1][l + 1];
                    alpha +=
                        refImg[y + k][x + l].alpha * gfilter[k + 1][l + 1];
                }
            }
        }
        image[index].red = red;
        image[index].green = green;
        image[index].blue = blue;
        image[index].alpha = alpha;
    }
}

__global__ void dilationKernel(rgba *image, bool **circleTable,rgba **refImg, int width, int height, int precision)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int index = y * width + x;
 
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

		    if(refImg[new_y][new_x].red > maxi)
		    {
			    maxi = refImg[new_y][new_x].red;
		    }
		}
        }
        image[index].red = maxi;
        image[index].green = maxi;
        image[index].blue = maxi;
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


void GaussianBlurGPU(rgba **image, int width, int height)
{
    int line_size = width * sizeof(rgba);
    rgba *d_image;
    cudaMalloc(&d_image, height*width*sizeof(rgba));
    for(int y = 0; y < height; y++)
    {
        cudaMemcpy(d_image +y*width, image[y], line_size, cudaMemcpyHostToDevice);
    }
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_image, image, width, height);
    for (int y = 0; y < height; y++)
    {
        cudaMemcpy(image[y], d_image + y*width, line_size, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_image);
}

void dilationGPU(rgba **image, int width, int height, int precision)
{
    bool **circleTable = getCircleTable(2 * precision);
    int line_size = width * sizeof(rgba);
    rgba *d_image;
    cudaMalloc(&d_image, height*width*sizeof(rgba));
    for(int y = 0; y < height; y++)
    {
        cudaMemcpy(d_image +y*width, image[y], line_size, cudaMemcpyHostToDevice);
    }
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    dilationKernel<<<numBlocks, threadsPerBlock>>>(d_image, circleTable, image, width, height, precision);
    for (int y = 0; y < height; y++)
    {
        cudaMemcpy(image[y], d_image + y*width, line_size, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_image);
}
