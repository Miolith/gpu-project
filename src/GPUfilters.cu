#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

#include "CImg.h"
#include "draw.h"
#include "tools.h"

using namespace std;
using namespace cimg_library;

__global__ void grayScaleKernel(rgba *image, int width, int height, size_t pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        image = (rgba *)((char *)image + y * pitch);
        int index = x;

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

__global__ void gaussianBlurKernel(rgba *dst_image, rgba *src_image, int width,
                                   int height, size_t pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        dst_image = (rgba *)((char *)dst_image + y * pitch);

        int index = x;

        float red = 0.0f, green = 0.0f, blue = 0.0f;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int x1 = x + j;
                int y1 = y + i;
                if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                {
                    rgba* img = (rgba *)((char *)src_image + y1 * pitch);
                    int index1 = x1;
                    red += img[index1].red * gfilter[i + 1][j + 1];
                    green += img[index1].green * gfilter[i + 1][j + 1];
                    blue += img[index1].blue * gfilter[i + 1][j + 1];
                }
            }
        }

        dst_image[index].red = red;
        dst_image[index].green = green;
        dst_image[index].blue = blue;
    }
}

__device__ int my_abs(int value)
{
	return (value < 0) ? -value : value;
}

__global__ void imageDiffKernel(rgba *dst_image, rgba *ref, int width, int height, size_t pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        dst_image = (rgba *)((char *)dst_image + y * pitch);
        ref = (rgba *)((char *)ref + y * pitch);

        int index = x;

        int red = my_abs(dst_image[index].red - ref[index].red);
        int green = my_abs(dst_image[index].green - ref[index].green);
        int blue = my_abs(dst_image[index].blue - ref[index].blue);

        dst_image[index].red = red;
        dst_image[index].green = green;
        dst_image[index].blue = blue;
    }
}

__global__ void dilationKernel(rgba *dst_image, rgba *src_image, int width,
                            int height, size_t pitch, int precision, bool* circleTable) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        dst_image = (rgba *)((char *)dst_image + y * pitch);
        int index = x;

        uint8_t maxi = 0;
        for (int yoffset = -precision; yoffset <= precision; yoffset++)
        {
            for (int xoffset = -precision; xoffset <= precision; xoffset++)
            {
                int new_y = y + yoffset;
                int new_x = x + xoffset;
                if (new_y < 0 || new_y >= height || new_x < 0 || new_x >= width
                    || circleTable[(yoffset + precision) * (precision * 2 + 1) + xoffset + precision])
                    continue;

                rgba* img = (rgba *)((char *)src_image + new_y * pitch);
                
                if (img[new_x].red > maxi)
                    maxi = img[new_x].red;
            }
        }
        dst_image[index].red = maxi;
        dst_image[index].green = maxi;
        dst_image[index].blue = maxi;
    }
}

// call gray scale kernel
void grayScaleGPU(rgba *image, int width, int height)
{
    rgba *dst_image;
    size_t pitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                 height, cudaMemcpyHostToDevice);
    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    grayScaleKernel<<<numBlocks, threadsPerBlock>>>(dst_image, width, height, pitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
}



void gaussianBlurGPU(rgba *image, int width, int height)
{
    rgba *dst_image, *src_image;
    size_t pitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&src_image, &pitch, width * sizeof(rgba), height);
    
    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(src_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);

    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(dst_image, src_image, width, height, pitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                    height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
    cudaFree(src_image);
}

void imageDiffGPU(rgba *ref, rgba *image, int width, int height)
{
    rgba *dst_image, *ref_image;
    size_t pitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&ref_image, &pitch, width * sizeof(rgba), height);

    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(ref_image, pitch, ref, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);


    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    imageDiffKernel<<<numBlocks, threadsPerBlock>>>(dst_image, ref_image, width, height, pitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
}
void dilationGPU(rgba *image, int width, int height, int precision)
{
    bool *circleTable = getCircleTableGPU(2 * precision);
    bool *circleTableCuda;

    rgba *dst_image, *src_image;

    size_t pitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&src_image, &pitch, width * sizeof(rgba), height);
    cudaMalloc(&circleTableCuda, (2 * precision + 1) * (2 * precision + 1) * sizeof(bool));

    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(src_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy(circleTableCuda, circleTable, (2 * precision + 1) * (2 * precision + 1) * sizeof(bool), cudaMemcpyHostToDevice);

    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    dilationKernel<<<numBlocks, threadsPerBlock>>>(dst_image, src_image, width, height, pitch, precision, circleTableCuda);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
    cudaFree(src_image);
    cudaFree(circleTableCuda);
    free(circleTable);
    return;
}
