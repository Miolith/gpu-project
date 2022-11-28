#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

#include "draw.h"
#include "tools.h"
#include "cuda.h"

using namespace std;

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
                            int height, size_t pitch, int precision, bool* circleTable, size_t circlePitch) 
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
                bool* circle = (bool *)((char *)circleTable + (yoffset + precision) * circlePitch);
                if (new_y < 0 || new_y >= height || new_x < 0 || new_x >= width
                    || circle[xoffset + precision])
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

__global__ void erosionKernel(rgba *dst_image, rgba *src_image, int width,
                            int height, size_t pitch, int precision, bool* circleTable, size_t circlePitch) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        dst_image = (rgba *)((char *)dst_image + y * pitch);
        int index = x;

        uint8_t mini = 255;
        for (int yoffset = -precision; yoffset <= precision; yoffset++)
        {
            for (int xoffset = -precision; xoffset <= precision; xoffset++)
            {
                int new_y = y + yoffset;
                int new_x = x + xoffset;
                bool* circle = (bool *)((char *)circleTable + (yoffset + precision) * circlePitch);
                if (new_y < 0 || new_y >= height || new_x < 0 || new_x >= width
                    || circle[xoffset + precision])
                    continue;

                rgba* img = (rgba *)((char *)src_image + new_y * pitch);
                
                if (img[new_x].red < mini)
                    mini = img[new_x].red;
            }
        }
        dst_image[index].red = mini;
        dst_image[index].green = mini;
        dst_image[index].blue = mini;
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

    size_t pitch, circlePitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&src_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&circleTableCuda, &circlePitch, (2 * precision + 1) * sizeof(bool), 2 * precision + 1);

    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(src_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(circleTableCuda, circlePitch, circleTable, (2 * precision + 1) * sizeof(bool),
                (2 * precision + 1) * sizeof(bool), 2 * precision + 1, cudaMemcpyHostToDevice);

    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    dilationKernel<<<numBlocks, threadsPerBlock>>>(dst_image, src_image, width, height, 
                                pitch, precision, circleTableCuda, circlePitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
    cudaFree(src_image);
    cudaFree(circleTableCuda);
    delete circleTable;
    return;
}

void erosionGPU(rgba *image, int width, int height, int precision)
{
    bool *circleTable = getCircleTableGPU(2 * precision);
    bool *circleTableCuda;

    rgba *dst_image, *src_image;

    size_t pitch, circlePitch;

    cudaMallocPitch(&dst_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&src_image, &pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&circleTableCuda, &circlePitch, (2 * precision + 1) * sizeof(bool), 2 * precision + 1);

    cudaMemcpy2D(dst_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(src_image, pitch, image, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(circleTableCuda, circlePitch, circleTable, (2 * precision + 1) * sizeof(bool),
                (2 * precision + 1) * sizeof(bool), 2 * precision + 1, cudaMemcpyHostToDevice);

    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    erosionKernel<<<numBlocks, threadsPerBlock>>>(dst_image, src_image, width, height, 
                                pitch, precision, circleTableCuda, circlePitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);
    cudaFree(dst_image);
    cudaFree(src_image);
    cudaFree(circleTableCuda);
    delete circleTable;
    return;
}


__global__ void basicThresholdKernel(rgba *image, int width, int height, size_t pitch, uint8_t threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        rgba *pixel = (rgba *)((char *)image + y * pitch);
        if (pixel[x].red < threshold)
        {
            pixel[x].red = 0;
            pixel[x].green = 0;
            pixel[x].blue = 0;
        }
        else
        {
            pixel[x].red = 255;
            pixel[x].green = 255;
            pixel[x].blue = 255;
        }
    }
}

void basicThresholdGPU(rgba *image, int height, int width, uint8_t threshold)
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

    basicThresholdKernel<<<numBlocks, threadsPerBlock>>>(dst_image, width, height, pitch, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy2D(image, width * sizeof(rgba), dst_image, pitch, width * sizeof(rgba),
                height, cudaMemcpyDeviceToHost);

    cudaFree(dst_image);
}

__global__ void initLabelKernel(size_t *label_image, rgba *src_image, int width, int height, size_t img_pitch, size_t label_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        rgba *pixel = (rgba *)((char *)src_image + y * img_pitch);
        size_t *label = (size_t *)((char *)label_image + y * label_pitch);
        if (pixel[x].red == 0)
        {
            label[x] = 0;
        }
        else
        {
            label[x] = y * width + x + 1;
        }
    }
}


__device__ size_t my_min(size_t a, size_t b)
{
    return (a < b) ? a : b;
}


__global__ void propagateKernel(size_t *src_label, size_t *label_image, int width, int height, size_t pitch,volatile bool *label_changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        size_t *label_line = (size_t *)((char *)src_label + y * pitch);
        if (label_line[x] != 0)
        {
            size_t min_label = label_line[x];
            for (int i = -1; i <= 1; i++)
            {
                size_t *label = (size_t *)((char *)src_label + (y + i) * pitch);
                for (int j = -1; j <= 1; j++)
                {
                    if (x + j >= 0 && x + j < width && y + i >= 0 && y + i < height)
                    {
                        if (label[x + j] != 0)
                        {
                            min_label = my_min(min_label, label[x + j]);
                        }
                    }
                }
            }
            if (min_label != label_line[x])
            {
                *label_changed = true;
            }
            size_t *label_dst = (size_t *)((char *)label_image + y * pitch);
            label_dst[x] = min_label;
        }
    }
}

vector<vector<size_t>> connectCompenentGPU(rgba* img, int height, int width, set<size_t> &labelSet)
{
    rgba *src_image;
    size_t *label_image, *src_label;
    bool *label_changed;
    bool label_changed_host = true;
    vector<vector<size_t>> labelMatrix;
    size_t *labelTable = new size_t[height * width];

    size_t img_pitch, label_pitch;

    cudaMallocPitch(&src_image, &img_pitch, width * sizeof(rgba), height);
    cudaMallocPitch(&label_image, &label_pitch, width * sizeof(size_t), height);
    cudaMallocPitch(&src_label, &label_pitch, width * sizeof(size_t), height);

    cudaMalloc(&label_changed, sizeof(bool));
    cudaMemcpy(label_changed, &label_changed_host, sizeof(bool), cudaMemcpyHostToDevice);
    

    cudaMemcpy2D(src_image, img_pitch, img, width * sizeof(rgba), width * sizeof(rgba),
                height, cudaMemcpyHostToDevice);


    // init label kernel
    int bsize = 32;
    int w = ceil((float)width / bsize);
    int h = ceil((float)height / bsize);

    dim3 threadsPerBlock(bsize, bsize);
    dim3 numBlocks(w, h);

    initLabelKernel<<<numBlocks, threadsPerBlock>>>(src_label, src_image, width, height, img_pitch, label_pitch);
    cudaDeviceSynchronize();

    while(label_changed_host)
    {
        label_changed_host = false;
        cudaMemcpy(label_changed, &label_changed_host, sizeof(bool), cudaMemcpyHostToDevice);

        // propagate kernel
        propagateKernel<<<numBlocks, threadsPerBlock>>>(src_label, label_image, width, height, label_pitch, label_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&label_changed_host, label_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaMemcpy2D(src_label, label_pitch, label_image, label_pitch, width * sizeof(size_t),
                    height, cudaMemcpyDeviceToDevice);

    }

    cudaMemcpy2D(labelTable, width * sizeof(size_t), label_image, label_pitch, width * sizeof(size_t),
                height, cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (labelTable[i * width + j] != 0)
            {
                labelSet.insert(labelTable[i * width + j]);
            }
        }
    }

    /*show_componentsGPU(img, labelTable, height, width, labelSet);
    saveImageGPU("result.png", img, height, width);*/
    
    // fill label matrix with labelTable
    for (int i = 0; i < height; i++)
    {
        vector<size_t> labelVector;
        for (int j = 0; j < width; j++)
        {
            labelVector.push_back(labelTable[i * width + j]);
        }
        labelMatrix.push_back(labelVector);
    }


    cudaFree(src_image);
    cudaFree(label_image);
    cudaFree(src_label);
    free(labelTable);
    
    return labelMatrix;
}
