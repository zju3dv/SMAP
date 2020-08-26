#include "cuda_cal.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

const auto THREADS_PER_BLOCK_1D = 16u;
const auto THREADS_PER_BLOCK = 512u;

// Note: Shared memory made this function slower, from 1.2 ms to about 2 ms.
template <typename T>
__global__ void nmsRegisterKernel(
	int* kernelPtr, const T* const sourcePtr, const int w, const int h, const T threshold)
{
	// get pixel location (x,y)
	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	const auto channel = blockIdx.z * blockDim.z + threadIdx.z;
	const auto channelOffset = channel * w * h;
	const auto index = y * w + x; // for one channel

	auto* kernelPtrOffset = &kernelPtr[channelOffset];          // the beginning of one channel
	const T* const sourcePtrOffset = &sourcePtr[channelOffset]; // the beginning of one channel

	if (0 < x && x < (w - 1) && 0 < y && y < (h - 1))
	{
		const auto value = sourcePtrOffset[index];
		if (value > threshold)
		{
			const auto topLeft = sourcePtrOffset[(y - 1)*w + x - 1];
			const auto top = sourcePtrOffset[(y - 1)*w + x];
			const auto topRight = sourcePtrOffset[(y - 1)*w + x + 1];
			const auto left = sourcePtrOffset[y*w + x - 1];
			const auto right = sourcePtrOffset[y*w + x + 1];
			const auto bottomLeft = sourcePtrOffset[(y + 1)*w + x - 1];
			const auto bottom = sourcePtrOffset[(y + 1)*w + x];
			const auto bottomRight = sourcePtrOffset[(y + 1)*w + x + 1];

			if (value > topLeft && value > top && value > topRight
				&& value > left && value > right
				&& value > bottomLeft && value > bottom && value > bottomRight)
				kernelPtrOffset[index] = 1;
			else
				kernelPtrOffset[index] = 0;
		}
		else
			kernelPtrOffset[index] = 0;
	}
	else if (x == 0 || x == (w - 1) || y == 0 || y == (h - 1))
		kernelPtrOffset[index] = 0;
}

template <typename T>
__global__ void writeResultKernel(
	T* output, const int length, const int* const kernelPtr, const T* const sourcePtr, const int width,
	const int height, const int maxPeaks, const T offsetX, const T offsetY, const int offsetTarget)
{
	__shared__ int local[THREADS_PER_BLOCK + 1]; // one more
	__shared__ int kernel0; // Offset for kernel
	const auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x; //[0,368*512]
	const auto channel = blockIdx.y * blockDim.y + threadIdx.y; //[0,15]
	const auto channelOffsetSource = channel * width*height;
	const auto channelOffset = channel * offsetTarget;

	// We need to substract the peak at pixel 0 of the current channel for all values
	if (threadIdx.x == 0)
		kernel0 = kernelPtr[channelOffsetSource];  // == a number of the base
	__syncthreads();

	if (globalIdx < length)
	{
		auto* outputOffset = &output[channelOffset];
		const auto* const kernelPtrOffset = &kernelPtr[channelOffsetSource];
		const auto* const sourcePtrOffset = &sourcePtr[channelOffsetSource];
		local[threadIdx.x] = kernelPtrOffset[globalIdx] - kernel0; // local, [0, 512], global, [0, 368*512]
		//last thread in the block but not globally last, load one more
		if (threadIdx.x == THREADS_PER_BLOCK - 1 && globalIdx != length - 1)
			local[threadIdx.x + 1] = kernelPtrOffset[globalIdx + 1] - kernel0;
		__syncthreads();

		// See difference, except the globally last one
		if (globalIdx != length - 1)
		{
			// A[globalIdx] == A[globalIdx + 1] means no peak
			if (local[threadIdx.x] != local[threadIdx.x + 1])
			{
				const auto peakIndex = local[threadIdx.x]; //0-index
				const auto peakLocX = (int)(globalIdx % width);
				const auto peakLocY = (int)(globalIdx / width);

				// Accurate peak location: considered neighboors
				if (peakIndex < maxPeaks) // limitation
				{
					T xAcc = 0.f;
					T yAcc = 0.f;
					T scoreAcc = 0.f;
					const auto dWidth = 3;
					const auto dHeight = 3;
					for (auto dy = -dHeight; dy <= dHeight; dy++)
					{
						const auto y = peakLocY + dy;
						if (0 <= y && y < height) // Default height = 368
						{
							for (auto dx = -dWidth; dx <= dWidth; dx++)
							{
								const auto x = peakLocX + dx;
								if (0 <= x && x < width) // Default width = 656
								{
									const auto score = sourcePtrOffset[y * width + x];
									if (score > 0)
									{
										// xAcc, yAcc are expectations   xAcc = E(x*p(x))
										xAcc += x * score;
										yAcc += y * score;
										scoreAcc += score;
									}
								}
							}
						}
					}

					// Offset to keep Matlab format (empirically higher acc)
					// Best results for 1 scale: x + 0, y + 0.5
					// +0.5 to both to keep Matlab format
					const auto outputIndex = (peakIndex + 1) * 3;  // 3,6,9
					outputOffset[outputIndex] = xAcc / scoreAcc + offsetX;
					outputOffset[outputIndex + 1] = yAcc / scoreAcc + offsetY;
					outputOffset[outputIndex + 2] = sourcePtrOffset[peakLocY*width + peakLocX];
				}
			}
		}
		// If index 0 --> Assign number of peaks (truncated to the maximum possible number of peaks)
		else
			outputOffset[0] = (local[threadIdx.x] < maxPeaks ? local[threadIdx.x] : maxPeaks); // the maximum index indicates how many peaks
	}
}

template <typename T>
void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold,
	const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const T& offset)
{
	const auto num = sourceSize[0];          // 1
	const auto height = sourceSize[2];       // 368
	const auto width = sourceSize[3];        // 512
	const auto channels = targetSize[1];     // 25
	const auto maxPeaks = targetSize[2] - 1; // 127 + 1 - 1
	const auto imageOffset = height * width; // 368*512
	const auto offsetTarget = (maxPeaks + 1)*targetSize[3]; // 128 * 3

	// Optimized code: Running 3 kernels in total
	// This returns kernelPtr, a binary array with 0s & 1s. 1s in the local maximum
	// positions (size = size(sourcePtrOffsetted))
	// Example result: [0,0,0,0,1,0,0,0,0,1,0,0,0,0]
	//                 [0,0,0,0,0,1,1,1,1,1,2,2,2,2]
	// time = 1.24 ms
	const dim3 threadsPerBlockRegister{ THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1 };
	const dim3 numBlocksRegister{ getNumberCudaBlocks(width,          threadsPerBlockRegister.x),
								  getNumberCudaBlocks(height,         threadsPerBlockRegister.y),
								  getNumberCudaBlocks(num * channels, threadsPerBlockRegister.z) };
	nmsRegisterKernel<<<numBlocksRegister, threadsPerBlockRegister>>> (
		kernelPtr, sourcePtr, width, height, threshold);
	// This modifies kernelPtrOffsetted, now it indicates the local maximum indexes
	// Format: 0,0,0,1,1,1,1,2,2,2,... First maximum at index 2, second at 6, etc...
	// Example result: [0,0,0,0,0,1,1,1,1,1,2,2,2,2]
	// time = 2.71 ms
	auto kernelThrustPtr = thrust::device_pointer_cast(kernelPtr);
	thrust::exclusive_scan(kernelThrustPtr, kernelThrustPtr + num * channels*imageOffset, kernelThrustPtr);
	// This returns targetPtrOffsetted, with the NMS applied over it
	// time = 1.10 ms
	const dim3 threadsPerBlockWrite{ THREADS_PER_BLOCK, 1 };
	const dim3 numBlocksWrite{ getNumberCudaBlocks(imageOffset,    threadsPerBlockWrite.x),
							   getNumberCudaBlocks(num * channels, threadsPerBlockWrite.y) };
	writeResultKernel<<<numBlocksWrite, threadsPerBlockWrite>>> (
		targetPtr, imageOffset, kernelPtr, sourcePtr, width, height,
		maxPeaks, offset, offset, offsetTarget);
}

template void nmsGpu(
	float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold,
	const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const float& offset);
template void nmsGpu(
	double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold,
	const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize, const double& offset);
