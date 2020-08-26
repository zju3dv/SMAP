#include "cuda_cal.h"

#include <cuda.h>
#include <cuda_runtime.h>


const auto THREADS_PER_BLOCK_1D = 16u;
//const auto THREADS_PER_BLOCK = 512u;


// Max/min functions
template<class T>
inline __device__ T fastMaxCuda(const T a, const T b)
{
	return (a > b ? a : b);
}

template<class T>
inline __device__ T fastMinCuda(const T a, const T b)
{
	return (a < b ? a : b);
}

template<class T>
inline __device__ T fastTruncateCuda(const T value, const T min = 0, const T max = 1)
{
	return fastMinCuda(max, fastMaxCuda(min, value));
}

// Cubic interpolation
template <typename T>
inline __device__ void cubicSequentialData(
	int* xIntArray, int* yIntArray, T& dx, T& dy, const T xSource, const T ySource, const int widthSource,
	const int heightSource)
{
	xIntArray[1] = fastTruncateCuda(int(floor(xSource)), 0, widthSource - 1);
	xIntArray[0] = fastMaxCuda(0, xIntArray[1] - 1);
	xIntArray[2] = fastMinCuda(widthSource - 1, xIntArray[1] + 1);
	xIntArray[3] = fastMinCuda(widthSource - 1, xIntArray[2] + 1);
	dx = xSource - xIntArray[1];

	yIntArray[1] = fastTruncateCuda(int(floor(ySource)), 0, heightSource - 1);
	yIntArray[0] = fastMaxCuda(0, yIntArray[1] - 1);
	yIntArray[2] = fastMinCuda(heightSource - 1, yIntArray[1] + 1);
	yIntArray[3] = fastMinCuda(heightSource - 1, yIntArray[2] + 1);
	dy = ySource - yIntArray[1];
}

template <typename T>
inline __device__ T cubicInterpolate(const T v0, const T v1, const T v2, const T v3, const T dx)
{
	// http://www.paulinternet.nl/?page=bicubic
	// const auto a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
	// const auto b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
	// const auto c = (-0.5f * v0 + 0.5f * v2);
	// out = ((a * dx + b) * dx + c) * dx + v1;
	return (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
		+ (v0 - 2.5f * v1 + 2.f * v2 - 0.5f * v3) * dx * dx
		- 0.5f * (v0 - v2) * dx // + (-0.5f * v0 + 0.5f * v2) * dx
		+ v1;
	// return v1 + 0.5f * dx * (v2 - v0 + dx * (2.f * v0 - 5.f * v1 + 4.f * v2 - v3 + dx * (3.f * (v1 - v2) + v3 - v0)));
}



template <typename T>
inline __device__ T bicubicInterpolate(
	const T* const sourcePtr, const T xSource, const T ySource, const int widthSource, const int heightSource,
	const int widthSourcePtr)
{
	int xIntArray[4];
	int yIntArray[4];
	T dx;
	T dy;
	cubicSequentialData(xIntArray, yIntArray, dx, dy, xSource, ySource, widthSource, heightSource);

	T temp[4];
	for (unsigned char i = 0; i < 4; i++)
	{
		const auto offset = yIntArray[i] * widthSourcePtr;
		temp[i] = cubicInterpolate(
			sourcePtr[offset + xIntArray[0]], sourcePtr[offset + xIntArray[1]], sourcePtr[offset + xIntArray[2]],
			sourcePtr[offset + xIntArray[3]], dx);
	}
	return cubicInterpolate(temp[0], temp[1], temp[2], temp[3], dy);
}

template <typename T>
inline __device__ T bicubicInterpolate(
	const unsigned char* const sourcePtr, const T xSource, const T ySource, const int widthSource,
	const int heightSource, const int widthSourcePtr)
{
	int xIntArray[4];
	int yIntArray[4];
	T dx;
	T dy;
	cubicSequentialData(xIntArray, yIntArray, dx, dy, xSource, ySource, widthSource, heightSource);

	T temp[4];
	for (unsigned char i = 0; i < 4; i++)
	{
		const auto offset = yIntArray[i] * widthSourcePtr;
		temp[i] = cubicInterpolate(
			T(sourcePtr[offset + xIntArray[0]]), T(sourcePtr[offset + xIntArray[1]]),
			T(sourcePtr[offset + xIntArray[2]]), T(sourcePtr[offset + xIntArray[3]]), dx);
	}
	return cubicInterpolate(temp[0], temp[1], temp[2], temp[3], dy);
}


template <typename T>
inline __device__ T bicubicInterpolate8Times(
	const T* const sourcePtr, const T xSource, const T ySource, const int widthSource, const int heightSource,
	const int threadIdxX, const int threadIdxY)
{
	// Now we only need dx and dy
	const T dx = xSource - fastTruncateCuda(int(floor(xSource)), 0, widthSource - 1);
	const T dy = ySource - fastTruncateCuda(int(floor(ySource)), 0, heightSource - 1);

	T temp[4];
	for (unsigned char i = 0; i < 4; i++)
	{
		const auto offset = 5 * (i + (threadIdxY > 3 ? 1 : 0)) + (threadIdxX > 3 ? 1 : 0);
		temp[i] = cubicInterpolate(
			sourcePtr[offset], sourcePtr[offset + 1], sourcePtr[offset + 2],
			sourcePtr[offset + 3], dx);
	}
	return cubicInterpolate(temp[0], temp[1], temp[2], temp[3], dy);
}


template <typename T>
__global__ void resize8TimesKernel(
	T* targetPtr, const T* const sourcePtr, const int widthSource, const int heightSource, const int widthTarget,
	const int heightTarget, const unsigned int rescaleFactor)
{
	const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

	if (x < widthTarget && y < heightTarget)
	{
		// Normal resize
		// Note: The first blockIdx of each dimension behaves differently, so applying old version in those
		if (blockIdx.x < 1 || blockIdx.y < 1)
			// Actually it is only required for the first 4, but then I would have not loaded the shared memory
			// if ((blockIdx.x < 1 || blockIdx.y < 1) && (threadIdx.x < 4 || threadIdx.y < 4))
		{
			const auto sourceArea = widthSource * heightSource;
			const auto targetArea = widthTarget * heightTarget;
			const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const T* const sourcePtrChannel = sourcePtr + channel * sourceArea;
			targetPtr[channel * targetArea + y * widthTarget + x] = bicubicInterpolate(
				sourcePtrChannel, xSource, ySource, widthSource, heightSource, widthSource);
			return;
		}

		// Load shared memory
		// If resize >= 5, then #threads per block >= # elements of shared memory
		const auto sharedSize = 25; // (4+1)^2
		__shared__ T sourcePtrShared[sharedSize];
		const auto sharedLoadId = threadIdx.x + rescaleFactor * threadIdx.y;
		if (sharedLoadId < sharedSize)
		{
			// Idea: Find minimum possible x and y
			const auto minTargetX = blockIdx.x * rescaleFactor;
			const auto minSourceXFloat = (minTargetX + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const auto minSourceXInt = int(floor(minSourceXFloat)) - 1;
			const auto minTargetY = blockIdx.y * rescaleFactor;
			const auto minSourceYFloat = (minTargetY + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const auto minSourceYInt = int(floor(minSourceYFloat)) - 1;
			// Get current x and y
			const auto xClean = fastTruncateCuda(minSourceXInt + int(sharedLoadId % 5), 0, widthSource - 1);
			const auto yClean = fastTruncateCuda(minSourceYInt + int(sharedLoadId / 5), 0, heightSource - 1);
			// Load into shared memory
			const auto sourceIndex = (channel * heightSource + yClean) * widthSource + xClean;
			sourcePtrShared[sharedLoadId] = sourcePtr[sourceIndex];
		}
		__syncthreads();

		// Apply resize
		const auto targetArea = widthTarget * heightTarget;
		const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
		const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
		targetPtr[channel * targetArea + y * widthTarget + x] = bicubicInterpolate8Times(
			sourcePtrShared, xSource, ySource, widthSource, heightSource, threadIdx.x, threadIdx.y);
	}
}




template <typename T>
__global__ void uCharImageCastKernel(
	unsigned char* targetPtr, const T* const srcPtr, const int volume)
{
	const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (x < volume)
		targetPtr[x] = (unsigned char)(fastTruncateCuda(srcPtr[x], T(0), T(255)));
}

template <typename T>
void uCharImageCast(unsigned char* targetPtr, const T* const srcPtr, const int volume)
{
	try
	{
		const dim3 threadsPerBlock{ 32, 1, 1 };
		const dim3 numBlocks{ getNumberCudaBlocks(volume, threadsPerBlock.x) };
		uCharImageCastKernel << <numBlocks, threadsPerBlock >> > (targetPtr, srcPtr, volume);
	}
	catch (const std::exception& e)
	{
		//error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

/*
template void uCharImageCast(
	unsigned char* targetPtr, const float* const srcPtr, const int volume);
template void uCharImageCast(
	unsigned char* targetPtr, const double* const srcPtr, const int volume);
*/



template <typename T>
__global__ void reorderAndNormalizeKernel(
	T* dstPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels)
{
	const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const auto c = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (x < width && y < height) {
		const auto dstIdx = c * width * height + (y * width + x);
		const auto srcIdx = (y * width + x) * channels + c;
		dstPtr[dstIdx] = T(srcPtr[srcIdx]) * T(1 / 256.f) - T(0.5f);
	}
}

template <typename T>
void reorderAndNormalize(
	T* targetPtr, const unsigned char* const srcPtr, int width, int height, int channels)
{
	try
	{
		const dim3 threadsPerBlock{ 32, 1, 1 };
		const dim3 numBlocks{
			getNumberCudaBlocks(width, threadsPerBlock.x),
			getNumberCudaBlocks(height, threadsPerBlock.y),
			getNumberCudaBlocks(channels, threadsPerBlock.z) };
		reorderAndNormalizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr, srcPtr, width, height, channels);
	}
	catch (const std::exception& e)
	{
		//error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}



template void reorderAndNormalize(
	float* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);
template void reorderAndNormalize(
	double* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);



template <typename T>
__global__ void resizeAndPadKernel(
	T* targetPtr, const T* const sourcePtr, const int widthSource, const int heightSource, 
	const int widthTarget, const int heightTarget, const T rescaleFactor)
{
	const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (x < widthTarget && y < heightTarget) {
		const auto targetArea = widthTarget * heightTarget;
		if (x < widthSource * rescaleFactor && y < heightSource * rescaleFactor) {
			const auto sourceArea = widthSource * heightSource;
			const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
			const T* const sourcePtrChannel = sourcePtr + channel * sourceArea;
			targetPtr[channel * targetArea + y * widthTarget + x] = bicubicInterpolate(
				sourcePtrChannel, xSource, ySource, widthSource, heightSource, widthSource);
		}
		else
			targetPtr[channel * targetArea + y * widthTarget + x] = 0;
	}
}

template <typename T>
__global__ void resizeAndPadKernel(
	T* targetPtr, const unsigned char* const sourcePtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const T rescaleFactor)
{
	const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (x < widthTarget && y < heightTarget) {
		const auto targetArea = widthTarget * heightTarget;
		if (x < widthSource * rescaleFactor && y < heightSource * rescaleFactor) {
			const auto sourceArea = widthSource * heightSource;
			const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f); // xSource = x;
			const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f); // ySource = y;
			const unsigned char* sourcePtrChannel = sourcePtr + channel * sourceArea;
			targetPtr[channel * targetArea + y * widthTarget + x] = bicubicInterpolate(
				sourcePtrChannel, xSource, ySource, widthSource, heightSource, widthSource);
		}
		else
			targetPtr[channel * targetArea + y * widthTarget + x] = 0;
	}
}

template <typename T>
void resizeAndPadRbgGpu(
	T* targetPtr, const T* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const T scaleFactor)
{
	try
	{
		const auto channels = 3;
		const dim3 threadsPerBlock{ THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1 };
		const dim3 numBlocks{
			getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
			getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
			getNumberCudaBlocks(channels, threadsPerBlock.z) };
		resizeAndPadKernel<<<numBlocks, threadsPerBlock>>>(
			targetPtr, srcPtr, widthSource, heightSource, widthTarget, heightTarget, scaleFactor);
	}
	catch (const std::exception& e)
	{
		//error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

template <typename T>
void resizeAndPadRbgGpu(
	T* targetPtr, const unsigned char* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const T scaleFactor)
{
	try
	{
		const auto channels = 3;
		const dim3 threadsPerBlock{ THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1 };
		const dim3 numBlocks{
			getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
			getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
			getNumberCudaBlocks(channels, threadsPerBlock.z) };
		resizeAndPadKernel<<<numBlocks, threadsPerBlock>>>(
			targetPtr, srcPtr, widthSource, heightSource, widthTarget, heightTarget, scaleFactor);
	}
	catch (const std::exception& e)
	{
		//error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

template void resizeAndPadRbgGpu(
	float* targetPtr, const float* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const float scaleFactor);
template void resizeAndPadRbgGpu(
	double* targetPtr, const double* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const double scaleFactor);

template void resizeAndPadRbgGpu(
	float* targetPtr, const unsigned char* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const float scaleFactor);
template void resizeAndPadRbgGpu(
	double* targetPtr, const unsigned char* const srcPtr, const int widthSource, const int heightSource,
	const int widthTarget, const int heightTarget, const double scaleFactor);





template <typename T>
void resizeAndMergeGpu(
	T* targetPtr, const T* sourcePtr, const std::array<int, 4>& targetSize,
	const std::array<int, 4>& sourceSize, const T& scaleInputToNetInputs)
{
	// Parameters
	const auto channels = targetSize[1]; // here channels == sourceSize[1] == targetSize[1] must be (18+1)*3
	const auto heightTarget = targetSize[2];
	const auto widthTarget = targetSize[3];
	const auto heightSource = sourceSize[2];
	const auto widthSource = sourceSize[3];

	const auto num = sourceSize[0];
	// No multi-scale merging or no merging required
	const auto rescaleFactor = (unsigned int)std::ceil(heightTarget / (float)(heightSource)); // == 8
	const dim3 threadsPerBlock{ rescaleFactor, rescaleFactor, 1 };
	const dim3 numBlocks{
		getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
		getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
		getNumberCudaBlocks(num * channels, threadsPerBlock.z) };
	resize8TimesKernel<<<numBlocks, threadsPerBlock>>>(
		targetPtr, sourcePtr, widthSource, heightSource, widthTarget, heightTarget,
		rescaleFactor);
}

template void resizeAndMergeGpu(
	float* targetPtr, const float* sourcePtr, const std::array<int, 4>& targetSize,
	const std::array<int, 4>& sourceSize, const float& scaleInputToNetInputs);
template void resizeAndMergeGpu(
	double* targetPtr, const double* sourcePtr, const std::array<int, 4>& targetSize,
	const std::array<int, 4>& sourceSize, const double& scaleInputToNetInputs);
