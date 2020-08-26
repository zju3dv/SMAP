#pragma once

#include <string>
#include <array>
#include <utility> // std::pair

const auto CUDA_NUM_THREADS = 512u;

//void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

//int getCudaGpuNumber();

inline unsigned int getNumberCudaBlocks(
	const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
{
	return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
}

/*
void getNumberCudaThreadsAndBlocks(
	dim3& numberCudaThreads, dim3& numberCudaBlocks, const Point<int>& frameSize);
*/

// Calculate peaks
template <typename T>
void nmsGpu(
	T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize,
	const std::array<int, 4>& sourceSize, const T& offset);


// body part connection
template <typename T>
void connectBodyPartsGpu(T* pairScoresGpuPtr,
	const T* const heatMapGpuPtr, const int& heatMapSizeW, const int& heatMapSizeH, const T* const peaksGpuPtr,
	const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr);
