#pragma once

#include <vector>
#include <memory>
#include <numeric>
using namespace std;

#include <cuda_runtime_api.h>

// store GPU data, for T type (since data type might have int, float)
template<typename T>
class ArrayGpu {
public:
	explicit ArrayGpu(const vector<int>& sizes, T* const dataPtr = nullptr, bool isFromCpu = true);
	ArrayGpu(){};
	~ArrayGpu();

	void reset(const vector<int>& sizes, T* const dataPtr = nullptr, bool isFromCpu = true);
	void destroy();

	inline size_t getDimensions() const	{ return mSize.size(); }
	inline vector<int> getSize() const { return mSize; }
	inline T* getPtr() { return pData; }
	inline const T* getConstPtr() const { return pData; }

	T* pData;

private:
	bool isInited;
	// channels, height, width
	vector<int> mSize;
	size_t mVolume;
	size_t mActualVolume;
};



// Implementation
template<typename T>
ArrayGpu<T>::ArrayGpu(const vector<int>& sizes, T* const dataPtr, bool isFromCpu)
	: isInited(false)
	, pData(nullptr)
{
	if (sizes.size() == 0) return;
	reset(sizes, dataPtr, isFromCpu);
}

template<typename T>
ArrayGpu<T>::~ArrayGpu() {
	destroy();
}

template<typename T>
void ArrayGpu<T>::destroy() {
	if (isInited) {
		cudaFree(pData);
		isInited = false;
	}
}

template<typename T>
void ArrayGpu<T>::reset(const vector<int>& sizes, T* const dataPtr, bool isFromCpu) {
	mSize = sizes;
	mVolume = accumulate(sizes.begin(), sizes.end(), size_t(1), multiplies<size_t>());
	if (isInited && mVolume > mActualVolume) {
		cudaFree(pData);
		isInited = false;
	}
	if (!isInited) {
		isInited = true;
		cudaMalloc((void**)&pData, mVolume * sizeof(T));
		mActualVolume = mVolume;
	}
	if (dataPtr == nullptr) return;
	cudaMemcpyKind kind = isFromCpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
	cudaMemcpy(pData, dataPtr, mVolume * sizeof(T), kind);
}
