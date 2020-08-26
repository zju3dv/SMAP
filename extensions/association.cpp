#include <math.h>
#include <memory>
#include <vector>
#include <tuple>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include "arraygpu.hpp"
#include "gpu/cuda_cal.h"
// #include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
// #include "Timer.hpp"

namespace py= pybind11;
using namespace std;
using Tensor = torch::Tensor;


const int nJoints = 15;
const int nLimbs = 14;
const int maxPeaks = 127;
vector<int> heatmapDim = {43, 128, 208};  // 1/4 height, 1/4 width
const float dsScale = 4;
vector<unsigned int> jointPairs = {0, 1,   0, 2,   0, 9,   9, 10,   10, 11,
		                           0, 3,   3, 4,   4, 5,   2, 12,   12, 13, 
		                           13,14,  2, 6,   6, 7,   7, 8};
// statistic bone length 
vector<float> bone_length = {26.42178982, 48.36980909,
                             14.88291009, 31.28002332, 23.915707,
                             14.97674918, 31.28002549, 23.91570732,
                             12.4644364,  48.26604433, 39.03553194,
                             12.4644364, 48.19076948, 39.03553252};


tuple< vector<Tensor>, vector<Tensor> > extract(Tensor &hmsIn)
{
    
    const float nmsThreshold = 0.2f;
    const float nmsOffset = 0.5f;
    
	vector<unsigned int> mapIdx; // idx of paf
	auto mapOffset = nJoints;    // no bkg
    for (auto i = 0; i < nLimbs; i++) {
        mapIdx.push_back(mapOffset + 2*i);
        mapIdx.push_back(mapOffset + 2*i+1);
    }

    shared_ptr<ArrayGpu<float>> peaks(new ArrayGpu<float>({ nJoints, maxPeaks + 1, 3 }));
    shared_ptr<ArrayGpu<int>> peakKernal(new ArrayGpu<int>({ heatmapDim[0], heatmapDim[1], heatmapDim[2] }));
    shared_ptr<ArrayGpu<float>> heatMap(new ArrayGpu<float>({ heatmapDim[0], heatmapDim[1], heatmapDim[2] }));

    cudaMemcpy(heatMap->getPtr(), hmsIn.data_ptr<float>(), heatmapDim[0]*heatmapDim[1]*heatmapDim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    array<int, 4> peakSourceSize{ 1, nJoints, heatmapDim[1], heatmapDim[2] };
    array<int, 4> peakTargetSize{ 1, nJoints, maxPeaks + 1, 3 };

    nmsGpu(peaks->getPtr(), peakKernal->getPtr(), heatMap->getPtr(), nmsThreshold, peakTargetSize, peakSourceSize, nmsOffset);

    // get PAF scores
    shared_ptr<ArrayGpu<float>> pairScoresGpuPtr(new ArrayGpu<float>({ nLimbs, maxPeaks, maxPeaks }));
    shared_ptr<ArrayGpu<unsigned int>> pBodyPartPairsGpuPtr(new ArrayGpu<unsigned int>({ (int)jointPairs.size() })); // 1D
	cudaMemcpy(pBodyPartPairsGpuPtr->getPtr(), jointPairs.data(), jointPairs.size() * sizeof(unsigned int),
		cudaMemcpyHostToDevice);
    shared_ptr<ArrayGpu<unsigned int>> pMapIdxGpuPtr(new ArrayGpu<unsigned int>({ (int)mapIdx.size() })); // 1D
	cudaMemcpy(pMapIdxGpuPtr->getPtr(), mapIdx.data(), mapIdx.size() * sizeof(unsigned int),
		cudaMemcpyHostToDevice);

    connectBodyPartsGpu(pairScoresGpuPtr->getPtr(),
        heatMap->getConstPtr(), heatMap->getSize()[2], heatMap->getSize()[1], peaks->getConstPtr(),
        pBodyPartPairsGpuPtr->getConstPtr(), pMapIdxGpuPtr->getConstPtr());

    // gpu --> cpu
    unique_ptr<float> peaksUpData(new float[(maxPeaks+1)*3*nJoints]);
    float* peaksCpuPtr = peaksUpData.get();
    cudaMemcpy(peaksCpuPtr, peaks->getConstPtr(), 
               sizeof(float)*nJoints*(maxPeaks+1)*3, cudaMemcpyDeviceToHost);
    
    // save nms
    vector<Tensor> poseCandidates;
    // #pragma omp parallel for  
    for (int i = 0; i < nJoints; i++) {
        float* curPeaksCpuPtr = peaksCpuPtr + i * (maxPeaks + 1) * 3;
        int mPeakSize = curPeaksCpuPtr[0];
        Tensor mPeakData = torch::empty({mPeakSize, 3});
        for (int i = 1; i <= mPeakSize; i++) {
            mPeakData[i-1][0] = curPeaksCpuPtr[3*i];
            mPeakData[i-1][1] = curPeaksCpuPtr[3*i+1];
            mPeakData[i-1][2] = curPeaksCpuPtr[3*i+2];
        }
        poseCandidates.push_back(mPeakData);
    }

    // gpu --> cpu
    unique_ptr<float> pariUpData(new float[maxPeaks*maxPeaks*nLimbs]);
	float* pairScoresCpuPtr = pariUpData.get();
	cudaMemcpy(pairScoresCpuPtr, pairScoresGpuPtr->getConstPtr(), 
               sizeof(float)*nLimbs*maxPeaks*maxPeaks, cudaMemcpyDeviceToHost);
    
    // save paf score
    vector<Tensor> pafCandidates(nLimbs);
    // #pragma omp parallel for
    for (int i = 0; i < nLimbs; i++) {
        float* curPairScoresCpuPtr = pairScoresCpuPtr + i*maxPeaks*maxPeaks;
        int joint1Idx = jointPairs[2*i];
        int joint2Idx = jointPairs[2*i+1];
        int nPeaks1 = poseCandidates[joint1Idx].sizes()[0];
        int nPeaks2 = poseCandidates[joint2Idx].sizes()[0];
        Tensor mPafData = torch::empty({nPeaks1, nPeaks2});
        for (int i = 0; i < nPeaks1; i++) {
            for (int j = 0; j < nPeaks2; j++) {
                float score = curPairScoresCpuPtr[maxPeaks*i+j];
                mPafData[i][j] = score;
            }
        }
        pafCandidates[i] = mPafData;
    }

    tuple< vector<Tensor>, vector<Tensor> > resInfo = {poseCandidates, pafCandidates};

    return resInfo;
}


Tensor findConnectedJoints(Tensor hmsIn, Tensor rDepth, int rootIdx = 2, bool distFlag = true)
{
    // rootIdx: 2 for pelvis, 0 for neck

    tuple< vector<Tensor>, vector<Tensor> > resInfo = extract(hmsIn);

    vector<Tensor> peaks = get<0>(resInfo);
    vector<Tensor> pafScores = get<1>(resInfo);

    int personNum = (int)peaks[rootIdx].sizes()[0];
    if (personNum == 0) {
        Tensor empty = torch::empty({0});
        return empty;
    }

    Tensor predRootDepth = torch::empty({personNum});
    for (int i = 0; i < personNum; i++) {
        Tensor dep = rDepth[ peaks[rootIdx][i][1].item().to<int>() ][ peaks[rootIdx][i][0].item().to<int>() ];
        predRootDepth[i] = dep;
    }
    // ordinal prior
    auto sortDep = predRootDepth.sort(0, false);
    auto sortDepth = get<0>(sortDep);
    auto sortIndex = get<1>(sortDep);

    vector<vector<int>> remap(nJoints);               
    for (int i = 0; i < nJoints; i++) {
        for (int j = 0; j < personNum; j++) {
            if (i == rootIdx)  remap[i].push_back(sortIndex[j].item().to<int>());
            else  remap[i].push_back(j);
        }
    }

    Tensor predBodys = torch::zeros({personNum, nJoints, 4});
    for (int i = 0; i < personNum; i++) {
        int sidx = sortIndex[i].item().to<int>();
        predBodys[i][rootIdx][0] = peaks[rootIdx][sidx][0];
        predBodys[i][rootIdx][1] = peaks[rootIdx][sidx][1];
        predBodys[i][rootIdx][3] = peaks[rootIdx][sidx][2];
    }

    for (int j = 0; j < nLimbs; j++) {
        int i, srcJointId, dstJointId;
        bool flip = false;
        // ATTN: messy !!!
        if (j == 0)  i = 1;  
        else if (j == 1)  i = 0;
        else  i = j;
        if ( rootIdx == 2 && i == 1 ) {
            srcJointId = jointPairs[2*i+1];
            dstJointId = jointPairs[2*i];
            flip = true;
        } else {                                   
            srcJointId = jointPairs[2*i];
            dstJointId = jointPairs[2*i+1];
        }

        vector<int> remapSrc = remap[srcJointId];

        Tensor srcList = predBodys.select(1, srcJointId); 
        Tensor dstList = peaks[dstJointId];                
        int dstSize = dstList.sizes()[0];
        if (dstSize == 0)  continue;

        Tensor curPafScore = pafScores[i];                 
        vector<int> used(dstSize, 0);
        for (int k1 = 0; k1 < srcList.sizes()[0]; k1++) {
            if (srcList[k1][3].item().to<float>() < 1e-5)
                continue;

            vector<float> srcJoint;
            if (distFlag) {
                srcJoint = {srcList[k1][0].item().to<float>(), srcList[k1][1].item().to<float>()};
            }

            float bone_dist = 1.2 * bone_length[i] / sortDepth[k1].item().to<float>();
            float maxScore = 0.0;
            int maxIdx = -1;
            for (int k2 = 0; k2 < dstList.sizes()[0]; k2++) {
                if (used[k2])  continue;
                float score;
                if (flip)  score = curPafScore[k2][remapSrc[k1]].item().to<float>();
                else  score = curPafScore[remapSrc[k1]][k2].item().to<float>();

                if (distFlag) {
                    if (score > 0) {
                        vector<float> dstJoint = {dstList[k2][0].item().to<float>(), dstList[k2][1].item().to<float>()};
                        float limb_dist = sqrt(pow(srcJoint[0] - dstJoint[0], 2) + pow(srcJoint[1] - dstJoint[1], 2));
                        // adaptive distance constraint
                        score += min(bone_dist / limb_dist / dsScale - 1, 0.0f);
                    }
                }
                if (score > maxScore) {
                    maxScore = score;
                    maxIdx = k2;
                }
            }
            if (maxScore > 0) {   // can be tuned, 0 as threshold
                predBodys[k1][dstJointId][0] = peaks[dstJointId][maxIdx][0];
                predBodys[k1][dstJointId][1] = peaks[dstJointId][maxIdx][1];
                predBodys[k1][dstJointId][3] = peaks[dstJointId][maxIdx][2]; 
                // remap
                remap[dstJointId][k1] = maxIdx;

                used[maxIdx] = 1;
            }
        }
    }

    return predBodys;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("extract", &extract);
    m.def("connect", &findConnectedJoints, py::arg("hmsIn"), py::arg("rDepth"), 
                                           py::arg("rootIdx")=2, py::arg("distFlag")=true);
}
