// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCDecompressionKernels.h
/// \author Gabriele Cimador

#ifndef GPUTPCDECOMPRESSIONKERNELS_H
#define GPUTPCDECOMPRESSIONKERNELS_H

#include "GPUGeneralKernels.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUConstantMem.h"
#include "DataFormatsTPC/CompressedClusters.h"

namespace o2::gpu
{

class GPUTPCDecompressionKernels : public GPUKernelTemplate
{
 public:
  GPUhdi() constexpr static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCDecompression; }

  enum K : int32_t {
    step0attached = 0,
    step1unattached = 1,
  };

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, Args... args);

  GPUd() static uint32_t computeLinearTmpBufferIndex(uint32_t sector, uint32_t row, uint32_t maxClustersPerBuffer)
  {
    return sector * (GPUCA_ROW_COUNT * maxClustersPerBuffer) + row * maxClustersPerBuffer;
  }

  template <typename T>
  GPUdi() static void decompressorMemcpyBasic(T* dst, const T* src, uint32_t size);
};

class GPUTPCDecompressionUtilKernels : public GPUKernelTemplate
{
 public:
  enum K : int32_t {
    countFilteredClusters = 0,
    storeFilteredClusters = 1,
    sortPerSectorRow = 2,
  };

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors);

  GPUdi() static bool isClusterKept(const o2::tpc::ClusterNative& cl, const GPUParam& GPUrestrict() param);
};

} // namespace o2::gpu
#endif // GPUTPCDECOMPRESSIONKERNELS_H
