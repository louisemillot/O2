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

/// \file GPUTRDTrackerKernels.cxx
/// \author David Rohr

#include "GPUTRDTrackerKernels.h"
#include "GPUTRDGeometry.h"
#include "GPUConstantMem.h"
#include "GPUCommonTypeTraits.h"

#include "GPUReconstructionThreading.h"

using namespace o2::gpu;

template <int32_t I, class T>
GPUdii() void GPUTRDTrackerKernels::Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, T* externalInstance)
{
  auto* trdTracker = &processors.getTRDTracker<I>();
#ifndef GPUCA_GPUCODE_DEVICE
  if constexpr (std::is_same_v<decltype(trdTracker), decltype(externalInstance)>) {
    if (externalInstance) {
      trdTracker = externalInstance;
    }
  }
#endif
  GPUCA_TBB_KERNEL_LOOP(trdTracker->GetRec(), int32_t, i, trdTracker->NTracks(), {
    trdTracker->DoTrackingThread(i, get_global_id(0));
  });
}

#if !defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_DEVICE) // FIXME: DR: WORKAROUND to avoid CUDA bug creating host symbols for device code.
template GPUdni() void GPUTRDTrackerKernels::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUTRDTrackerGPU* externalInstance);
template GPUdni() void GPUTRDTrackerKernels::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, GPUTRDTracker* externalInstance);
#endif
