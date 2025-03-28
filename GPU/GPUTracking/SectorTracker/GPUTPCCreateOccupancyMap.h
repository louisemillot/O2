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

/// \file GPUTPCCreateOccupancyMap.h
/// \author David Rohr

#ifndef GPUTPCCREATEOCCUPANCYMAP_H
#define GPUTPCCREATEOCCUPANCYMAP_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
struct GPUTPCClusterOccupancyMapBin;

class GPUTPCCreateOccupancyMap : public GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           fill = 0,
           fold = 1 };
  GPUhdi() constexpr static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCSectorTracking; }
  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, Args... args);
};

} // namespace o2::gpu

#endif
