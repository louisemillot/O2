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

/// \file GPUTPCStartHitsSorter.h
/// \author David Rohr

#ifndef GPUTPCSTARTHITSSORTER_H
#define GPUTPCSTARTHITSSORTER_H

#include "GPUTPCDef.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

/**
 * @class GPUTPCStartHitsSorter
 *
 */
class GPUTPCStartHitsSorter : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory {
    int32_t mStartRow;    // start row index
    int32_t mNRows;       // number of rows to process
    int32_t mStartOffset; // start offset for hits sorted by this block
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static GPUDataTypes::RecoStep GetRecoStep() { return GPUDataTypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);
};
} // namespace o2::gpu

#endif // GPUTPCSTARTHITSSORTER_H
