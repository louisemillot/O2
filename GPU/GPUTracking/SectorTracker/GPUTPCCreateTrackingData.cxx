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

/// \file GPUTPCCreateTrackingData.cxx
/// \author David Rohr

#include "GPUTPCCreateTrackingData.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCCreateTrackingData::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  tracker.Data().InitFromClusterData(nBlocks, nThreads, iBlock, iThread, tracker.GetConstantMem(), tracker.ISector(), s.tmp);
}
