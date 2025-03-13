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

/// \file GPUTPCTracker.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCTracker.h"
#include "GPUTPCRow.h"
#include "GPUTPCTrack.h"
#include "GPUCommonMath.h"

#include "GPUTPCClusterData.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCTrackParam.h"
#include "GPUParam.inc"
#include "GPUTPCConvertImpl.h"

#if !defined(GPUCA_GPUCODE)
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "GPUReconstruction.h"
#include "GPUMemorySizeScalers.h"
#endif

using namespace o2::gpu;
using namespace o2::tpc;

#if !defined(GPUCA_GPUCODE)

GPUTPCTracker::~GPUTPCTracker() = default;

// ----------------------------------------------------------------------------------
void GPUTPCTracker::SetSector(int32_t iSector) { mISector = iSector; }
void GPUTPCTracker::InitializeProcessor()
{
  if (mISector < 0) {
    throw std::runtime_error("Sector not set");
  }
  InitializeRows(&Param());
  SetupCommonMemory();
}

void* GPUTPCTracker::SetPointersDataLinks(void* mem) { return mData.SetPointersLinks(mem); }
void* GPUTPCTracker::SetPointersDataWeights(void* mem) { return mData.SetPointersWeights(mem); }
void* GPUTPCTracker::SetPointersDataScratch(void* mem) { return mData.SetPointersScratch(mem, mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging); }
void* GPUTPCTracker::SetPointersDataRows(void* mem) { return mData.SetPointersRows(mem); }

void* GPUTPCTracker::SetPointersScratch(void* mem)
{
  computePointerWithAlignment(mem, mTrackletStartHits, mNMaxStartHits);
  if (mRec->GetProcessingSettings().memoryAllocationStrategy != GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mem = SetPointersTracklets(mem);
  }
  if (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSectorTracking) {
    computePointerWithAlignment(mem, mTrackletTmpStartHits, GPUCA_ROW_COUNT * mNMaxRowStartHits);
    computePointerWithAlignment(mem, mRowStartHitCountOffset, GPUCA_ROW_COUNT);
  }
  return mem;
}

void* GPUTPCTracker::SetPointersScratchHost(void* mem)
{
  if (mRec->GetProcessingSettings().keepDisplayMemory) {
    computePointerWithAlignment(mem, mLinkTmpMemory, mRec->Res(mMemoryResLinks).Size());
  }
  mem = mData.SetPointersClusterIds(mem, mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging);
  return mem;
}

void* GPUTPCTracker::SetPointersCommon(void* mem)
{
  computePointerWithAlignment(mem, mCommonMem, 1);
  return mem;
}

void GPUTPCTracker::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  bool reuseCondition = !mRec->GetProcessingSettings().keepDisplayMemory && ((mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSectorTracking) || mRec->GetProcessingSettings().inKernelParallel == 1 || mRec->GetProcessingSettings().nHostThreads == 1);
  GPUMemoryReuse reLinks{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerDataLinks, (uint16_t)(mISector % mRec->GetProcessingSettings().nStreams)};
  mMemoryResLinks = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataLinks, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCSectorLinks", reLinks);
  mMemoryResSectorScratch = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK | GPUMemoryResource::MEMORY_CUSTOM, "TPCSectorScratch");
  GPUMemoryReuse reWeights{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerDataWeights, (uint16_t)(mISector % mRec->GetProcessingSettings().nStreams)};
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataWeights, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCSectorWeights", reWeights);
  GPUMemoryReuse reScratch{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerScratch, (uint16_t)(mISector % mRec->GetProcessingSettings().nStreams)};
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCTrackerScratch", reScratch);
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersScratchHost, GPUMemoryResource::MEMORY_SCRATCH_HOST, "TPCTrackerHost");
  mMemoryResCommon = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersCommon, GPUMemoryResource::MEMORY_PERMANENT, "TPCTrackerCommon");
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataRows, GPUMemoryResource::MEMORY_PERMANENT, "TPCSectorRows");

  uint32_t type = GPUMemoryResource::MEMORY_SCRATCH;
  if (mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) { // For individual scheme, we allocate tracklets separately, and change the type for the following allocations to custom
    type |= GPUMemoryResource::MEMORY_CUSTOM;
    mMemoryResTracklets = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersTracklets, type, "TPCTrackerTracklets");
  }
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersOutput, type, "TPCTrackerTracks");
}

GPUhd() void* GPUTPCTracker::SetPointersTracklets(void* mem)
{
  computePointerWithAlignment(mem, mTracklets, mNMaxTracklets);
  computePointerWithAlignment(mem, mTrackletRowHits, mNMaxRowHits);
  return mem;
}

GPUhd() void* GPUTPCTracker::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mTrackHits, mNMaxTrackHits);
  return mem;
}

void GPUTPCTracker::SetMaxData(const GPUTrackingInOutPointers& io)
{
  if (mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mNMaxStartHits = mData.NumberOfHits();
  } else {
    mNMaxStartHits = mRec->MemoryScalers()->NTPCStartHits(mData.NumberOfHits());
  }
  if (io.clustersNative) {
    uint32_t maxRowHits = 0;
    for (uint32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
      if (io.clustersNative->nClusters[mISector][i] > maxRowHits) {
        maxRowHits = io.clustersNative->nClusters[mISector][i];
      }
    }
    mNMaxRowStartHits = mRec->MemoryScalers()->NTPCRowStartHits(maxRowHits * GPUCA_ROW_COUNT);
  } else {
    mNMaxRowStartHits = mRec->MemoryScalers()->NTPCRowStartHits(mData.NumberOfHits());
  }
  mNMaxTracklets = mRec->MemoryScalers()->NTPCTracklets(mData.NumberOfHits());
  mNMaxRowHits = mRec->MemoryScalers()->NTPCTrackletHits(mData.NumberOfHits());
  mNMaxTracks = mRec->MemoryScalers()->NTPCSectorTracks(mData.NumberOfHits());
  mNMaxTrackHits = mRec->MemoryScalers()->NTPCSectorTrackHits(mData.NumberOfHits(), mRec->GetProcessingSettings().tpcInputWithClusterRejection);
#ifdef GPUCA_SORT_STARTHITS_GPU
  if (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSectorTracking) {
    if (mNMaxStartHits > mNMaxRowStartHits * GPUCA_ROW_COUNT) {
      mNMaxStartHits = mNMaxRowStartHits * GPUCA_ROW_COUNT;
    }
  }
#endif
  mData.SetMaxData();
}

void GPUTPCTracker::UpdateMaxData()
{
  mNMaxTracklets = mCommonMem->nStartHits;
  mNMaxTracks = mNMaxTracklets * 2 + 50;
  mNMaxRowHits = mNMaxTracklets * GPUCA_ROW_COUNT;
}

void GPUTPCTracker::SetupCommonMemory() { new (mCommonMem) commonMemoryStruct; }

GPUh() int32_t GPUTPCTracker::CheckEmptySector()
{
  // Check if the Sector is empty, if so set the output apropriate and tell the reconstuct procesdure to terminate
  if (NHitsTotal() < 1) {
    mCommonMem->nTracks = mCommonMem->nTrackHits = 0;
    return 1;
  }
  return 0;
}

#endif
