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

/// \file GPUMemoryResource.h
/// \author David Rohr

#ifndef GPUMEMORYRESOURCE_H
#define GPUMEMORYRESOURCE_H

#include "GPUCommonDef.h"
#include "GPUProcessor.h"

namespace o2::gpu
{

struct GPUMemoryReuse {
  enum Type : int32_t {
    NONE = 0,
    REUSE_1TO1 = 1
  };
  enum Group : uint16_t {
    ClustererScratch,
    NNClusterer,
    ClustererZS,
    TrackerScratch,
    TrackerDataLinks,
    TrackerDataWeights
  };
  using ID = uint32_t;

  GPUMemoryReuse(Type t, Group g, uint16_t i) : type(t), id(((uint32_t)g << 16) | ((uint32_t)i & 0xFFFF)) {}
  GPUMemoryReuse(bool condition, Type t, Group g, uint16_t i) : GPUMemoryReuse()
  {
    if (condition) {
      *this = GPUMemoryReuse{t, g, i};
    }
  }
  constexpr GPUMemoryReuse() = default;

  Type type = NONE;
  ID id = 0;
};

class GPUMemoryResource
{
  friend class GPUReconstruction;
  friend class GPUReconstructionCPU;

 public:
  enum MemoryType {
    MEMORY_HOST = 1,
    MEMORY_GPU = 2,
    MEMORY_INPUT_FLAG = 4,
    MEMORY_INPUT = 7,
    MEMORY_OUTPUT_FLAG = 8,
    MEMORY_OUTPUT = 11,
    MEMORY_INOUT = 15,
    MEMORY_SCRATCH = 16,
    MEMORY_SCRATCH_HOST = 17,
    MEMORY_EXTERNAL = 32,
    MEMORY_PERMANENT = 64,
    MEMORY_CUSTOM = 128,
    MEMORY_CUSTOM_TRANSFER = 256,
    MEMORY_STACK = 512
  };
  enum AllocationType { ALLOCATION_AUTO = 0,
                        ALLOCATION_INDIVIDUAL = 1,
                        ALLOCATION_GLOBAL = 2 };

  GPUMemoryResource(GPUProcessor* proc, void* (GPUProcessor::*setPtr)(void*), MemoryType type, const char* name = "") : mProcessor(proc), mPtr(nullptr), mPtrDevice(nullptr), mSetPointers(setPtr), mName(name), mSize(0), mOverrideSize(0), mReuse(-1), mType(type)
  {
  }
  GPUMemoryResource(const GPUMemoryResource&) = default;

  void* SetPointers(void* ptr)
  {
    return (mProcessor->*mSetPointers)(ptr);
  }
  void* SetDevicePointers(void* ptr) { return (mProcessor->mLinkedProcessor->*mSetPointers)(ptr); }
  void* Ptr() { return mPtr; }
  void* PtrDevice() { return mPtrDevice; }
  size_t Size() const { return mSize; }
  const char* Name() const { return mName; }
  MemoryType Type() const { return mType; }

 private:
  GPUProcessor* mProcessor;
  void* mPtr;
  void* mPtrDevice;
  void* (GPUProcessor::*mSetPointers)(void*);
  const char* mName;
  size_t mSize;
  size_t mOverrideSize;
  int32_t mReuse;
  MemoryType mType;
};
} // namespace o2::gpu

#endif
