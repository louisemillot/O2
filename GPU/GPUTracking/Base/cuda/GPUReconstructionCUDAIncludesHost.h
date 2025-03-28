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

/// \file GPUReconstructionCUDAIncludes.h
/// \author David Rohr

#ifndef O2_GPU_GPURECONSTRUCTIONCUDAINCLUDES_H
#define O2_GPU_GPURECONSTRUCTIONCUDAINCLUDES_H

#include <cstdint>
#include <type_traits>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#pragma GCC diagnostic push // FIXME: Is this still needed?
#pragma GCC diagnostic ignored "-Wshadow"
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop
#include <sm_20_atomic_functions.h>
#include <cuda_fp16.h>

#ifndef GPUCA_RTC_CODE
#include "GPUReconstructionCUDADef.h"
#endif

#endif
