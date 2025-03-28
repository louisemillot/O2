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

/// \file GPUReconstructionIncludes.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLUDES_H
#define GPURECONSTRUCTIONINCLUDES_H

#ifndef WIN32
#include <sys/syscall.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sched.h>
#endif

#include "GPUDef.h"
#include "GPULogging.h"
#include "GPUDataTypes.h"

#include <iostream>
#include <fstream>

#define GPUCA_GPUReconstructionUpdateDefaults()                                             \
  if (mProcessingSettings.alternateBorderSort < 0) {                                        \
    mProcessingSettings.alternateBorderSort = GPUCA_ALTERNATE_BORDER_SORT;                  \
  }                                                                                         \
  if (mProcessingSettings.mergerSortTracks < 0) {                                           \
    mProcessingSettings.mergerSortTracks = GPUCA_SORT_BEFORE_FIT;                           \
  }                                                                                         \
  if (param().rec.tpc.looperInterpolationInExtraPass < 0) {                                 \
    param().rec.tpc.looperInterpolationInExtraPass = GPUCA_MERGER_SPLIT_LOOP_INTERPOLATION; \
  }                                                                                         \
  if (mProcessingSettings.tpcCompressionGatherModeKernel < 0) {                             \
    mProcessingSettings.tpcCompressionGatherModeKernel = GPUCA_COMP_GATHER_KERNEL;          \
  }                                                                                         \
  if (mProcessingSettings.tpcCompressionGatherMode < 0) {                                   \
    mProcessingSettings.tpcCompressionGatherMode = GPUCA_COMP_GATHER_MODE;                  \
  }

#endif
