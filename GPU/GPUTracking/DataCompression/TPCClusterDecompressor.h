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

/// \file TPCClusterDecompressor.h
/// \author David Rohr

#ifndef TPCCLUSTERDECOMPRESSOR_H
#define TPCCLUSTERDECOMPRESSOR_H

#include "GPUTPCCompression.h"
#include <vector>
#include <functional>

namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace o2::tpc

namespace o2::gpu
{
struct GPUParam;

class TPCClusterDecompressor
{
 public:
  static constexpr uint32_t NSECTORS = GPUCA_NSECTORS;
  static int32_t decompress(const o2::tpc::CompressedClustersFlat* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param, bool deterministicRec);
  static int32_t decompress(const o2::tpc::CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param, bool deterministicRec);
};
} // namespace o2::gpu

#endif
