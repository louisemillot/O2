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

/// \file TPCPadBitMap.h
/// \author Jens Wiechula

#ifndef O2_GPU_TPC_PAD_BITMAP_H
#define O2_GPU_TPC_PAD_BITMAP_H

#include "clusterFinderDefs.h"
#include "GPUCommonMath.h"

namespace o2::tpc
{
template <class T>
class CalDet;
} // namespace o2::tpc

namespace o2::gpu
{

struct TPCPadBitMap {
 public:
#ifndef GPUCA_GPUCODE
  TPCPadBitMap();
  TPCPadBitMap(const o2::tpc::CalDet<bool>&);

  /// setting the stored values from CalDet
  void setFromMap(const o2::tpc::CalDet<bool>&);
#endif

  GPUdi() void set(int32_t sector, tpccf::Row row, tpccf::Pad pad, bool c)
  {
    mBitMap[sector].set(globalPad(row, pad), c);
  }

  GPUdi() void set(int32_t sector, uint16_t globalPad, bool c)
  {
    mBitMap[sector].set(globalPad, c);
  }

  GPUdi() bool isSet(int32_t sector, tpccf::Row row, tpccf::Pad pad) const
  {
    return mBitMap[sector].get(globalPad(row, pad));
  }

  GPUdi() bool isSet(int32_t sector, uint16_t globalPad) const
  {
    return mBitMap[sector].get(globalPad);
  }

  GPUdi() uint16_t globalPad(tpccf::Row row, tpccf::Pad pad) const
  {
    return mPadOffsetPerRow[row] + pad;
  }

 private:
  /// Flat dead channel map, one value per pad in sector
  class SectorBitMap
  {
   public:
    using T = uint32_t;
    static constexpr int32_t NWORDS = (TPC_PADS_IN_SECTOR + sizeof(T) * 8 - 1) / sizeof(T);
    GPUdi() SectorBitMap()
    {
      reset();
    }

    GPUdi() void set(uint16_t globalPad, bool c)
    {
      const auto word = globalPad / (sizeof(T) * 8);
      const auto pos = globalPad % (sizeof(T) * 8);
      const auto mask = T(1) << pos;
      mDeadChannelMap[word] = (mDeadChannelMap[word] & ~mask) | (T(c) << pos);
    }

    GPUdi() bool get(uint16_t globalPad) const
    {
      const auto word = globalPad / (sizeof(T) * 8);
      const auto pos = globalPad % (sizeof(T) * 8);
      const auto mask = T(1) << pos;
      return (mDeadChannelMap[word] & mask) == mask;
    }

    GPUd() void reset()
    {
      for (int32_t iword = 0; iword < NWORDS; ++iword) {
        mDeadChannelMap[iword] = 0;
      }
    }

   private:
    T mDeadChannelMap[NWORDS];
  };

  uint16_t mPadOffsetPerRow[GPUCA_ROW_COUNT];
  SectorBitMap mBitMap[GPUCA_NSECTORS];
};

} // namespace o2::gpu

#endif
