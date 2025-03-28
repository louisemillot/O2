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

/// \file GPUTriggerOutputs.h
/// \author David Rohr

#ifndef GPUTRIGGEROUTPUTS_H
#define GPUTRIGGEROUTPUTS_H

#include "GPUCommonDef.h"
#include "DataFormatsTPC/ZeroSuppression.h"

#include <unordered_set>
#include <array>

namespace o2::gpu
{

struct GPUTriggerOutputs {
  struct hasher {
    size_t operator()(const o2::tpc::TriggerInfoDLBZS& key) const
    {
      std::array<uint32_t, sizeof(key) / sizeof(uint32_t)> tmp;
      memcpy((void*)tmp.data(), (const void*)&key, sizeof(key));
      std::hash<uint32_t> std_hasher;
      size_t result = 0;
      for (size_t i = 0; i < tmp.size(); ++i) {
        result ^= std_hasher(tmp[i]);
      }
      return result;
    }
  };

  struct equal {
    bool operator()(const o2::tpc::TriggerInfoDLBZS& lhs, const o2::tpc::TriggerInfoDLBZS& rhs) const
    {
      return memcmp((const void*)&lhs, (const void*)&rhs, sizeof(lhs)) == 0;
    }
  };

  std::unordered_set<o2::tpc::TriggerInfoDLBZS, hasher, equal> triggers;
  static_assert(sizeof(o2::tpc::TriggerInfoDLBZS) % sizeof(uint32_t) == 0);
};

} // namespace o2::gpu

#endif
