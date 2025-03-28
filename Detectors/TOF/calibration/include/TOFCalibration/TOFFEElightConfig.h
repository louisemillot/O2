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

#ifndef DETECTOR_TOFFEELIGHTCONFIG_H_
#define DETECTOR_TOFFEELIGHTCONFIG_H_

#include "Rtypes.h"
#include "TOFBase/Geo.h"
#include <array>

using namespace o2::tof;

namespace o2
{
namespace tof
{

struct TOFFEEchannelConfig {

  enum EStatus_t {
    kStatusEnabled = 0x1
  };
  unsigned char mStatus = 0x0; // status
  int mMatchingWindow = 0;     // matching window [ns] // can this be int32?
  int mLatencyWindow = 0;      // latency window [ns] // can this be int32?
  TOFFEEchannelConfig() = default;
  bool isEnabled() const { return mStatus & kStatusEnabled; };

  ClassDefNV(TOFFEEchannelConfig, 1);
};

//_____________________________________________________________________________

struct TOFFEEtriggerConfig {

  unsigned int mStatusMap = 0; // status // can it be uint32?
  TOFFEEtriggerConfig() = default;

  ClassDefNV(TOFFEEtriggerConfig, 1);
};

//_____________________________________________________________________________

struct TOFFEEmapHVConfig {

  unsigned int mHVstat[Geo::NPLATES]; // 1 bit per strip status inside 5 modules
  TOFFEEmapHVConfig() = default;

  ClassDefNV(TOFFEEmapHVConfig, 1);
};

//_____________________________________________________________________________

struct TOFFEElightConfig {

  static constexpr int NCHANNELS = 172800;
  static constexpr int NTRIGGERMAPS = Geo::kNCrate;

  int mVersion = 0;   // version
  int mRunNumber = 0; // run number
  int mRunType = 0;   // run type

  // std::array<TOFFEEchannelConfig, NCHANNELS> mChannelConfig;
  TOFFEEchannelConfig mChannelConfig[Geo::kNCrate][Geo::kNTRM - 2][Geo::kNChain][Geo::kNTdc][Geo::kNCh]; // in O2, the number of TRMs is 12, but in the FEE world it is 10
  TOFFEEtriggerConfig mTriggerConfig[NTRIGGERMAPS];
  TOFFEEmapHVConfig mHVConfig[Geo::NSECTORS];
  TOFFEElightConfig() = default;
  const TOFFEEchannelConfig* getChannelConfig(int icrate, int itrm, int ichain, int itdc, int ich) const;
  const TOFFEEtriggerConfig* getTriggerConfig(int idx) const { return idx < NTRIGGERMAPS ? &mTriggerConfig[idx] : nullptr; }
  const TOFFEEmapHVConfig* getHVConfig(int isector) const { return (isector < Geo::NSECTORS) ? &mHVConfig[isector] : nullptr; }
  unsigned int getHVConfig(int isector, int iplate) const { return (isector < Geo::NSECTORS && iplate < Geo::NPLATES) ? mHVConfig[isector].mHVstat[iplate] : 0; }
  ClassDefNV(TOFFEElightConfig, 2);
};

} // namespace tof
} // namespace o2

#endif
