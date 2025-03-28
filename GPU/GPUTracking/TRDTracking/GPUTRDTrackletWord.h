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

/// \file GPUTRDTrackletWord.h
/// \brief TRD Tracklet word for GPU tracker - 32bit tracklet info + half chamber ID + index

/// \author Ole Schmidt

#ifndef GPUTRDTRACKLETWORD_H
#define GPUTRDTRACKLETWORD_H

#include "GPUDef.h"

#ifndef GPUCA_TPC_GEOMETRY_O2 // compatibility to Run 2 data types

class AliTRDtrackletWord;
class AliTRDtrackletMCM;

namespace o2::gpu
{

class GPUTRDTrackletWord
{
 public:
  GPUd() GPUTRDTrackletWord(uint32_t trackletWord = 0);
  GPUd() GPUTRDTrackletWord(uint32_t trackletWord, int32_t hcid);
  GPUdDefault() GPUTRDTrackletWord(const GPUTRDTrackletWord& rhs) = default;
  GPUdDefault() GPUTRDTrackletWord& operator=(const GPUTRDTrackletWord& rhs) = default;
  GPUdDefault() ~GPUTRDTrackletWord() = default;
#ifndef GPUCA_GPUCODE_DEVICE
  GPUTRDTrackletWord(const AliTRDtrackletWord& rhs);
  GPUTRDTrackletWord(const AliTRDtrackletMCM& rhs);
  GPUTRDTrackletWord& operator=(const AliTRDtrackletMCM& rhs);
#endif

  // ----- Override operators < and > to enable tracklet sorting by HCId -----
  GPUd() bool operator<(const GPUTRDTrackletWord& t) const { return (GetHCId() < t.GetHCId()); }
  GPUd() bool operator>(const GPUTRDTrackletWord& t) const { return (GetHCId() > t.GetHCId()); }
  GPUd() bool operator<=(const GPUTRDTrackletWord& t) const { return (GetHCId() < t.GetHCId()) || (GetHCId() == t.GetHCId()); }

  // ----- Getters for contents of tracklet word -----
  GPUd() int32_t GetYbin() const;
  GPUd() int32_t GetdYbin() const;
  GPUd() int32_t GetZbin() const { return ((mTrackletWord >> 20) & 0xf); }
  GPUd() int32_t GetPID() const { return ((mTrackletWord >> 24) & 0xff); }

  // ----- Getters for offline corresponding values -----
  GPUd() double GetPID(int32_t /* is */) const { return (double)GetPID() / 256.f; }
  GPUd() int32_t GetDetector() const { return mHCId / 2; }
  GPUd() int32_t GetHCId() const { return mHCId; }
  GPUd() float GetdYdX() const { return (GetdYbin() * 140e-4f / 3.f); }
  GPUd() float GetdY() const { return GetdYbin() * 140e-4f; }
  GPUd() float GetY() const { return (GetYbin() * 160e-4f); }
  GPUd() uint32_t GetTrackletWord() const { return mTrackletWord; }

  GPUd() void SetTrackletWord(uint32_t trackletWord) { mTrackletWord = trackletWord; }
  GPUd() void SetDetector(int32_t id) { mHCId = 2 * id + (GetYbin() < 0 ? 0 : 1); }
  GPUd() void SetHCId(int32_t id) { mHCId = id; }

 protected:
  int32_t mHCId;          // half-chamber ID
  uint32_t mTrackletWord; // tracklet word: PID | Z | deflection length | Y
                          //          bits:   8   4            7          13
};
} // namespace o2::gpu

#else // compatibility with Run 3 data types

#include "DataFormatsTRD/Tracklet64.h"

namespace o2::gpu
{

class GPUTRDTrackletWord : private o2::trd::Tracklet64
{
 public:
  GPUd() GPUTRDTrackletWord(uint64_t trackletWord = 0) : o2::trd::Tracklet64(trackletWord) {};
  GPUdDefault() GPUTRDTrackletWord(const GPUTRDTrackletWord& rhs) = default;
  GPUdDefault() GPUTRDTrackletWord& operator=(const GPUTRDTrackletWord& rhs) = default;
  GPUdDefault() ~GPUTRDTrackletWord() = default;

  // ----- Override operators < and > to enable tracklet sorting by HCId -----
  GPUd() bool operator<(const GPUTRDTrackletWord& t) const { return (getHCID() < t.getHCID()); }
  GPUd() bool operator>(const GPUTRDTrackletWord& t) const { return (getHCID() > t.getHCID()); }
  GPUd() bool operator<=(const GPUTRDTrackletWord& t) const { return (getHCID() < t.getHCID()) || (getHCID() == t.getHCID()); }

  GPUd() int32_t GetZbin() const { return getPadRow(); }
  GPUd() float GetY() const { return getUncalibratedY(); }
  GPUd() float GetdY() const { return getUncalibratedDy(); }
  GPUd() int32_t GetDetector() const { return getDetector(); }
  GPUd() int32_t GetHCId() const { return getHCID(); }

  // IMPORTANT: Do not add members, this class must keep the same memory layout as o2::trd::Tracklet64
};

static_assert(sizeof(GPUTRDTrackletWord) == sizeof(o2::trd::Tracklet64), "Incorrect memory layout");

} // namespace o2::gpu

#endif // GPUCA_TPC_GEOMETRY_O2

#endif // GPUTRDTRACKLETWORD_H
