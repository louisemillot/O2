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

/// \file GPUdEdxInfo.h
/// \author David Rohr

#ifndef GPUDEDXINFO_H
#define GPUDEDXINFO_H

#ifdef GPUCA_HAVE_O2HEADERS
#include "DataFormatsTPC/dEdxInfo.h"
#endif

namespace o2
{
namespace gpu
{
#ifdef GPUCA_HAVE_O2HEADERS
using GPUdEdxInfo = o2::tpc::dEdxInfo;
#else
struct GPUdEdxInfo {
};
#endif
} // namespace gpu
} // namespace o2

#endif
