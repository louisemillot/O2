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

/// \file qconfigrtc.h
/// \author David Rohr

#ifndef QCONFIG_RTC_H
#define QCONFIG_RTC_H

#include "qconfig.h"
#include "qconfig_helpers.h"

#ifndef qon_mxstr
#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)
#endif
#ifndef qon_mxcat
#define qon_mcat(a, b) a##b
#define qon_mxcat(a, b) qon_mcat(a, b)
#endif

template <class T>
static std::string qConfigPrintRtc(const T& tSrc, bool useConstexpr)
{
  std::stringstream out;
  out << std::hexfloat;
#define QCONFIG_PRINT_RTC
#include "qconfig.h"
#undef QCONFIG_PRINT_RTC
  return out.str();
}

#define QCONFIG_CONVERT_RTC
#include "qconfig.h"
#undef QCONFIG_CONVERT_RTC

#endif
