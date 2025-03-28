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

/// \file GPUROOTSMatrixFwd.h
/// \author Matteo Concas

#ifndef GPUROOTSMATRIXFWD_H
#define GPUROOTSMATRIXFWD_H

// Standalone forward declarations for Svector / SMatrix / etc.
// To be used on GPU where ROOT is not available.

#include "GPUCommonDef.h"

namespace ROOT
{
namespace Math
{
template <typename T, uint32_t N>
class SVector;
template <class T, uint32_t D1, uint32_t D2, class R>
class SMatrix;
template <class T, uint32_t D>
class MatRepSym;
template <class T, uint32_t D1, uint32_t D2>
class MatRepStd;
} // namespace Math
} // namespace ROOT

namespace o2::math_utils
{

namespace detail
{
template <typename T, uint32_t N>
class SVectorGPU;
template <class T, uint32_t D1, uint32_t D2, class R>
class SMatrixGPU;
template <class T, uint32_t D>
class MatRepSymGPU;
template <class T, uint32_t D1, uint32_t D2>
class MatRepStdGPU;
} // namespace detail

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE) && !defined(GPUCOMMONRTYPES_H_ACTIVE)
template <typename T, uint32_t N>
using SVector = ROOT::Math::SVector<T, N>;
template <class T, uint32_t D1, uint32_t D2, class R>
using SMatrix = ROOT::Math::SMatrix<T, D1, D2, R>;
template <class T, uint32_t D>
using MatRepSym = ROOT::Math::MatRepSym<T, D>;
template <class T, uint32_t D1, uint32_t D2 = D1>
using MatRepStd = ROOT::Math::MatRepStd<T, D1, D2>;
#else
template <typename T, uint32_t N>
using SVector = detail::SVectorGPU<T, N>;
template <class T, uint32_t D1, uint32_t D2 = D1, class R = detail::MatRepStdGPU<T, D1, D2>>
using SMatrix = detail::SMatrixGPU<T, D1, D2, R>;
template <class T, uint32_t D>
using MatRepSym = detail::MatRepSymGPU<T, D>;
template <class T, uint32_t D1, uint32_t D2 = D1>
using MatRepStd = detail::MatRepStdGPU<T, D1, D2>;
#endif

} // namespace o2::math_utils

#endif
