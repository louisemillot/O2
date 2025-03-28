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

/// \file ChargeMapFiller.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFChargeMapFiller.h"
#include "ChargePos.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCPadGainCalib.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillIndexMap>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uint32_t> indexMap(clusterer.mPindexMap);
  fillIndexMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPmemory->fragment, clusterer.mPdigits, indexMap, clusterer.mPmemory->counters.nDigitsInFragment);
}

GPUd() void GPUTPCCFChargeMapFiller::fillIndexMapImpl(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread,
                                                      const CfFragment& fragment,
                                                      const tpc::Digit* digits,
                                                      Array2D<uint32_t>& indexMap,
                                                      size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  CPU_ONLY(idx += fragment.digitsStart);
  CPU_ONLY(tpc::Digit digit = digits[idx]);
  CPU_ONLY(ChargePos pos(digit.getRow(), digit.getPad(), fragment.toLocal(digit.getTimeStamp())));
  CPU_ONLY(indexMap.safeWrite(pos, idx));
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillFromDigits>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  fillFromDigitsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, clusterer.mPmemory->counters.nPositions, clusterer.mPdigits, clusterer.mPpositions, chargeMap);
}

GPUd() void GPUTPCCFChargeMapFiller::fillFromDigitsImpl(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, processorType& clusterer, const CfFragment& fragment, size_t digitNum,
                                                        const tpc::Digit* digits,
                                                        ChargePos* positions,
                                                        Array2D<PackedCharge>& chargeMap)
{
  size_t idx = get_global_id(0);
  if (idx >= digitNum) {
    return;
  }
  tpc::Digit digit = digits[fragment.digitsStart + idx];

  ChargePos pos(digit.getRow(), digit.getPad(), fragment.toLocal(digit.getTimeStamp()));
  positions[idx] = pos;
  float q = digit.getChargeFloat();
  q *= clusterer.GetConstantMem()->calibObjects.tpcPadGain->getGainCorrection(clusterer.mISector, digit.getRow(), digit.getPad());
  chargeMap[pos] = PackedCharge(q);
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::findFragmentStart>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t setPositions)
{
  if (iThread != 0 || iBlock != 0) {
    return;
  }

  size_t nDigits = clusterer.mPmemory->counters.nDigits;
  const tpc::Digit* digits = clusterer.mPdigits;
  size_t st = findTransition(clusterer.mPmemory->fragment.first(), digits, nDigits, 0);
  size_t end = findTransition(clusterer.mPmemory->fragment.last(), digits, nDigits, st);

  clusterer.mPmemory->fragment.digitsStart = st;

  size_t elems = end - st;

  clusterer.mPmemory->counters.nDigitsInFragment = elems;
  if (setPositions) {
    clusterer.mPmemory->counters.nPositions = elems;
  }
}

GPUd() size_t GPUTPCCFChargeMapFiller::findTransition(int32_t time, const tpc::Digit* digits, size_t nDigits, size_t lower)
{
  if (!nDigits) {
    return 0;
  }
  size_t upper = nDigits - 1;

  while (lower < upper) {
    size_t middle = (lower + upper) / 2;

    if (digits[middle].getTimeStamp() < time) {
      lower = middle + 1;
    } else if (middle == lower || digits[middle - 1].getTimeStamp() < time) {
      return middle;
    } else {
      upper = middle - 1;
    }
  }
  return lower;
}
