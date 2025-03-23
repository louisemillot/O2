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

/// \file GPUChainTrackingClusterizer.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUChainTrackingDefs.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include "GPUNewCalibValues.h"
#include <fstream>

#ifdef GPUCA_O2_LIB
#include "CommonDataFormat/InteractionRecord.h"
#endif
#include "GPUTriggerOutputs.h"
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/RDHUtils.h"

#include "utils/strtag.h"

#ifndef GPUCA_NO_VC
#include <Vc/Vc>
#endif

#ifdef GPUCA_HAS_ONNX
#include "GPUTPCNNClusterizerKernels.h"
#include "GPUTPCNNClusterizerHost.h"
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::tpc::constants;
using namespace o2::dataformats;

#ifdef GPUCA_TPC_GEOMETRY_O2
std::pair<uint32_t, uint32_t> GPUChainTracking::TPCClusterizerDecodeZSCountUpdate(uint32_t iSector, const CfFragment& fragment)
{
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
  GPUTPCClusterFinder::ZSOffset* o = processors()->tpcClusterer[iSector].mPzsOffsets;
  uint32_t digits = 0;
  uint32_t pages = 0;
  for (uint16_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    clusterer.mMinMaxCN[j] = mCFContext->fragmentData[fragment.index].minMaxCN[iSector][j];
    if (doGPU) {
      uint16_t posInEndpoint = 0;
      uint16_t pagesEndpoint = 0;
      for (uint32_t k = clusterer.mMinMaxCN[j].zsPtrFirst; k < clusterer.mMinMaxCN[j].zsPtrLast; k++) {
        const uint32_t pageFirst = (k == clusterer.mMinMaxCN[j].zsPtrFirst) ? clusterer.mMinMaxCN[j].zsPageFirst : 0;
        const uint32_t pageLast = (k + 1 == clusterer.mMinMaxCN[j].zsPtrLast) ? clusterer.mMinMaxCN[j].zsPageLast : mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k];
        for (uint32_t l = pageFirst; l < pageLast; l++) {
          uint16_t pageDigits = mCFContext->fragmentData[fragment.index].pageDigits[iSector][j][posInEndpoint++];
          if (pageDigits) {
            *(o++) = GPUTPCClusterFinder::ZSOffset{digits, j, pagesEndpoint};
            digits += pageDigits;
          }
          pagesEndpoint++;
        }
      }
      if (pagesEndpoint != mCFContext->fragmentData[fragment.index].pageDigits[iSector][j].size()) {
        if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
          GPUError("TPC raw page count mismatch in TPCClusterizerDecodeZSCountUpdate: expected %d / buffered %lu", pagesEndpoint, mCFContext->fragmentData[fragment.index].pageDigits[iSector][j].size());
          return {0, 0};
        } else {
          GPUFatal("TPC raw page count mismatch in TPCClusterizerDecodeZSCountUpdate: expected %d / buffered %lu", pagesEndpoint, mCFContext->fragmentData[fragment.index].pageDigits[iSector][j].size());
        }
      }
    } else {
      clusterer.mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{digits, j, 0};
      digits += mCFContext->fragmentData[fragment.index].nDigits[iSector][j];
      pages += mCFContext->fragmentData[fragment.index].nPages[iSector][j];
    }
  }
  if (doGPU) {
    pages = o - processors()->tpcClusterer[iSector].mPzsOffsets;
  }
  if (!doGPU && GetProcessingSettings().debugLevel >= 4 && mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
    TPCClusterizerEnsureZSOffsets(iSector, fragment);
  }
  return {digits, pages};
}

void GPUChainTracking::TPCClusterizerEnsureZSOffsets(uint32_t iSector, const CfFragment& fragment)
{
  GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
  uint32_t nAdcs = 0;
  for (uint16_t endpoint = 0; endpoint < GPUTrackingInOutZS::NENDPOINTS; endpoint++) {
    const auto& data = mCFContext->fragmentData[fragment.index];
    uint32_t pagesEndpoint = 0;
    const uint32_t nAdcsExpected = data.nDigits[iSector][endpoint];
    const uint32_t nPagesExpected = data.nPages[iSector][endpoint];

    uint32_t nAdcDecoded = 0;
    const auto& zs = mIOPtrs.tpcZS->sector[iSector];
    for (uint32_t i = data.minMaxCN[iSector][endpoint].zsPtrFirst; i < data.minMaxCN[iSector][endpoint].zsPtrLast; i++) {
      const uint32_t pageFirst = (i == data.minMaxCN[iSector][endpoint].zsPtrFirst) ? data.minMaxCN[iSector][endpoint].zsPageFirst : 0;
      const uint32_t pageLast = (i + 1 == data.minMaxCN[iSector][endpoint].zsPtrLast) ? data.minMaxCN[iSector][endpoint].zsPageLast : zs.nZSPtr[endpoint][i];
      for (uint32_t j = pageFirst; j < pageLast; j++) {
        const uint8_t* page = static_cast<const uint8_t*>(zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const header::RAWDataHeader* rawDataHeader = reinterpret_cast<const header::RAWDataHeader*>(page);
        const TPCZSHDRV2* decHdr = reinterpret_cast<const TPCZSHDRV2*>(page + raw::RDHUtils::getMemorySize(*rawDataHeader) - sizeof(TPCZSHDRV2));
        const uint16_t nSamplesInPage = decHdr->nADCsamples;

        nAdcDecoded += nSamplesInPage;
        pagesEndpoint++;
      }
    }

    if (pagesEndpoint != nPagesExpected) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC raw page count mismatch: expected %d / buffered %u", iSector, endpoint, fragment.index, pagesEndpoint, nPagesExpected);
    }

    if (nAdcDecoded != nAdcsExpected) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC ADC count mismatch: expected %u, buffered %u", iSector, endpoint, fragment.index, nAdcsExpected, nAdcDecoded);
    }

    if (nAdcs != clusterer.mPzsOffsets[endpoint].offset) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC ADC offset mismatch: expected %u, buffered %u", iSector, endpoint, fragment.index, nAdcs, clusterer.mPzsOffsets[endpoint].offset);
    }

    nAdcs += nAdcsExpected;
  }
}

namespace
{
struct TPCCFDecodeScanTmp {
  int32_t zsPtrFirst, zsPageFirst, zsPtrLast, zsPageLast, hasData, pageCounter;
};
} // namespace

std::pair<uint32_t, uint32_t> GPUChainTracking::TPCClusterizerDecodeZSCount(uint32_t iSector, const CfFragment& fragment)
{
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  uint32_t nDigits = 0;
  uint32_t nPages = 0;
  uint32_t endpointAdcSamples[GPUTrackingInOutZS::NENDPOINTS];
  memset(endpointAdcSamples, 0, sizeof(endpointAdcSamples));
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  int32_t firstHBF = (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasTfStartOrbit) ? mIOPtrs.settingsTF->tfStartOrbit : ((mIOPtrs.tpcZS->sector[iSector].count[0] && mIOPtrs.tpcZS->sector[iSector].nZSPtr[0][0]) ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->sector[iSector].zsPtr[0][0]) : 0);

  for (uint16_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
#ifndef GPUCA_NO_VC
    if (GetProcessingSettings().prefetchTPCpageScan >= 3 && j < GPUTrackingInOutZS::NENDPOINTS - 1) {
      for (uint32_t k = 0; k < mIOPtrs.tpcZS->sector[iSector].count[j + 1]; k++) {
        for (uint32_t l = 0; l < mIOPtrs.tpcZS->sector[iSector].nZSPtr[j + 1][k]; l++) {
          Vc::Common::prefetchMid(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchMid(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
        }
      }
    }
#endif

    std::vector<std::pair<CfFragment, TPCCFDecodeScanTmp>> fragments;
    fragments.reserve(mCFContext->nFragments);
    fragments.emplace_back(std::pair<CfFragment, TPCCFDecodeScanTmp>{fragment, {0, 0, 0, 0, 0, -1}});
    for (uint32_t i = 1; i < mCFContext->nFragments; i++) {
      fragments.emplace_back(std::pair<CfFragment, TPCCFDecodeScanTmp>{fragments.back().first.next(), {0, 0, 0, 0, 0, -1}});
    }
    std::vector<bool> fragmentExtends(mCFContext->nFragments, false);

    uint32_t firstPossibleFragment = 0;
    uint32_t pageCounter = 0;
    uint32_t emptyPages = 0;
    for (uint32_t k = 0; k < mIOPtrs.tpcZS->sector[iSector].count[j]; k++) {
      if (GetProcessingSettings().tpcSingleSector != -1 && GetProcessingSettings().tpcSingleSector != (int32_t)iSector) {
        break;
      }
      nPages += mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k];
      for (uint32_t l = 0; l < mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k]; l++) {
#ifndef GPUCA_NO_VC
        if (GetProcessingSettings().prefetchTPCpageScan >= 2 && l + 1 < mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k]) {
          Vc::Common::prefetchForOneRead(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchForOneRead(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
        }
#endif
        const uint8_t* const page = ((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)page;
        if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
          emptyPages++;
          continue;
        }
        pageCounter++;
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(rdh_utils::getLink(o2::raw::RDHUtils::getFEEID(*rdh)) == rdh_utils::DLBZSLinkID ? (page + o2::raw::RDHUtils::getMemorySize(*rdh) - sizeof(TPCZSHDRV2)) : (page + sizeof(o2::header::RAWDataHeader)));
        if (mCFContext->zsVersion == -1) {
          mCFContext->zsVersion = hdr->version;
          if (GetProcessingSettings().param.tpcTriggerHandling && mCFContext->zsVersion < ZSVersion::ZSVersionDenseLinkBased) { // TODO: Move tpcTriggerHandling to recoSteps bitmask
            static bool errorShown = false;
            if (errorShown == false) {
              GPUAlarm("Trigger handling only possible with TPC Dense Link Based data, received version %d, disabling", mCFContext->zsVersion);
            }
            errorShown = true;
          }
        } else if (mCFContext->zsVersion != (int32_t)hdr->version) {
          GPUError("Received TPC ZS 8kb page of mixed versions, expected %d, received %d (linkid %d, feeCRU %d, feeEndpoint %d, feelinkid %d)", mCFContext->zsVersion, (int32_t)hdr->version, (int32_t)o2::raw::RDHUtils::getLinkID(*rdh), (int32_t)rdh_utils::getCRU(*rdh), (int32_t)rdh_utils::getEndPoint(*rdh), (int32_t)rdh_utils::getLink(*rdh));
          constexpr size_t bufferSize = 3 * std::max(sizeof(*rdh), sizeof(*hdr)) + 1;
          char dumpBuffer[bufferSize];
          for (size_t i = 0; i < sizeof(*rdh); i++) {
            // "%02X " guaranteed to be 3 chars + ending 0.
            snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int32_t)((uint8_t*)rdh)[i]);
          }
          GPUAlarm("RDH of page: %s", dumpBuffer);
          for (size_t i = 0; i < sizeof(*hdr); i++) {
            // "%02X " guaranteed to be 3 chars + ending 0.
            snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int32_t)((uint8_t*)hdr)[i]);
          }
          GPUAlarm("Metainfo of page: %s", dumpBuffer);
          if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
            mCFContext->abandonTimeframe = true;
            return {0, 0};
          } else {
            GPUFatal("Cannot process with invalid TPC ZS data, exiting");
          }
        }
        if (GetProcessingSettings().param.tpcTriggerHandling) {
          const TPCZSHDRV2* const hdr2 = (const TPCZSHDRV2*)hdr;
          if (hdr2->flags & TPCZSHDRV2::ZSFlags::TriggerWordPresent) {
            const char* triggerWord = (const char*)hdr - TPCZSHDRV2::TRIGGER_WORD_SIZE;
            o2::tpc::TriggerInfoDLBZS tmp;
            memcpy((void*)&tmp.triggerWord, triggerWord, TPCZSHDRV2::TRIGGER_WORD_SIZE);
            tmp.orbit = o2::raw::RDHUtils::getHeartBeatOrbit(*rdh);
            if (tmp.triggerWord.isValid(0)) {
              mTriggerBuffer->triggers.emplace(tmp);
            }
          }
        }
        nDigits += hdr->nADCsamples;
        endpointAdcSamples[j] += hdr->nADCsamples;
        uint32_t timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
        uint32_t maxTimeBin = timeBin + hdr->nTimeBinSpan;
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          const TPCZSHDRV2* const hdr2 = (const TPCZSHDRV2*)hdr;
          if (hdr2->flags & TPCZSHDRV2::ZSFlags::nTimeBinSpanBit8) {
            maxTimeBin += 256;
          }
        }
        if (maxTimeBin > mCFContext->tpcMaxTimeBin) {
          mCFContext->tpcMaxTimeBin = maxTimeBin;
        }
        bool extendsInNextPage = false;
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          if (l + 1 < mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k] && o2::raw::RDHUtils::getMemorySize(*rdh) == TPCZSHDR::TPC_ZS_PAGE_SIZE) {
            const o2::header::RAWDataHeader* nextrdh = (const o2::header::RAWDataHeader*)(page + TPCZSHDR::TPC_ZS_PAGE_SIZE);
            extendsInNextPage = o2::raw::RDHUtils::getHeartBeatOrbit(*nextrdh) == o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) && o2::raw::RDHUtils::getMemorySize(*nextrdh) > sizeof(o2::header::RAWDataHeader);
          }
        }
        while (firstPossibleFragment && (uint32_t)fragments[firstPossibleFragment - 1].first.last() > timeBin) {
          firstPossibleFragment--;
        }
        auto handleExtends = [&](uint32_t ff) {
          if (fragmentExtends[ff]) {
            if (doGPU) {
              // Only add extended page on GPU. On CPU the pages are in consecutive memory anyway.
              // Not adding the page prevents an issue where a page is decoded twice on CPU, when only the extend should be decoded.
              fragments[ff].second.zsPageLast++;
              mCFContext->fragmentData[ff].nPages[iSector][j]++;
              mCFContext->fragmentData[ff].pageDigits[iSector][j].emplace_back(0);
            }
            fragmentExtends[ff] = false;
          }
        };
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          for (uint32_t ff = 0; ff < firstPossibleFragment; ff++) {
            handleExtends(ff);
          }
        }
        for (uint32_t f = firstPossibleFragment; f < mCFContext->nFragments; f++) {
          if (timeBin < (uint32_t)fragments[f].first.last() && (uint32_t)fragments[f].first.first() <= maxTimeBin) {
            if (!fragments[f].second.hasData) {
              fragments[f].second.hasData = 1;
              fragments[f].second.zsPtrFirst = k;
              fragments[f].second.zsPageFirst = l;
            } else {
              if (pageCounter > (uint32_t)fragments[f].second.pageCounter + 1) {
                mCFContext->fragmentData[f].nPages[iSector][j] += emptyPages + pageCounter - fragments[f].second.pageCounter - 1;
                for (uint32_t k2 = fragments[f].second.zsPtrLast - 1; k2 <= k; k2++) {
                  for (uint32_t l2 = ((int32_t)k2 == fragments[f].second.zsPtrLast - 1) ? fragments[f].second.zsPageLast : 0; l2 < (k2 < k ? mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k2] : l); l2++) {
                    if (doGPU) {
                      mCFContext->fragmentData[f].pageDigits[iSector][j].emplace_back(0);
                    } else {
                      // CPU cannot skip unneeded pages, so we must keep space to store the invalid dummy clusters
                      const uint8_t* const pageTmp = ((const uint8_t*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k2]) + l2 * TPCZSHDR::TPC_ZS_PAGE_SIZE;
                      const o2::header::RAWDataHeader* rdhTmp = (const o2::header::RAWDataHeader*)pageTmp;
                      if (o2::raw::RDHUtils::getMemorySize(*rdhTmp) != sizeof(o2::header::RAWDataHeader)) {
                        const TPCZSHDR* const hdrTmp = (const TPCZSHDR*)(rdh_utils::getLink(o2::raw::RDHUtils::getFEEID(*rdhTmp)) == rdh_utils::DLBZSLinkID ? (pageTmp + o2::raw::RDHUtils::getMemorySize(*rdhTmp) - sizeof(TPCZSHDRV2)) : (pageTmp + sizeof(o2::header::RAWDataHeader)));
                        mCFContext->fragmentData[f].nDigits[iSector][j] += hdrTmp->nADCsamples;
                      }
                    }
                  }
                }
              } else if (emptyPages) {
                mCFContext->fragmentData[f].nPages[iSector][j] += emptyPages;
                if (doGPU) {
                  for (uint32_t m = 0; m < emptyPages; m++) {
                    mCFContext->fragmentData[f].pageDigits[iSector][j].emplace_back(0);
                  }
                }
              }
            }
            fragments[f].second.zsPtrLast = k + 1;
            fragments[f].second.zsPageLast = l + 1;
            fragments[f].second.pageCounter = pageCounter;
            mCFContext->fragmentData[f].nPages[iSector][j]++;
            mCFContext->fragmentData[f].nDigits[iSector][j] += hdr->nADCsamples;
            if (doGPU) {
              mCFContext->fragmentData[f].pageDigits[iSector][j].emplace_back(hdr->nADCsamples);
            }
            fragmentExtends[f] = extendsInNextPage;
          } else {
            handleExtends(f);
            if (timeBin < (uint32_t)fragments[f].first.last()) {
              if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
                for (uint32_t ff = f + 1; ff < mCFContext->nFragments; ff++) {
                  handleExtends(ff);
                }
              }
              break;
            } else {
              firstPossibleFragment = f + 1;
            }
          }
        }
        emptyPages = 0;
      }
    }
    for (uint32_t f = 0; f < mCFContext->nFragments; f++) {
      mCFContext->fragmentData[f].minMaxCN[iSector][j].zsPtrLast = fragments[f].second.zsPtrLast;
      mCFContext->fragmentData[f].minMaxCN[iSector][j].zsPtrFirst = fragments[f].second.zsPtrFirst;
      mCFContext->fragmentData[f].minMaxCN[iSector][j].zsPageLast = fragments[f].second.zsPageLast;
      mCFContext->fragmentData[f].minMaxCN[iSector][j].zsPageFirst = fragments[f].second.zsPageFirst;
    }
  }
  mCFContext->nPagesTotal += nPages;
  mCFContext->nPagesSector[iSector] = nPages;

  mCFContext->nDigitsEndpointMax[iSector] = 0;
  for (uint32_t i = 0; i < GPUTrackingInOutZS::NENDPOINTS; i++) {
    if (endpointAdcSamples[i] > mCFContext->nDigitsEndpointMax[iSector]) {
      mCFContext->nDigitsEndpointMax[iSector] = endpointAdcSamples[i];
    }
  }
  uint32_t nDigitsFragmentMax = 0;
  for (uint32_t i = 0; i < mCFContext->nFragments; i++) {
    uint32_t pagesInFragment = 0;
    uint32_t digitsInFragment = 0;
    for (uint16_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      pagesInFragment += mCFContext->fragmentData[i].nPages[iSector][j];
      digitsInFragment += mCFContext->fragmentData[i].nDigits[iSector][j];
    }
    mCFContext->nPagesFragmentMax = std::max(mCFContext->nPagesFragmentMax, pagesInFragment);
    nDigitsFragmentMax = std::max(nDigitsFragmentMax, digitsInFragment);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();
  return {nDigits, nDigitsFragmentMax};
}

void GPUChainTracking::RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int32_t stage, bool doGPU, int32_t lane)
{
  auto& in = stage ? clustererShadow.mPpeakPositions : clustererShadow.mPpositions;
  auto& out = stage ? clustererShadow.mPfilteredPeakPositions : clustererShadow.mPpeakPositions;
  if (doGPU) {
    const uint32_t iSector = clusterer.mISector;
    auto& count = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;

    std::vector<size_t> counts;

    uint32_t nSteps = clusterer.getNSteps(count);
    if (nSteps > clusterer.mNBufs) {
      GPUError("Clusterer buffers exceeded (%u > %u)", nSteps, (int32_t)clusterer.mNBufs);
      exit(1);
    }

    size_t tmpCount = count;
    if (nSteps > 1) {
      for (uint32_t i = 1; i < nSteps; i++) {
        counts.push_back(tmpCount);
        if (i == 1) {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanStart>({GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSector}}, i, stage);
        } else {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanUp>({GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSector}}, i, tmpCount);
        }
        tmpCount = (tmpCount + clusterer.mScanWorkGroupSize - 1) / clusterer.mScanWorkGroupSize;
      }

      runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanTop>({GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSector}}, nSteps, tmpCount);

      for (uint32_t i = nSteps - 1; i > 1; i--) {
        tmpCount = counts[i - 1];
        runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanDown>({GetGrid(tmpCount - clusterer.mScanWorkGroupSize, clusterer.mScanWorkGroupSize, lane), {iSector}}, i, clusterer.mScanWorkGroupSize, tmpCount);
      }
    }

    runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::compactDigits>({GetGrid(count, clusterer.mScanWorkGroupSize, lane), {iSector}}, 1, stage, in, out);
  } else {
    auto& nOut = stage ? clusterer.mPmemory->counters.nClusters : clusterer.mPmemory->counters.nPeaks;
    auto& nIn = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
    size_t count = 0;
    for (size_t i = 0; i < nIn; i++) {
      if (clusterer.mPisPeak[i]) {
        out[count++] = in[i];
      }
    }
    nOut = count;
  }
}

std::pair<uint32_t, uint32_t> GPUChainTracking::RunTPCClusterizer_transferZS(int32_t iSector, const CfFragment& fragment, int32_t lane)
{
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  if (mCFContext->abandonTimeframe) {
    return {0, 0};
  }
  const auto& retVal = TPCClusterizerDecodeZSCountUpdate(iSector, fragment);
  if (doGPU) {
    GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
    GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;
    uint32_t nPagesSector = 0;
    for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      uint32_t nPages = 0;
      mInputsHost->mPzsMeta->sector[iSector].zsPtr[j] = &mInputsShadow->mPzsPtrs[iSector * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsPtrs[iSector * GPUTrackingInOutZS::NENDPOINTS + j] = clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      for (uint32_t k = clusterer.mMinMaxCN[j].zsPtrFirst; k < clusterer.mMinMaxCN[j].zsPtrLast; k++) {
        const uint32_t min = (k == clusterer.mMinMaxCN[j].zsPtrFirst) ? clusterer.mMinMaxCN[j].zsPageFirst : 0;
        const uint32_t max = (k + 1 == clusterer.mMinMaxCN[j].zsPtrLast) ? clusterer.mMinMaxCN[j].zsPageLast : mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k];
        if (max > min) {
          char* src = (char*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k] + min * TPCZSHDR::TPC_ZS_PAGE_SIZE;
          char* ptrLast = (char*)mIOPtrs.tpcZS->sector[iSector].zsPtr[j][k] + (max - 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
          size_t size = (ptrLast - src) + o2::raw::RDHUtils::getMemorySize(*(const o2::header::RAWDataHeader*)ptrLast);
          GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE, src, size, lane, true);
        }
        nPages += max - min;
      }
      mInputsHost->mPzsMeta->sector[iSector].nZSPtr[j] = &mInputsShadow->mPzsSizes[iSector * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsSizes[iSector * GPUTrackingInOutZS::NENDPOINTS + j] = nPages;
      mInputsHost->mPzsMeta->sector[iSector].count[j] = 1;
      nPagesSector += nPages;
    }
    GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzsOffsets, clusterer.mPzsOffsets, clusterer.mNMaxPages * sizeof(*clusterer.mPzsOffsets), lane, true);
  }
  return retVal;
}

int32_t GPUChainTracking::RunTPCClusterizer_prepare(bool restorePointers)
{
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  if (restorePointers) {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      processors()->tpcClusterer[iSector].mPzsOffsets = mCFContext->ptrSave[iSector].zsOffsetHost;
      processorsShadow()->tpcClusterer[iSector].mPzsOffsets = mCFContext->ptrSave[iSector].zsOffsetDevice;
      processorsShadow()->tpcClusterer[iSector].mPzs = mCFContext->ptrSave[iSector].zsDevice;
    }
    processorsShadow()->ioPtrs.clustersNative = mCFContext->ptrClusterNativeSave;
    return 0;
  }
  const auto& threadContext = GetThreadContext();
  mRec->MemoryScalers()->nTPCdigits = 0;
  if (mCFContext == nullptr) {
    mCFContext.reset(new GPUTPCCFChainContext);
  }
  const int16_t maxFragmentLen = GetProcessingSettings().overrideClusterizerFragmentLen;
  const uint32_t maxAllowedTimebin = param().par.continuousTracking ? std::max<int32_t>(param().continuousMaxTimeBin, maxFragmentLen) : TPC_MAX_TIME_BIN_TRIGGERED;
  mCFContext->tpcMaxTimeBin = maxAllowedTimebin;
  const CfFragment fragmentMax{(tpccf::TPCTime)mCFContext->tpcMaxTimeBin + 1, maxFragmentLen};
  mCFContext->prepare(mIOPtrs.tpcZS, fragmentMax);
  if (GetProcessingSettings().param.tpcTriggerHandling) {
    mTriggerBuffer->triggers.clear();
  }
  if (mIOPtrs.tpcZS) {
    uint32_t nDigitsFragmentMax[NSECTORS];
    mCFContext->zsVersion = -1;
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      if (mIOPtrs.tpcZS->sector[iSector].count[0]) {
        const void* rdh = mIOPtrs.tpcZS->sector[iSector].zsPtr[0][0];
        if (rdh && o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeaderV6>() > o2::raw::RDHUtils::getVersion(rdh)) {
          GPUError("Data has invalid RDH version %d, %d required\n", o2::raw::RDHUtils::getVersion(rdh), o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
          return 1;
        }
      }
#ifndef GPUCA_NO_VC
      if (GetProcessingSettings().prefetchTPCpageScan >= 1 && iSector < NSECTORS - 1) {
        for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (uint32_t k = 0; k < mIOPtrs.tpcZS->sector[iSector].count[j]; k++) {
            for (uint32_t l = 0; l < mIOPtrs.tpcZS->sector[iSector].nZSPtr[j][k]; l++) {
              Vc::Common::prefetchFar(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
              Vc::Common::prefetchFar(((const uint8_t*)mIOPtrs.tpcZS->sector[iSector + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
            }
          }
        }
      }
#endif
      const auto& x = TPCClusterizerDecodeZSCount(iSector, fragmentMax);
      nDigitsFragmentMax[iSector] = x.first;
      processors()->tpcClusterer[iSector].mPmemory->counters.nDigits = x.first;
      mRec->MemoryScalers()->nTPCdigits += x.first;
    }
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      uint32_t nDigitsBase = nDigitsFragmentMax[iSector];
      uint32_t threshold = 40000000;
      uint32_t nDigitsScaled = nDigitsBase > threshold ? nDigitsBase : std::min((threshold + nDigitsBase) / 2, 2 * nDigitsBase);
      processors()->tpcClusterer[iSector].SetNMaxDigits(processors()->tpcClusterer[iSector].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled, mCFContext->nDigitsEndpointMax[iSector]);
      if (doGPU) {
        processorsShadow()->tpcClusterer[iSector].SetNMaxDigits(processors()->tpcClusterer[iSector].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled, mCFContext->nDigitsEndpointMax[iSector]);
      }
      if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSector].mZSOffsetId, mRec);
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSector].mZSId, mRec);
      } else {
        AllocateRegisteredMemory(processors()->tpcClusterer[iSector].mZSOffsetId);
        AllocateRegisteredMemory(processors()->tpcClusterer[iSector].mZSId);
      }
    }
  } else {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      uint32_t nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSector];
      mRec->MemoryScalers()->nTPCdigits += nDigits;
      processors()->tpcClusterer[iSector].SetNMaxDigits(nDigits, mCFContext->nPagesFragmentMax, nDigits, 0);
    }
  }

  if (mIOPtrs.tpcZS) {
    GPUInfo("Event has %u 8kb TPC ZS pages (version %d), %ld digits", mCFContext->nPagesTotal, mCFContext->zsVersion, (int64_t)mRec->MemoryScalers()->nTPCdigits);
  } else {
    GPUInfo("Event has %ld TPC Digits", (int64_t)mRec->MemoryScalers()->nTPCdigits);
  }

  if (mCFContext->tpcMaxTimeBin > maxAllowedTimebin) {
    GPUError("Input data has invalid time bin %u > %d", mCFContext->tpcMaxTimeBin, maxAllowedTimebin);
    if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
      mCFContext->abandonTimeframe = true;
      mCFContext->tpcMaxTimeBin = maxAllowedTimebin;
    } else {
      return 1;
    }
  }

  mCFContext->fragmentFirst = CfFragment{std::max<int32_t>(mCFContext->tpcMaxTimeBin + 1, maxFragmentLen), maxFragmentLen};
  for (int32_t iSector = 0; iSector < GetProcessingSettings().nTPCClustererLanes && iSector < NSECTORS; iSector++) {
    if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSector] && mCFContext->zsVersion != -1) {
      mCFContext->nextPos[iSector] = RunTPCClusterizer_transferZS(iSector, mCFContext->fragmentFirst, GetProcessingSettings().nTPCClustererLanes + iSector);
    }
  }

  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      mCFContext->ptrSave[iSector].zsOffsetHost = processors()->tpcClusterer[iSector].mPzsOffsets;
      mCFContext->ptrSave[iSector].zsOffsetDevice = processorsShadow()->tpcClusterer[iSector].mPzsOffsets;
      mCFContext->ptrSave[iSector].zsDevice = processorsShadow()->tpcClusterer[iSector].mPzs;
    }
  }
  return 0;
}
#endif

int32_t GPUChainTracking::RunTPCClusterizer(bool synchronizeOutput)
{
  if (param().rec.fwdTPCDigitsAsClusters) {
    return ForwardTPCDigits();
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  int32_t tpcTimeBinCut = mUpdateNewCalibObjects && mNewCalibValues->newTPCTimeBinCut ? mNewCalibValues->tpcTimeBinCut : param().tpcCutTimeBin;
  mRec->PushNonPersistentMemory(qStr2Tag("TPCCLUST"));
  const auto& threadContext = GetThreadContext();
  const bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  if (RunTPCClusterizer_prepare(mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer)) {
    return 1;
  }
  if (GetProcessingSettings().autoAdjustHostThreads && !doGPU) {
    mRec->SetNActiveThreads(mRec->MemoryScalers()->nTPCdigits / 6000);
  }

  mRec->MemoryScalers()->nTPCHits = mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
  float tpcHitLowOccupancyScalingFactor = 1.f;
  if (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasNHBFPerTF) {
    uint32_t nHitsBase = mRec->MemoryScalers()->nTPCHits;
    uint32_t threshold = 30000000 / 256 * mIOPtrs.settingsTF->nHBFPerTF;
    if (mIOPtrs.settingsTF->nHBFPerTF < 64) {
      threshold *= 2;
    }
    mRec->MemoryScalers()->nTPCHits = std::max<uint32_t>(nHitsBase, std::min<uint32_t>(threshold, nHitsBase * 3.5f)); // Increase the buffer size for low occupancy data to compensate for noisy pads creating exceiive clusters
    if (nHitsBase < threshold) {
      float maxFactor = mRec->MemoryScalers()->nTPCHits < threshold * 2 / 3 ? 3 : (mRec->MemoryScalers()->nTPCHits < threshold ? 2.25f : 1.75f);
      mRec->MemoryScalers()->temporaryFactor *= std::min(maxFactor, (float)threshold / nHitsBase);
      tpcHitLowOccupancyScalingFactor = std::min(3.5f, (float)threshold / nHitsBase);
    }
  }
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    processors()->tpcClusterer[iSector].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    SetupGPUProcessor(&processors()->tpcClusterer[iSector], true); // Now we allocate
  }
  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    RunTPCClusterizer_prepare(true); // Restore some pointers, allocated by the other pipeline, and set to 0 by SetupGPUProcessor (since not allocated in this pipeline)
  }

#ifdef GPUCA_HAS_ONNX
  if (GetProcessingSettings().nn.applyNNclusterizer) {
    uint32_t maxClusters = -1;
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      maxClusters = std::max(maxClusters, processors()->tpcClusterer[iSector].mNMaxClusters);
    }
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      GPUTPCNNClusterizer& clustererNN = processors()->tpcNNClusterer[iSector];
      const GPUSettingsProcessingNNclusterizer& nn_settings = GetProcessingSettings().nn;
      clustererNN.nnClusterizerUseCfRegression = nn_settings.nnClusterizerUseCfRegression;
      clustererNN.nnClusterizerSizeInputRow = nn_settings.nnClusterizerSizeInputRow;
      clustererNN.nnClusterizerSizeInputPad = nn_settings.nnClusterizerSizeInputPad;
      clustererNN.nnClusterizerSizeInputTime = nn_settings.nnClusterizerSizeInputTime;
      clustererNN.nnClusterizerAddIndexData = nn_settings.nnClusterizerAddIndexData;
      clustererNN.nnClusterizerElementSize = ((2 * nn_settings.nnClusterizerSizeInputRow + 1) * (2 * nn_settings.nnClusterizerSizeInputPad + 1) * (2 * nn_settings.nnClusterizerSizeInputTime + 1)) + (nn_settings.nnClusterizerAddIndexData ? 3 : 0);
      clustererNN.nnClusterizerBatchedMode = nn_settings.nnClusterizerBatchedMode;
      clustererNN.nnClusterizerBoundaryFillValue = nn_settings.nnClusterizerBoundaryFillValue;
      clustererNN.nnClusterizerTotalClusters = maxClusters;
      clustererNN.nnClassThreshold = nn_settings.nnClassThreshold;
      clustererNN.nnSigmoidTrafoClassThreshold = nn_settings.nnSigmoidTrafoClassThreshold;
      if (clustererNN.nnSigmoidTrafoClassThreshold) {
        clustererNN.nnClassThreshold = (float)std::log(clustererNN.nnClassThreshold / (1.f - clustererNN.nnClassThreshold));
      }
      if (nn_settings.nnClusterizerVerbosity < 0) {
        clustererNN.nnClusterizerVerbosity = nn_settings.nnInferenceVerbosity;
      } else {
        clustererNN.nnClusterizerVerbosity = nn_settings.nnClusterizerVerbosity;
      }
      clustererNN.nnClusterizerDtype = nn_settings.nnInferenceDtype.find("32") != std::string::npos;
      GPUTPCNNClusterizerHost nnApplication(nn_settings, clustererNN);
      AllocateRegisteredMemory(clustererNN.mMemoryId);
    }
  }
#endif

  if (doGPU && mIOPtrs.tpcZS) {
    processorsShadow()->ioPtrs.tpcZS = mInputsShadow->mPzsMeta;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), mRec->NStreams() - 1);
  }
  if (doGPU) {
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSECTORS, mRec->NStreams() - 1, &mEvents->init);
  }

  size_t nClsTotal = 0;
  ClusterNativeAccess* tmpNativeAccess = mClusterNativeAccess.get();
  ClusterNative* tmpNativeClusters = nullptr;
  std::unique_ptr<ClusterNative[]> tmpNativeClusterBuffer;

  // setup MC Labels
  bool propagateMCLabels = GetProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  bool buildNativeGPU = doGPU && NeedTPCClustersOnGPU();
  bool buildNativeHost = (mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters) || GetProcessingSettings().deterministicGPUReconstruction; // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output

  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits * tpcHitLowOccupancyScalingFactor;
  if (buildNativeGPU) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
  }
  if (buildNativeHost && !(buildNativeGPU && GetProcessingSettings().delayedOutput)) {
    if (mWaitForFinalInputs) {
      GPUFatal("Cannot use waitForFinalInput callback without delayed output");
    }
    if (!GetProcessingSettings().tpcApplyClusterFilterOnCPU) {
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
      tmpNativeClusters = mInputsHost->mPclusterNativeOutput;
    } else {
      tmpNativeClusterBuffer = std::make_unique<ClusterNative[]>(mInputsHost->mNClusterNative);
      tmpNativeClusters = tmpNativeClusterBuffer.get();
    }
  }

  GPUTPCLinearLabels mcLinearLabels;
  if (propagateMCLabels) {
    // No need to overallocate here, nTPCHits is anyway an upper bound used for the GPU cluster buffer, and we can always enlarge the buffer anyway
    mcLinearLabels.header.reserve(mRec->MemoryScalers()->nTPCHits / 2);
    mcLinearLabels.data.reserve(mRec->MemoryScalers()->nTPCHits);
  }

  int8_t transferRunning[NSECTORS] = {0};
  uint32_t outputQueueStart = mOutputQueue.size();

  auto notifyForeignChainFinished = [this]() {
    if (mPipelineNotifyCtx) {
      SynchronizeStream(OutputStream()); // Must finish before updating ioPtrs in (global) constant memory
      {
        std::lock_guard<std::mutex> lock(mPipelineNotifyCtx->mutex);
        mPipelineNotifyCtx->ready = true;
      }
      mPipelineNotifyCtx->cond.notify_one();
    }
  };
  bool synchronizeCalibUpdate = false;

  for (uint32_t iSectorBase = 0; iSectorBase < NSECTORS; iSectorBase += GetProcessingSettings().nTPCClustererLanes) {
    std::vector<bool> laneHasData(GetProcessingSettings().nTPCClustererLanes, false);
    static_assert(NSECTORS <= GPUCA_MAX_STREAMS, "Stream events must be able to hold all sectors");
    const int32_t maxLane = std::min<int32_t>(GetProcessingSettings().nTPCClustererLanes, NSECTORS - iSectorBase);
    for (CfFragment fragment = mCFContext->fragmentFirst; !fragment.isEnd(); fragment = fragment.next()) {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d) for sectors %d to %d", fragment.start, fragment.last(), iSectorBase, iSectorBase + GetProcessingSettings().nTPCClustererLanes - 1);
      }
      mRec->runParallelOuterLoop(doGPU, maxLane, [&](uint32_t lane) {
        if (doGPU && fragment.index != 0) {
          SynchronizeStream(lane); // Don't overwrite charge map from previous iteration until cluster computation is finished
        }

        uint32_t iSector = iSectorBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;
        clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;
        clusterer.mPmemory->fragment = fragment;

        if (mIOPtrs.tpcPackedDigits) {
          bool setDigitsOnGPU = doGPU && not mIOPtrs.tpcZS;
          bool setDigitsOnHost = (not doGPU && not mIOPtrs.tpcZS) || propagateMCLabels;
          auto* inDigits = mIOPtrs.tpcPackedDigits;
          size_t numDigits = inDigits->nTPCDigits[iSector];
          if (setDigitsOnGPU) {
            GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSector], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
          }
          if (setDigitsOnHost) {
            clusterer.mPdigits = const_cast<o2::tpc::Digit*>(inDigits->tpcDigits[iSector]); // TODO: Needs fixing (invalid const cast)
          }
          clusterer.mPmemory->counters.nDigits = numDigits;
        }

        if (mIOPtrs.tpcZS) {
          if (mCFContext->nPagesSector[iSector] && mCFContext->zsVersion != -1) {
            clusterer.mPmemory->counters.nPositions = mCFContext->nextPos[iSector].first;
            clusterer.mPmemory->counters.nPagesSubsector = mCFContext->nextPos[iSector].second;
          } else {
            clusterer.mPmemory->counters.nPositions = clusterer.mPmemory->counters.nPagesSubsector = 0;
          }
        }
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

        using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
        using PeakMapType = decltype(*clustererShadow.mPpeakMap);
        runKernel<GPUMemClean16>({GetGridAutoStep(lane, RecoStep::TPCClusterFinding)}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items(GetProcessingSettings().overrideClusterizerFragmentLen) * sizeof(ChargeMapType)); // TODO: Not working in OpenCL2!!!
        runKernel<GPUMemClean16>({GetGridAutoStep(lane, RecoStep::TPCClusterFinding)}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items(GetProcessingSettings().overrideClusterizerFragmentLen) * sizeof(PeakMapType));
        if (fragment.index == 0) {
          runKernel<GPUMemClean16>({GetGridAutoStep(lane, RecoStep::TPCClusterFinding)}, clustererShadow.mPpadIsNoisy, TPC_PADS_IN_SECTOR * sizeof(*clustererShadow.mPpadIsNoisy));
        }
        DoDebugAndDump(RecoStep::TPCClusterFinding, 262144, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Zeroed Charges");

        if (doGPU) {
          if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSector] && mCFContext->zsVersion != -1) {
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
            SynchronizeStream(GetProcessingSettings().nTPCClustererLanes + lane);
          }
          SynchronizeStream(mRec->NStreams() - 1); // Wait for copying to constant memory
        }

        if (mIOPtrs.tpcZS && (mCFContext->abandonTimeframe || !mCFContext->nPagesSector[iSector] || mCFContext->zsVersion == -1)) {
          clusterer.mPmemory->counters.nPositions = 0;
          return;
        }
        if (!mIOPtrs.tpcZS && mIOPtrs.tpcPackedDigits->nTPCDigits[iSector] == 0) {
          clusterer.mPmemory->counters.nPositions = 0;
          return;
        }

        if (propagateMCLabels && fragment.index == 0) {
          clusterer.PrepareMC();
          clusterer.mPinputLabels = digitsMC->v[iSector];
          if (clusterer.mPinputLabels == nullptr) {
            GPUFatal("MC label container missing, sector %d", iSector);
          }
          if (clusterer.mPinputLabels->getIndexedSize() != mIOPtrs.tpcPackedDigits->nTPCDigits[iSector]) {
            GPUFatal("MC label container has incorrect number of entries: %d expected, has %d\n", (int32_t)mIOPtrs.tpcPackedDigits->nTPCDigits[iSector], (int32_t)clusterer.mPinputLabels->getIndexedSize());
          }
        }

        if (GetProcessingSettings().tpcSingleSector == -1 || GetProcessingSettings().tpcSingleSector == (int32_t)iSector) {
          if (not mIOPtrs.tpcZS) {
            runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>({GetGrid(1, lane), {iSector}}, mIOPtrs.tpcZS == nullptr);
            TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
          } else if (propagateMCLabels) {
            runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>({GetGrid(1, lane, GPUReconstruction::krnlDeviceType::CPU), {iSector}}, mIOPtrs.tpcZS == nullptr);
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
          }
        }

        if (mIOPtrs.tpcZS) {
          int32_t firstHBF = (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasTfStartOrbit) ? mIOPtrs.settingsTF->tfStartOrbit : ((mIOPtrs.tpcZS->sector[iSector].count[0] && mIOPtrs.tpcZS->sector[iSector].nZSPtr[0][0]) ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->sector[iSector].zsPtr[0][0]) : 0);
          uint32_t nBlocks = doGPU ? clusterer.mPmemory->counters.nPagesSubsector : GPUTrackingInOutZS::NENDPOINTS;

          (void)tpcTimeBinCut; // TODO: To be used in decoding kernels
          switch (mCFContext->zsVersion) {
            default:
              GPUFatal("Data with invalid TPC ZS mode (%d) received", mCFContext->zsVersion);
              break;
            case ZSVersionRowBased10BitADC:
            case ZSVersionRowBased12BitADC:
              runKernel<GPUTPCCFDecodeZS>({GetGridBlk(nBlocks, lane), {iSector}}, firstHBF);
              break;
            case ZSVersionLinkBasedWithMeta:
              runKernel<GPUTPCCFDecodeZSLink>({GetGridBlk(nBlocks, lane), {iSector}}, firstHBF);
              break;
            case ZSVersionDenseLinkBased:
              runKernel<GPUTPCCFDecodeZSDenseLink>({GetGridBlk(nBlocks, lane), {iSector}}, firstHBF);
              break;
          }
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        } // clang-format off
      });
      mRec->runParallelOuterLoop(doGPU, maxLane, [&](uint32_t lane) {
        uint32_t iSector = iSectorBase + lane;
        if (doGPU) {
          SynchronizeStream(lane);
        }
        if (mIOPtrs.tpcZS) {
          CfFragment f = fragment.next();
          int32_t nextSector = iSector;
          if (f.isEnd()) {
            nextSector += GetProcessingSettings().nTPCClustererLanes;
            f = mCFContext->fragmentFirst;
          }
          if (nextSector < NSECTORS && mIOPtrs.tpcZS && mCFContext->nPagesSector[nextSector] && mCFContext->zsVersion != -1 && !mCFContext->abandonTimeframe) {
            mCFContext->nextPos[nextSector] = RunTPCClusterizer_transferZS(nextSector, f, GetProcessingSettings().nTPCClustererLanes + lane);
          }
        }
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;
        if (clusterer.mPmemory->counters.nPositions == 0) {
          return;
        }
        if (!mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>({GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSector}});
        }
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 1, clusterer, &GPUTPCClusterFinder::DumpDigits, *mDebugFile)) {
          clusterer.DumpChargeMap(*mDebugFile, "Charges");
        }

        if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>({GetGrid(clusterer.mPmemory->counters.nDigitsInFragment, lane, GPUReconstruction::krnlDeviceType::CPU), {iSector}});
        }

        bool checkForNoisyPads = (rec()->GetParam().rec.tpc.maxTimeBinAboveThresholdIn1000Bin > 0) || (rec()->GetParam().rec.tpc.maxConsecTimeBinAboveThreshold > 0);
        checkForNoisyPads &= (rec()->GetParam().rec.tpc.noisyPadsQuickCheck ? fragment.index == 0 : true);
        checkForNoisyPads &= !GetProcessingSettings().disableTPCNoisyPadFilter;

        if (checkForNoisyPads) {
          int32_t nBlocks = TPC_PADS_IN_SECTOR / GPUTPCCFCheckPadBaseline::PadsPerCacheline;

          runKernel<GPUTPCCFCheckPadBaseline>({GetGridBlk(nBlocks, lane), {iSector}});
        }

        runKernel<GPUTPCCFPeakFinder>({GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSector}});
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 2, clusterer, &GPUTPCClusterFinder::DumpPeaks, *mDebugFile)) {
          clusterer.DumpPeakMap(*mDebugFile, "Peaks");
        }

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 0, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 2, clusterer, &GPUTPCClusterFinder::DumpPeaksCompacted, *mDebugFile); // clang-format off
      });
      mRec->runParallelOuterLoop(doGPU, maxLane, [&](uint32_t lane) {
        uint32_t iSector = iSectorBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;
        if (doGPU) {
          SynchronizeStream(lane);
        }
        if (clusterer.mPmemory->counters.nPeaks == 0) {
          return;
        }
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::noiseSuppression>({GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSector}});
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::updatePeaks>({GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSector}});
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 3, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaks, *mDebugFile)) {
          clusterer.DumpPeakMap(*mDebugFile, "Suppressed Peaks");
        }

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 3, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaksCompacted, *mDebugFile); // clang-format off
      });
      mRec->runParallelOuterLoop(doGPU, maxLane, [&](uint32_t lane) {
        uint32_t iSector = iSectorBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;

        if (doGPU) {
          SynchronizeStream(lane);
        }

        if (fragment.index == 0) {
          deviceEvent* waitEvent = nullptr;
          if (transferRunning[lane] == 1) {
            waitEvent = &mEvents->stream[lane];
            transferRunning[lane] = 2;
          }
          runKernel<GPUMemClean16>({GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {nullptr, waitEvent}}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
        }

        if (clusterer.mPmemory->counters.nClusters == 0) {
          return;
        }

        if (GetProcessingSettings().nn.applyNNclusterizer) {
#ifdef GPUCA_HAS_ONNX
          GPUTPCNNClusterizer& clustererNN = processors()->tpcNNClusterer[iSector];
          const GPUSettingsProcessingNNclusterizer& nn_settings = GetProcessingSettings().nn;
          GPUTPCNNClusterizerHost nnApplication(nn_settings, clustererNN);

          if (clustererNN.nnClusterizerUseCfRegression || (int)(nn_settings.nnClusterizerApplyCfDeconvolution)) {
            runKernel<GPUTPCCFDeconvolution>({GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSector}});
            DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 4, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Split Charges");
          }

          float time_clusterizer = 0, time_fill = 0;
          for (int batch = 0; batch < std::ceil((float)clusterer.mPmemory->counters.nClusters / clustererNN.nnClusterizerBatchedMode); batch++) {
            uint batchStart = batch * clustererNN.nnClusterizerBatchedMode;
            size_t iSize = CAMath::Min((uint)clustererNN.nnClusterizerBatchedMode, (uint)(clusterer.mPmemory->counters.nClusters - batchStart));

            auto start0 = std::chrono::high_resolution_clock::now();
            runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::fillInputNN>({GetGrid(iSize, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, batchStart); // Filling the data

            auto stop0 = std::chrono::high_resolution_clock::now();
            auto start1 = std::chrono::high_resolution_clock::now();
            nnApplication.networkInference(nnApplication.model_class, clustererNN, iSize, clustererNN.modelProbabilities, clustererNN.nnClusterizerDtype);
            if (nnApplication.model_class.getNumOutputNodes()[0][1] == 1) {
              runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::determineClass1Labels>({GetGrid(iSize, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, batchStart); // Assigning class labels
            } else {
              runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::determineClass2Labels>({GetGrid(iSize, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, batchStart); // Assigning class labels
            }

            if (!clustererNN.nnClusterizerUseCfRegression) {
              nnApplication.networkInference(nnApplication.model_reg_1, clustererNN, iSize, clustererNN.outputDataReg1, clustererNN.nnClusterizerDtype);
              runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::publishClass1Regression>({GetGrid(iSize, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, batchStart); // Running the NN for regression class 1
              if (nnApplication.model_class.getNumOutputNodes()[0][1] > 1 && nnApplication.reg_model_paths.size() > 1) {
                nnApplication.networkInference(nnApplication.model_reg_2, clustererNN, iSize, clustererNN.outputDataReg2, clustererNN.nnClusterizerDtype);
                runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::publishClass2Regression>({GetGrid(iSize, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, batchStart); // Running the NN for regression class 2
              }
            }
            auto stop1 = std::chrono::high_resolution_clock::now();

            time_clusterizer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1).count() / 1e9;
            time_fill += std::chrono::duration_cast<std::chrono::nanoseconds>(stop0 - start0).count() / 1e9;
          }
          auto start1 = std::chrono::high_resolution_clock::now();
          if (clustererNN.nnClusterizerUseCfRegression) {
            runKernel<GPUTPCNNClusterizerKernels, GPUTPCNNClusterizerKernels::runCfClusterizer>({GetGrid(clusterer.mPmemory->counters.nClusters, lane), krnlRunRangeNone}, iSector, clustererNN.nnClusterizerDtype, 0, 0); // Running the CF regression kernel - no batching needed: batchStart = 0
          }
          auto stop1 = std::chrono::high_resolution_clock::now();
          time_clusterizer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1).count() / 1e9;
          if (clustererNN.nnClusterizerVerbosity < 3) {
            int acceptedClusters = 0;
            for (size_t i = 0; i < clusterer.mPmemory->counters.nClusters; ++i) {
              acceptedClusters += clustererNN.outputDataClass[i];
            }
            LOG(info) << "[NN CF] Apply NN (fragment " << fragment.index << ", lane: " << lane << ", sector: " << iSector << "): filling data " << time_fill << "s ; clusterizer: " << time_clusterizer << "s ; " << clusterer.mPmemory->counters.nClusters << " clusters, " << acceptedClusters << " accepted. --> " << clusterer.mPmemory->counters.nClusters / (time_fill + time_clusterizer) << " clusters/s";
          }
#else
          GPUFatal("Project not compiled with neural network clusterization. Aborting.");
#endif
        } else {
          runKernel<GPUTPCCFDeconvolution>({GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSector}});
          DoDebugAndDump(RecoStep::TPCClusterFinding, 262144 << 4, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Split Charges");
          runKernel<GPUTPCCFClusterizer>({GetGrid(clusterer.mPmemory->counters.nClusters, lane), {iSector}}, 0);
        }

        if (doGPU && propagateMCLabels) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mScratchId, lane);
          if (doGPU) {
            SynchronizeStream(lane);
          }
          runKernel<GPUTPCCFClusterizer>({GetGrid(clusterer.mPmemory->counters.nClusters, lane, GPUReconstruction::krnlDeviceType::CPU), {iSector}}, 1); // Computes MC labels
        }

        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Sector %02d Fragment %02d Lane %d: Found clusters: digits %u peaks %u clusters %u", iSector, fragment.index, lane, (int32_t)clusterer.mPmemory->counters.nPositions, (int32_t)clusterer.mPmemory->counters.nPeaks, (int32_t)clusterer.mPmemory->counters.nClusters);
        }

        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
        laneHasData[lane] = true;
        // Include clusters in default debug mask, exclude other debug output by default
        DoDebugAndDump(RecoStep::TPCClusterFinding, 131072, clusterer, &GPUTPCClusterFinder::DumpClusters, *mDebugFile); // clang-format off
      });
      mRec->SetNActiveThreadsOuterLoop(1);
    }

    size_t nClsFirst = nClsTotal;
    bool anyLaneHasData = false;
    for (int32_t lane = 0; lane < maxLane; lane++) {
      uint32_t iSector = iSectorBase + lane;
      std::fill(&tmpNativeAccess->nClusters[iSector][0], &tmpNativeAccess->nClusters[iSector][0] + MAXGLOBALPADROW, 0);
      if (doGPU) {
        SynchronizeStream(lane);
      }
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSector];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSector] : clusterer;

      if (laneHasData[lane]) {
        anyLaneHasData = true;
        if (buildNativeGPU && GetProcessingSettings().tpccfGatherKernel) {
          runKernel<GPUTPCCFGather>({GetGridBlk(GPUCA_ROW_COUNT, mRec->NStreams() - 1), {iSector}}, &mInputsShadow->mPclusterNativeBuffer[nClsTotal]);
        }
        for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
          if (nClsTotal + clusterer.mPclusterInRow[j] > mInputsHost->mNClusterNative) {
            clusterer.raiseError(GPUErrors::ERROR_CF_GLOBAL_CLUSTER_OVERFLOW, iSector * 1000 + j, nClsTotal + clusterer.mPclusterInRow[j], mInputsHost->mNClusterNative);
            continue;
          }
          if (buildNativeGPU) {
            if (!GetProcessingSettings().tpccfGatherKernel) {
              GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsShadow->mPclusterNativeBuffer[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, -2);
            }
          } else if (buildNativeHost) {
            GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&tmpNativeClusters[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, false);
          }
          tmpNativeAccess->nClusters[iSector][j] += clusterer.mPclusterInRow[j];
          nClsTotal += clusterer.mPclusterInRow[j];
        }
        if (transferRunning[lane]) {
          ReleaseEvent(mEvents->stream[lane], doGPU);
        }
        RecordMarker(&mEvents->stream[lane], mRec->NStreams() - 1);
        transferRunning[lane] = 1;
      }

      if (not propagateMCLabels || not laneHasData[lane]) {
        assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>({GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSector}});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>({GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSector}}, &mcLinearLabels);
      clusterer.clearMCMemory();
      assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
    }
    if (propagateMCLabels) {
      for (int32_t lane = 0; lane < maxLane; lane++) {
        processors()->tpcClusterer[iSectorBase + lane].clearMCMemory();
      }
    }
    if (buildNativeHost && buildNativeGPU && anyLaneHasData) {
      if (GetProcessingSettings().delayedOutput) {
        mOutputQueue.emplace_back(outputQueueEntry{(void*)((char*)&tmpNativeClusters[nClsFirst] - (char*)&tmpNativeClusters[0]), &mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(tmpNativeClusters[0]), RecoStep::TPCClusterFinding});
      } else {
        GPUMemCpy(RecoStep::TPCClusterFinding, (void*)&tmpNativeClusters[nClsFirst], (const void*)&mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(tmpNativeClusters[0]), mRec->NStreams() - 1, false);
      }
    }

    if (mWaitForFinalInputs && iSectorBase >= 21 && (int32_t)iSectorBase < 21 + GetProcessingSettings().nTPCClustererLanes) {
      notifyForeignChainFinished();
    }
    if (mWaitForFinalInputs && iSectorBase >= 30 && (int32_t)iSectorBase < 30 + GetProcessingSettings().nTPCClustererLanes) {
      mWaitForFinalInputs();
      synchronizeCalibUpdate = DoQueuedUpdates(0, false);
    }
  }
  for (int32_t i = 0; i < GetProcessingSettings().nTPCClustererLanes; i++) {
    if (transferRunning[i]) {
      ReleaseEvent(mEvents->stream[i], doGPU);
    }
  }

  if (GetProcessingSettings().param.tpcTriggerHandling) {
    GPUOutputControl* triggerOutput = mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTriggerWords)];
    if (triggerOutput && triggerOutput->allocator) {
      // GPUInfo("Storing %lu trigger words", mTriggerBuffer->triggers.size());
      auto* outputBuffer = (decltype(mTriggerBuffer->triggers)::value_type*)triggerOutput->allocator(mTriggerBuffer->triggers.size() * sizeof(decltype(mTriggerBuffer->triggers)::value_type));
      std::copy(mTriggerBuffer->triggers.begin(), mTriggerBuffer->triggers.end(), outputBuffer);
    }
    mTriggerBuffer->triggers.clear();
  }

  ClusterNativeAccess::ConstMCLabelContainerView* mcLabelsConstView = nullptr;
  if (propagateMCLabels) {
    // TODO: write to buffer directly
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;
    std::pair<ConstMCLabelContainer*, ConstMCLabelContainerView*> buffer;
    if (mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)] && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->useExternal()) {
      if (!mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->allocator) {
        throw std::runtime_error("Cluster MC Label buffer missing");
      }
      ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer* container = reinterpret_cast<ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer*>(mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->allocator(0));
      buffer = {&container->first, &container->second};
    } else {
      mIOMem.clusterNativeMCView = std::make_unique<ConstMCLabelContainerView>();
      mIOMem.clusterNativeMCBuffer = std::make_unique<ConstMCLabelContainer>();
      buffer.first = mIOMem.clusterNativeMCBuffer.get();
      buffer.second = mIOMem.clusterNativeMCView.get();
    }

    assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
    assert(propagateMCLabels ? mcLinearLabels.data.size() >= nClsTotal : true);

    mcLabels.setFrom(mcLinearLabels.header, mcLinearLabels.data);
    mcLabels.flatten_to(*buffer.first);
    *buffer.second = *buffer.first;
    mcLabelsConstView = buffer.second;
  }

  if (buildNativeHost && buildNativeGPU && GetProcessingSettings().delayedOutput) {
    mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = nClsTotal;
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
    tmpNativeClusters = mInputsHost->mPclusterNativeOutput;
    for (uint32_t i = outputQueueStart; i < mOutputQueue.size(); i++) {
      mOutputQueue[i].dst = (char*)tmpNativeClusters + (size_t)mOutputQueue[i].dst;
    }
  }

  if (buildNativeHost) {
    tmpNativeAccess->clustersLinear = tmpNativeClusters;
    tmpNativeAccess->clustersMCTruth = mcLabelsConstView;
    tmpNativeAccess->setOffsetPtrs();
    mIOPtrs.clustersNative = tmpNativeAccess;
    if (GetProcessingSettings().tpcApplyClusterFilterOnCPU) {
      auto allocator = [this, &tmpNativeClusters](size_t size) {
        this->mInputsHost->mNClusterNative = size;
        this->AllocateRegisteredMemory(this->mInputsHost->mResourceClusterNativeOutput, this->mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
        return (tmpNativeClusters = this->mInputsHost->mPclusterNativeOutput);
      };
      RunTPCClusterFilter(tmpNativeAccess, allocator, false);
      nClsTotal = tmpNativeAccess->nClustersTotal;
    }
  }

  if (!mWaitForFinalInputs) {
    notifyForeignChainFinished();
  }

  if (buildNativeGPU) {
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceClusterNativeAccess, 0);
  }
  if (doGPU && synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 1);
  }
  if (doGPU && synchronizeCalibUpdate) {
    SynchronizeStream(0);
  }
  if (buildNativeHost && (GetProcessingSettings().deterministicGPUReconstruction || GetProcessingSettings().debugLevel >= 4)) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
        std::sort(&tmpNativeClusters[tmpNativeAccess->clusterOffset[i][j]], &tmpNativeClusters[tmpNativeAccess->clusterOffset[i][j] + tmpNativeAccess->nClusters[i][j]]);
      }
    }
    if (buildNativeGPU) {
      GPUMemCpy(RecoStep::TPCClusterFinding, (void*)mInputsShadow->mPclusterNativeBuffer, (const void*)tmpNativeClusters, nClsTotal * sizeof(tmpNativeClusters[0]), -1, true);
    }
  }
  mRec->MemoryScalers()->nTPCHits = nClsTotal;
  mRec->PopNonPersistentMemory(RecoStep::TPCClusterFinding, qStr2Tag("TPCCLUST"));
  if (mPipelineNotifyCtx) {
    mRec->UnblockStackedMemory();
    mPipelineNotifyCtx = nullptr;
  }

  if (GetProcessingSettings().autoAdjustHostThreads && !doGPU) {
    mRec->SetNActiveThreads(-1);
  }

#endif
  return 0;
}
