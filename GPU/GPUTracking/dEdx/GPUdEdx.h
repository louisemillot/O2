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

/// \file GPUdEdx.h
/// \author David Rohr

#ifndef GPUDEDX_H
#define GPUDEDX_H

#include "GPUDef.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"
#include "GPUdEdxInfo.h"
#include "DataFormatsTPC/Defs.h"
#include "CalibdEdxContainer.h"
#include "GPUTPCGeometry.h"
#include "GPUDebugStreamer.h"

namespace o2::gpu
{

class GPUdEdx
{
 public:
  // The driver must call clear(), fill clusters row by row outside-in, then run computedEdx() to get the result
  GPUd() void clear();
  GPUd() void fillCluster(float qtot, float qmax, int32_t padRow, uint8_t sector, float trackSnp, float trackTgl, const GPUCalibObjectsConst& calib, float z, float pad, float relTime);
  GPUd() void fillSubThreshold(int32_t padRow);
  GPUd() void computedEdx(GPUdEdxInfo& output, const GPUParam& param);

 private:
  GPUd() float GetSortTruncMean(GPUCA_DEDX_STORAGE_TYPE_A* array, int32_t count, int32_t trunclow, int32_t trunchigh);
  GPUd() void checkSubThresh(int32_t roc);

  template <typename T, typename fake = void>
  struct scalingFactor;
  template <typename fake>
  struct scalingFactor<uint16_t, fake> {
    static constexpr float factor = 4.f;
    static constexpr float round = 0.5f;
  };
  template <typename fake>
  struct scalingFactor<float, fake> {
    static constexpr float factor = 1.f;
    static constexpr float round = 0.f;
  };
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename fake>
  struct scalingFactor<half, fake> {
    static constexpr float factor = 1.f;
    static constexpr float round = 0.f;
  };
#endif

  static constexpr int32_t MAX_NCL = GPUCA_ROW_COUNT; // Must fit in mNClsROC (uint8_t)!

  GPUCA_DEDX_STORAGE_TYPE_A mChargeTot[MAX_NCL]; // No need for default, just some memory
  GPUCA_DEDX_STORAGE_TYPE_A mChargeMax[MAX_NCL]; // No need for default, just some memory
  float mSubThreshMinTot = 0.f;
  float mSubThreshMinMax = 0.f;
  uint8_t mNClsROC[4] = {0};
  uint8_t mNClsROCSubThresh[4] = {0};
  uint8_t mCount = 0;
  uint8_t mLastROC = 255;
  uint8_t mNSubThresh = 0;
};

GPUdi() void GPUdEdx::checkSubThresh(int32_t roc)
{
  if (roc != mLastROC) {
    if (mNSubThresh && mCount + mNSubThresh <= MAX_NCL) {
      for (int32_t i = 0; i < mNSubThresh; i++) {
        mChargeTot[mCount] = (GPUCA_DEDX_STORAGE_TYPE_A)(mSubThreshMinTot * scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::round);
        mChargeMax[mCount++] = (GPUCA_DEDX_STORAGE_TYPE_A)(mSubThreshMinMax * scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::round);
      }
      mNClsROC[mLastROC] += mNSubThresh;
      mNClsROCSubThresh[mLastROC] += mNSubThresh;
    }
    mNSubThresh = 0;
    mSubThreshMinTot = 1e10f;
    mSubThreshMinMax = 1e10f;
  }

  mLastROC = roc;
}

GPUdnii() void GPUdEdx::fillCluster(float qtot, float qmax, int32_t padRow, uint8_t sector, float trackSnp, float trackTgl, const GPUCalibObjectsConst& calib, float z, float pad, float relTime)
{
  if (mCount >= MAX_NCL) {
    return;
  }

  // container containing all the dE/dx corrections
  auto calibContainer = calib.dEdxCalibContainer;
  constexpr GPUTPCGeometry geo;

  const int32_t roc = geo.GetROC(padRow);
  checkSubThresh(roc);
  float snp2 = trackSnp * trackSnp;
  if (snp2 > GPUCA_MAX_SIN_PHI_LOW) {
    snp2 = GPUCA_MAX_SIN_PHI_LOW;
  }

  // setting maximum for snp for which the calibration object was created
  const float snp = CAMath::Abs(trackSnp);

  // tanTheta local dip angle: z angle - dz/dx (cm/cm)
  const float sec2 = 1.f / (1.f - snp2);
  const float tgl2 = trackTgl * trackTgl;
  const float tanTheta = CAMath::Sqrt(tgl2 * sec2);

  // getting the topology correction
  const uint32_t padPos = CAMath::Float2UIntRn(pad); // position of the pad is shifted half a pad ( pad=3 -> centre position of third pad)
  const float absRelPad = CAMath::Abs(pad - padPos);
  const int32_t region = geo.GetRegion(padRow);
  z = CAMath::Abs(z);
  const float threshold = calibContainer->getZeroSupressionThreshold(sector, padRow, padPos); // TODO: Use the mean zero supresion threshold of all pads in the cluster?
  const bool useFullGainMap = calibContainer->isUsageOfFullGainMap();
  float qTotIn = qtot;
  const float fullGainMapGain = calibContainer->getGain(sector, padRow, padPos);
  if (useFullGainMap) {
    qmax /= fullGainMapGain;
    qtot /= fullGainMapGain;
  } else {
    qTotIn *= fullGainMapGain;
  }

  const float qMaxTopologyCorr = calibContainer->getTopologyCorrection(region, o2::tpc::ChargeType::Max, tanTheta, snp, z, absRelPad, relTime, threshold, qTotIn);
  const float qTotTopologyCorr = calibContainer->getTopologyCorrection(region, o2::tpc::ChargeType::Tot, tanTheta, snp, z, absRelPad, relTime, threshold, qTotIn);
  qmax /= qMaxTopologyCorr;
  qtot /= qTotTopologyCorr;

  tpc::StackID stack{
    sector,
    static_cast<tpc::GEMstack>(roc)};

  const float qMaxResidualCorr = calibContainer->getResidualCorrection(stack, tpc::ChargeType::Max, trackTgl, trackSnp);
  const float qTotResidualCorr = calibContainer->getResidualCorrection(stack, tpc::ChargeType::Tot, trackTgl, trackSnp);
  qmax /= qMaxResidualCorr;
  qtot /= qTotResidualCorr;

  const float residualGainMapGain = calibContainer->getResidualGain(sector, padRow, padPos);
  qmax /= residualGainMapGain;
  qtot /= residualGainMapGain;

  mChargeTot[mCount] = (GPUCA_DEDX_STORAGE_TYPE_A)(qtot * scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::round);
  mChargeMax[mCount++] = (GPUCA_DEDX_STORAGE_TYPE_A)(qmax * scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE_A>::round);
  mNClsROC[roc]++;
  if (qtot < mSubThreshMinTot) {
    mSubThreshMinTot = qtot;
  }
  if (qmax < mSubThreshMinMax) {
    mSubThreshMinMax = qmax;
  }

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamdEdx)) {
    float padlx = geo.Row2X(padRow);
    float padly = geo.LinearPad2Y(sector, padRow, padPos);
    o2::utils::DebugStreamer::instance()->getStreamer("debug_dedx", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_dedx").data()
                                                                              << "qTot=" << mChargeTot[mCount - 1]
                                                                              << "qMax=" << mChargeMax[mCount - 1]
                                                                              << "region=" << region
                                                                              << "padRow=" << padRow
                                                                              << "sector=" << sector
                                                                              << "lx=" << padlx
                                                                              << "ly=" << padly
                                                                              << "tanTheta=" << tanTheta
                                                                              << "trackTgl=" << trackTgl
                                                                              << "sinPhi=" << trackSnp
                                                                              << "z=" << z
                                                                              << "absRelPad=" << absRelPad
                                                                              << "relTime=" << relTime
                                                                              << "threshold=" << threshold
                                                                              << "qTotIn=" << qTotIn
                                                                              << "qMaxTopologyCorr=" << qMaxTopologyCorr
                                                                              << "qTotTopologyCorr=" << qTotTopologyCorr
                                                                              << "qMaxResidualCorr=" << qMaxResidualCorr
                                                                              << "qTotResidualCorr=" << qTotResidualCorr
                                                                              << "residualGainMapGain=" << residualGainMapGain
                                                                              << "fullGainMapGain=" << fullGainMapGain
                                                                              << "\n";
  })
}

GPUdi() void GPUdEdx::fillSubThreshold(int32_t padRow)
{
  const int32_t roc = GPUTPCGeometry::GetROC(padRow);
  checkSubThresh(roc);
  mNSubThresh++;
}

} // namespace o2::gpu

#endif
