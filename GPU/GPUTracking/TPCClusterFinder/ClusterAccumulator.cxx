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

/// \file ClusterAccumulator.cxx
/// \author Felix Weiglhofer

#include "ClusterAccumulator.h"
#include "GPUTPCGeometry.h"
#include "CfUtils.h"
#include "GPUParam.h"
#include "DataFormatsTPC/ClusterNative.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

GPUd() bool ClusterAccumulator::toNative(const ChargePos& pos, Charge q, tpc::ClusterNative& cn, const GPUParam& param) const
{
  cn.qTot = CAMath::Float2UIntRn(mQtot);
  if (cn.qTot <= param.rec.tpc.cfQTotCutoff) {
    return false;
  }
  cn.qMax = q;
  if (cn.qMax <= param.rec.tpc.cfQMaxCutoff) {
    return false;
  }
  if (mTimeMean < param.rec.tpc.clustersShiftTimebinsClusterizer) {
    return false;
  }
  if (q <= param.rec.tpc.cfQMaxCutoffSingleTime && mTimeSigma == 0) {
    return false;
  }
  if (q <= param.rec.tpc.cfQMaxCutoffSinglePad && mPadSigma == 0) {
    return false;
  }

  bool isEdgeCluster = CfUtils::isAtEdge(pos, param.tpcGeometry.NPads(pos.row()));
  bool wasSplitInTime = mSplitInTime >= param.rec.tpc.cfMinSplitNum;
  bool wasSplitInPad = mSplitInPad >= param.rec.tpc.cfMinSplitNum;
  bool isSingleCluster = (mPadSigma == 0) || (mTimeSigma == 0);

  uint8_t flags = 0;
  flags |= (isEdgeCluster) ? tpc::ClusterNative::flagEdge : 0;
  flags |= (wasSplitInTime) ? tpc::ClusterNative::flagSplitTime : 0;
  flags |= (wasSplitInPad) ? tpc::ClusterNative::flagSplitPad : 0;
  flags |= (isSingleCluster) ? tpc::ClusterNative::flagSingle : 0;

  cn.setTimeFlags(mTimeMean - param.rec.tpc.clustersShiftTimebinsClusterizer, flags);
  cn.setPad(mPadMean);
  cn.setSigmaTime(mTimeSigma);
  cn.setSigmaPad(mPadSigma);

  return true;
}

GPUd() void ClusterAccumulator::update(Charge splitCharge, Delta2 d)
{
  mQtot += splitCharge;
  mPadMean += splitCharge * d.x;
  mTimeMean += splitCharge * d.y;
  mPadSigma += splitCharge * d.x * d.x;
  mTimeSigma += splitCharge * d.y * d.y;
}

GPUd() Charge ClusterAccumulator::updateInner(PackedCharge charge, Delta2 d)
{
  Charge q = charge.unpack();

  update(q, d);

  bool split = charge.isSplit();
  mSplitInTime += (d.y != 0 && split);
  mSplitInPad += (d.x != 0 && split);

  return q;
}

GPUd() Charge ClusterAccumulator::updateOuter(PackedCharge charge, Delta2 d)
{
  Charge q = charge.unpack();

  bool split = charge.isSplit();
  bool has3x3 = charge.has3x3Peak();

  update((has3x3) ? 0.f : q, d);

  mSplitInTime += (d.y != 0 && split && !has3x3);
  mSplitInPad += (d.x != 0 && split && !has3x3);

  return q;
}

GPUd() void ClusterAccumulator::finalize(const ChargePos& pos, Charge q, TPCTime timeOffset, const GPUTPCGeometry& geo, Charge* padBoundaryCharges)
{
  mQtot += q;

  mPadMean /= mQtot;
  mTimeMean /= mQtot;
  mPadSigma /= mQtot;
  mTimeSigma /= mQtot;

  mPadSigma = CAMath::Sqrt(mPadSigma - mPadMean * mPadMean);
  mTimeSigma = CAMath::Sqrt(mTimeSigma - mTimeMean * mTimeMean);

  Pad pad = pos.pad();
  mPadMean += pad;
  mTimeMean += timeOffset + pos.time();

  if (CfUtils::isAtEdge(pos, geo.NPads(pos.row()))) {
    bool leftEdge = (pad < 2);
    bool correct = (leftEdge) ? (pad < mPadMean) : (pad > mPadMean);
    if (leftEdge && pad == 1) {                         // only check charge at boundary if maximum is at least one pad away from boundary
      correct = correct && (padBoundaryCharges[0] > 0); // Only correct if cluster is asymmetric with charge > 0 towards sector boundary, otherwise all charge is found
    } else if (!leftEdge && pad == (geo.NPads(pos.row()) - 2)) {
      correct = correct && (padBoundaryCharges[1] > 0);
    }
    mPadMean = (correct) ? pad : mPadMean;
  }
}
