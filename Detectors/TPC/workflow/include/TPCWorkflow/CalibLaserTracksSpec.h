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

#ifndef O2_TPC_CalibLaserTracksSpec_H
#define O2_TPC_CalibLaserTracksSpec_H

/// @file   CalibLaserTracksSpec.h
/// @brief  Device to run tpc laser track calibration

#include "TFile.h"
#include "TPCCalibration/CalibLaserTracks.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "TPCCalibration/VDriftHelper.h"

#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibLaserTracksDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    mWriteDebug = ic.options().get<bool>("write-debug");
    mCalib.setWriteDebugTree(mWriteDebug);
    mMinNumberTFs = ic.options().get<int>("min-tfs");
    mOnlyPublishOnEOS = ic.options().get<bool>("only-publish-on-eos");
    mNormalize = !ic.options().get<bool>("ignore-normalization");
    auto finishFunction = [this]() {
      if (!mPublished) {
        const auto nTFs = mCalib.getCalibData().processedTFs;
        const auto nMatchA = mCalib.getMatchedPairsA();
        const auto nMatchC = mCalib.getMatchedPairsC();
        LOGP(error, "Calibration data was not published, laser track calibration might not have enough statistics: {} ({}) matched tracks in {} TFs on the A (C) < {} min TFs * {} min matches per side per TF? Or eos was not called by the framework.", nMatchA, nMatchC, nTFs, mMinNumberTFs, CalibLaserTracks::MinTrackPerSidePerTF);
      }
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(finishFunction);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto dph = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header);
    if (!dph) {
      LOGP(warning, "CalibLaserTracksDevice::run: No DataProcessingHeader found for \"input\". Only conditions? Skipping event.");
      return;
    }
    mTPCVDriftHelper.extractCCDBInputs(pc);
    if (mTPCVDriftHelper.isUpdated()) {
      mTPCVDriftHelper.acknowledgeUpdate();
      mCalib.setVDriftRef(mTPCVDriftHelper.getVDriftObject().getVDrift());
      LOGP(info, "Updated reference drift velocity to: {}", mTPCVDriftHelper.getVDriftObject().getVDrift());
    }
    const auto startTime = dph->startTime;
    const auto endTime = dph->startTime + dph->duration;
    mRunNumber = processing_helpers::getRunNumber(pc);

    auto data = pc.inputs().get<gsl::span<TrackTPC>>("input");
    mCalib.setTFtimes(startTime, endTime);
    mCalib.fill(data);

    if (!mOnlyPublishOnEOS && mCalib.hasEnoughData(mMinNumberTFs) && !mPublished) {
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "CalibLaserTracksDevice::endOfStream: Finalizing calibration");
    if (!mCalib.hasEnoughData(mMinNumberTFs)) {
      const auto nTFs = mCalib.getCalibData().processedTFs;
      const auto nMatchA = mCalib.getMatchedPairsA();
      const auto nMatchC = mCalib.getMatchedPairsC();
      LOGP(warning, "laser track calibration does not have enough statistics: {} ({}) matched tracks in {} TFs on the A (C) < {} min TFs * {} min matches per side per TF ", nMatchA, nMatchC, nTFs, mMinNumberTFs, CalibLaserTracks::MinTrackPerSidePerTF);
    }
    sendOutput(ec.outputs());
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
      return;
    }
  }

 private:
  CalibLaserTracks mCalib; ///< laser track calibration component
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  uint64_t mRunNumber{0};        ///< processed run number
  int mMinNumberTFs{100};        ///< minimum number of TFs required for good calibration
  bool mPublished{false};        ///< if calibration was already published
  bool mOnlyPublishOnEOS{false}; ///< if to only publish the calibration on EOS, not during running
  bool mNormalize{true};         ///< normalize reference to have mean correction = 1
  bool mWriteDebug{false};       ///< Write debug output

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    mCalib.finalize();
    mCalib.print();

    std::map<std::string, std::string> md;

    using clbUtils = o2::calibration::Utils;
    auto ltrCalib = mCalib.getCalibData();
    if (!ltrCalib.isValid()) {
      LOGP(error, "Invalid Laser calibration (corrections: A-side={}, C-side={}, NTracks: A-side={} C-side={}), will NOT upload to CCDB", ltrCalib.dvCorrectionA, ltrCalib.dvCorrectionC, ltrCalib.nTracksA, ltrCalib.nTracksC);
      return;
    }

    if (mNormalize) {
      ltrCalib.normalize(0.);
      LOGP(info, "After normalization: correction factors: {} / {} for A- / C-Side, reference: {}, vdrift correction: {}", ltrCalib.dvCorrectionA, ltrCalib.dvCorrectionC, ltrCalib.refVDrift, ltrCalib.getDriftVCorrection());
    }

    o2::ccdb::CcdbObjectInfo w;
    auto image = o2::ccdb::CcdbApi::createObjectImage(&ltrCalib, &w);

    md = w.getMetaData();
    md[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(mRunNumber);
    w.setMetaData(md);

    const auto now = std::chrono::system_clock::now();
    const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    // const auto timeStart = ltrCalib.firstTime; //TODO: use once it is a correct time not TFid
    const long timeEnd = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;

    w.setPath("TPC/Calib/LaserTracks");
    w.setStartValidityTimestamp(timeStart);
    w.setEndValidityTimestamp(timeEnd);

    LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr", 0}, w);

    mPublished = true;

    if (mWriteDebug) {
      TFile f("LaserTracks.snapshot.root", "recreate");
      f.WriteObject(&ltrCalib, "ccdb_object");
    }
  }
};

DataProcessorSpec getCalibLaserTracks(const std::string inputSpec)
{
  using device = o2::tpc::CalibLaserTracksDevice;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{"TPC", "LtrCalibData"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs = select(inputSpec.data());
  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);

  return DataProcessorSpec{
    "tpc-calib-laser-tracks",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"write-debug", VariantType::Bool, false, {"write a debug output tree."}},
      {"min-tfs", VariantType::Int, 100, {"minimum number of TFs with enough laser tracks to finalize the calibration."}},
      {"only-publish-on-eos", VariantType::Bool, false, {"only publish the calibration on eos, not during running"}},
      {"ignore-normalization", VariantType::Bool, false, {"ignore normalization of reference to have mean correction factor 1"}},
    }};
}

} // namespace o2::tpc

#endif
