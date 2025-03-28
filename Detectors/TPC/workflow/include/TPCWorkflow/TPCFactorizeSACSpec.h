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

/// \file TPCFactorizeSACSpec.h
/// \brief TPC factorization of SACs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jul 6, 2022

#ifndef O2_TPCFACTORIZESACSPEC_H
#define O2_TPCFACTORIZESACSPEC_H

#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/SACFactorization.h"
#include "CCDB/CcdbApi.h"
#include "TPCWorkflow/TPCDistributeSACSpec.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/CDBInterface.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/InputRecordWalker.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFactorizeSACSpec : public o2::framework::Task
{
 public:
  TPCFactorizeSACSpec(const unsigned int timeframes, const SACFactorization::SACDeltaCompression compression, const bool debug, const int lane) : mSACFactorization{timeframes}, mCompressionDeltaSAC{compression}, mDebug{debug}, mLaneId{lane} {};

  void run(o2::framework::ProcessingContext& pc) final
  {
    int countStacks = 0;
    mCCDBTimeStamp = pc.inputs().get<uint64_t>("sacccdb");
    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcStackHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int stack = tpcStackHeader->subSpecification;
      mSACFactorization.setSACs(pc.inputs().get<std::vector<int32_t>>(ref), stack);
      ++countStacks;
    }

    if (countStacks != GEMSTACKS) {
      LOGP(warning, "Received only {} out of {}", countStacks, GEMSTACKS);
    }

    mSACFactorization.factorizeSACs();

    if (mDebug) {
      LOGP(info, "dumping aggregated and factorized SACs to file");
      const auto currTF = processing_helpers::getCurrentTF(pc);
      mSACFactorization.dumpToFile(fmt::format("SACFactorized_{:02}.root", currTF).data());
    }

    // storing to CCDB
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionSAC1() { return header::DataDescription{"SAC1"}; }
  static constexpr header::DataDescription getDataDescriptionTimeStamp() { return header::DataDescription{"FOURIERTSSAC"}; }
  static constexpr header::DataDescription getDataDescriptionLane() { return header::DataDescription{"SACLANE"}; }

  // for CCDB
  static constexpr header::DataDescription getDataDescriptionCCDBSAC() { return header::DataDescription{"TPC_CalibSAC"}; }

 private:
  SACFactorization mSACFactorization;                                                                                                                                     ///< object for performing the factorization of the SACs
  const SACFactorization::SACDeltaCompression mCompressionDeltaSAC{};                                                                                                     ///< compression type for SAC Delta
  const bool mDebug{false};                                                                                                                                               ///< dump SACs to tree for debugging
  const int mLaneId{0};                                                                                                                                                   ///< the id of the current process within the parallel pipeline
  uint64_t mCCDBTimeStamp{0};                                                                                                                                             ///< time stamp of first SACs which are received for the current aggreagtion interval, which is used for setting the time when writing to the CCDB
  const std::vector<InputSpec> mFilter = {{"sac", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeSACSpec::getDataDescriptionSACVec(mLaneId)}, Lifetime::Sporadic}}; ///< filter for looping over input data

  void sendOutput(DataAllocator& output)
  {
    const uint64_t timeStampStart = mCCDBTimeStamp;
    const uint64_t timeStampEnd = timeStampStart + o2::ccdb::CcdbObjectInfo::DAY;

    // do check if received data is empty
    if (timeStampStart != 0) {
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionSAC1(), header::DataHeader::SubSpecificationType{Side::A}}, mSACFactorization.getSACOne(Side::A));
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionSAC1(), header::DataHeader::SubSpecificationType{Side::C}}, mSACFactorization.getSACOne(Side::C));
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionTimeStamp()}, std::vector<uint64_t>{timeStampStart, timeStampEnd});
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionLane()}, mLaneId);

      o2::ccdb::CcdbObjectInfo ccdbInfoSAC(CDBTypeMap.at(CDBType::CalSAC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, timeStampEnd);

      std::unique_ptr<std::vector<char>> imageSACDelta{};
      switch (mCompressionDeltaSAC) {
        case SACFactorization::SACDeltaCompression::MEDIUM:
        default: {
          SAC<unsigned short> sacContainer{mSACFactorization.getSACZero(), mSACFactorization.getSACOne(), mSACFactorization.getSACDeltaMediumCompressed()};
          imageSACDelta = o2::ccdb::CcdbApi::createObjectImage(&sacContainer, &ccdbInfoSAC);
          break;
        }
        case SACFactorization::SACDeltaCompression::HIGH: {
          SAC<unsigned char> sacContainer{mSACFactorization.getSACZero(), mSACFactorization.getSACOne(), mSACFactorization.getSACDeltaHighCompressed()};
          imageSACDelta = o2::ccdb::CcdbApi::createObjectImage(&sacContainer, &ccdbInfoSAC);
          break;
        }
        case SACFactorization::SACDeltaCompression::NO:
          SAC<float> sacContainer{mSACFactorization.getSACZero(), mSACFactorization.getSACOne(), std::move(mSACFactorization).getSACDeltaUncompressed()};
          imageSACDelta = o2::ccdb::CcdbApi::createObjectImage(&sacContainer, &ccdbInfoSAC);
          break;
      }

      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoSAC.getPath(), ccdbInfoSAC.getFileName(), imageSACDelta->size(), ccdbInfoSAC.getStartValidityTimestamp(), ccdbInfoSAC.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBSAC(), 0}, *imageSACDelta.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBSAC(), 0}, ccdbInfoSAC);
    } else {
      LOGP(warning, "Received empty data for SACs! SACs will not be stored for the current aggregation interval!");
    }

    mSACFactorization.reset();
  }
};

DataProcessorSpec getTPCFactorizeSACSpec(const int lane, const unsigned int timeframes, const SACFactorization::SACFactorization::SACDeltaCompression compression, const bool debug)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeSACSpec::getDataDescriptionCCDBSAC()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeSACSpec::getDataDescriptionCCDBSAC()}, Lifetime::Sporadic);

  outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1(), header::DataHeader::SubSpecificationType{Side::A}}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1(), header::DataHeader::SubSpecificationType{Side::C}}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionTimeStamp(), header::DataHeader::SubSpecificationType{0}}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionLane(), header::DataHeader::SubSpecificationType{0}}, Lifetime::Sporadic);

  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"sac", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeSACSpec::getDataDescriptionSACVec(lane)}, Lifetime::Sporadic});
  inputSpecs.emplace_back(InputSpec{"sacccdb", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeSACSpec::getDataDescriptionSACCCDB()}, Lifetime::Sporadic});

  DataProcessorSpec spec{
    fmt::format("tpc-factorize-sac-{:02}", lane).data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFactorizeSACSpec>(timeframes, compression, debug, lane)}};
  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif
