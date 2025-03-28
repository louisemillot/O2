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

#include "GlobalTrackingStudy/SVStudy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CallbacksPolicy.h"
#include "DetectorsBase/DPLWorkflowUtils.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of track sources to use"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"ignore-tpc-occ", VariantType::Bool, false, {"do not fill TPC occupancy (needs TPC clusters)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  GID::mask_t allowedSourcesTrc = GID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  GID::mask_t srcTrc = allowedSourcesTrc & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  GID::mask_t srcCls{};
  bool fillTPCOcc = !configcontext.options().get<bool>("ignore-tpc-occ");
  if (fillTPCOcc) {
    srcCls = srcCls | GID::getSourcesMask("TPC");
  }
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcCls, srcTrc, srcTrc, useMC);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC); // P-vertex is always needed
  o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);        // S-vertex is always needed
  specs.emplace_back(o2::svstudy::getSVStudySpec(srcTrc, srcCls, useMC));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
