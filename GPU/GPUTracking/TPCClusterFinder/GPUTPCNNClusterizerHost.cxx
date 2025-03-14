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

/// \file GPUTPCNNClusterizerHost.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizerHost.h"
#include "GPUTPCNNClusterizer.h"
#include "GPUSettings.h"
#include "ML/3rdparty/GPUORTFloat16.h"

using namespace o2::gpu;

GPUTPCNNClusterizerHost::GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer& settings, GPUTPCNNClusterizer& clusterer)
{
  OrtOptions = {
    {"model-path", settings.nnClassificationPath},
    {"device", settings.nnInferenceDevice},
    {"device-id", std::to_string(settings.nnInferenceDeviceId)},
    {"allocate-device-memory", std::to_string(settings.nnInferenceAllocateDevMem)},
    {"dtype", settings.nnInferenceDtype},
    {"intra-op-num-threads", std::to_string(settings.nnInferenceIntraOpNumThreads)},
    {"inter-op-num-threads", std::to_string(settings.nnInferenceInterOpNumThreads)},
    {"enable-optimizations", std::to_string(settings.nnInferenceEnableOrtOptimization)},
    {"enable-profiling", std::to_string(settings.nnInferenceOrtProfiling)},
    {"profiling-output-path", settings.nnInferenceOrtProfilingPath},
    {"logging-level", std::to_string(settings.nnInferenceVerbosity)}};

  model_class.init(OrtOptions);
  clusterer.nnClusterizerModelClassNumOutputNodes = model_class.getNumOutputNodes()[0][1];

  reg_model_paths = splitString(settings.nnRegressionPath, ":");

  if (!settings.nnClusterizerUseCfRegression) {
    if (model_class.getNumOutputNodes()[0][1] == 1 || reg_model_paths.size() == 1) {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
    } else {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
      OrtOptions["model-path"] = reg_model_paths[1];
      model_reg_2.init(OrtOptions);
      clusterer.nnClusterizerModelReg2NumOutputNodes = model_reg_2.getNumOutputNodes()[0][1];
    }
  }
}

void GPUTPCNNClusterizerHost::networkInference(o2::ml::OrtModel model, GPUTPCNNClusterizer& clusterer, size_t size, float* output, int32_t dtype)
{
  if (dtype == 0) {
    model.inference<OrtDataType::Float16_t, float>(clusterer.inputData16, size * clusterer.nnClusterizerElementSize, output);
  } else {
    model.inference<float, float>(clusterer.inputData32, size * clusterer.nnClusterizerElementSize, output);
  }
}
