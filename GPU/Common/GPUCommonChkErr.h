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

/// \file GPUCommonChkErr.h
/// \author David Rohr

// GPUChkErr and GPUChkErrI will both check x for an error, using the loaded backend of GPUReconstruction (requiring GPUReconstruction.h to be included by the user).
// In case of an error, it will print out the corresponding CUDA / HIP / OpenCL error code
// GPUChkErr will download GPUReconstruction error values from GPU, print them, and terminate the application with an exception if an error occured.
// GPUChkErrI will return 0 or 1, depending on whether an error has occurred.
// The Macros must be called ona GPUReconstruction instance, e.g.:
// if (mRec->GPUChkErrI(cudaMalloc(...))) { exit(1); }
// gpuRecObj.GPUChkErr(cudaMalloc(...));

#ifndef GPUCOMMONCHKERR_H
#define GPUCOMMONCHKERR_H

// Please #include "GPUReconstruction.h" in your code, if you use these 2!
#define GPUChkErr(x) GPUChkErrA(x, __FILE__, __LINE__, true)
#define GPUChkErrI(x) GPUChkErrA(x, __FILE__, __LINE__, false)

#endif
