# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# CUDA
if(CUDA_ENABLED)
find_package(CUDAToolkit)
message(STATUS "Building ITS CUDA tracker")
# add_compile_options(-O0 -g -lineinfo -fPIC)
# add_compile_definitions(ITS_MEASURE_GPU_TIME)
o2_add_library(ITStrackingCUDA
               SOURCES ClusterLinesGPU.cu
                       Context.cu
                       Stream.cu
                       TrackerTraitsGPU.cxx
                       TimeFrameGPU.cu
                       TimeFrameChunk.cu
                       TracerGPU.cu
                       TrackingKernels.cu
                       VertexingKernels.cu
                       VertexerTraitsGPU.cxx
                       Utils.cu
               PUBLIC_INCLUDE_DIRECTORIES ../
               PUBLIC_LINK_LIBRARIES O2::ITStracking
                                     O2::SimConfig
                                     O2::SimulationDataFormat
                                     O2::ReconstructionDataFormats
                                     O2::GPUCommon
               PRIVATE_LINK_LIBRARIES O2::GPUTrackingCUDAExternalProvider
               TARGETVARNAME targetName)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)
set_target_cuda_arch(${targetName})

endif()
