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

set(MODULE GPUTracking)

# set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -O0") # to uncomment if needed, tired of typing this...
# set(GPUCA_BUILD_DEBUG 1)

if(GPUCA_DETERMINISTIC_MODE GREATER_EQUAL ${GPUCA_DETERMINISTIC_MODE_MAP_NO_FAST_MATH})
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${GPUCA_CXX_NO_FAST_MATH_FLAGS}")
  if(GPUCA_DETERMINISTIC_MODE GREATER_EQUAL ${GPUCA_DETERMINISTIC_MODE_MAP_OPTO2})
    set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -O2")
  endif()
elseif(NOT CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -O3 -ffast-math")
endif()
set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${GPUCA_CXX_DENORMALS_FLAGS}")

include(cmake/helpers.cmake)

if((ALIGPU_BUILD_TYPE STREQUAL "Standalone" AND GPUCA_BUILD_EVENT_DISPLAY) OR (ALIGPU_BUILD_TYPE STREQUAL "O2" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND TARGET AliceO2::DebugGUI AND OPENGL_FOUND AND GLFW_FOUND))
  set(GPUCA_EVENT_DISPLAY ON)
endif()
if(ROOT_FOUND)
  set(GPUCA_QA ON)
endif()

# General sources
set(SRCS
    SectorTracker/GPUTPCTrack.cxx
    SectorTracker/GPUTPCTrackParam.cxx
    SectorTracker/GPUTPCStartHitsFinder.cxx
    SectorTracker/GPUTPCStartHitsSorter.cxx
    SectorTracker/GPUTPCNeighboursCleaner.cxx
    SectorTracker/GPUTPCTracker.cxx
    SectorTracker/GPUTPCTrackingData.cxx
    SectorTracker/GPUTPCTrackletConstructor.cxx
    SectorTracker/GPUTPCSectorDebugSortKernels.cxx
    SectorTracker/GPUTPCCreateOccupancyMap.cxx
    SectorTracker/GPUTPCNeighboursFinder.cxx
    SectorTracker/GPUTPCGrid.cxx
    SectorTracker/GPUTPCTrackletSelector.cxx
    SectorTracker/GPUTPCExtrapolationTracking.cxx
    SectorTracker/GPUTPCCreateTrackingData.cxx
    Merger/GPUTPCGMMerger.cxx
    Merger/GPUTPCGMSectorTrack.cxx
    Merger/GPUTPCGMTrackParam.cxx
    Merger/GPUTPCGMPropagator.cxx
    Merger/GPUTPCGlobalDebugSortKernels.cxx
    Merger/GPUTPCGMPhysicalTrackModel.cxx
    Merger/GPUTPCGMPolynomialFieldManager.cxx
    DataTypes/GPUTRDTrack.cxx
    TRDTracking/GPUTRDTracker.cxx
    TRDTracking/GPUTRDTrackletWord.cxx
    TRDTracking/GPUTRDTrackerKernels.cxx
    Base/GPUParam.cxx)

set(SRCS_DATATYPES DataTypes/GPUDataTypes.cxx DataTypes/GPUConfigDump.cxx DataTypes/GPUTPCGMPolynomialField.cxx)

set(HDRS_CINT_O2 Merger/GPUTPCGMTrackParam.h Merger/GPUTPCGMMergedTrack.h Merger/GPUTPCGMSectorTrack.h Merger/GPUTPCGMBorderTrack.h TRDTracking/GPUTRDInterfaces.h)
set(HDRS_CINT_DATATYPES DataTypes/GPUTPCGMMergedTrackHit.h)
set(HDRS_CINT_O2_ADDITIONAL DataTypes/GPUSettings.h Definitions/GPUSettingsList.h DataTypes/GPUDataTypes.h DataTypes/GPUTRDTrack.h DataTypes/CalibdEdxTrackTopologyPol.h DataTypes/CalibdEdxTrackTopologySpline.h) # Manual dependencies for ROOT dictionary generation

set(SRCS_NO_CINT
    DataTypes/GPUMemorySizeScalers.cxx
    DataTypes/GPUNewCalibValues.cxx
    DataTypes/GPUTPCClusterOccupancyMap.cxx
    Base/GPUReconstruction.cxx
    Base/GPUReconstructionProcessing.cxx
    Base/GPUReconstructionCPU.cxx
    Base/GPUProcessor.cxx
    Base/GPUMemoryResource.cxx
    Base/GPUGeneralKernels.cxx
    Base/GPUReconstructionDeviceBase.cxx
    Base/GPUReconstructionConvert.cxx
    Base/GPUKernelDebugOutput.cxx
    Global/GPUChain.cxx
    Global/GPUChainTracking.cxx
    Global/GPUTrackingInputProvider.cxx
    Global/GPUErrors.cxx
    Merger/GPUTPCGMMergerGPU.cxx
    Debug/GPUROOTDumpCore.cxx
    Debug/GPUTPCClusterFilter.cxx
    utils/timer.cxx)

set(SRCS_NO_H SectorTracker/GPUTPCTrackerDump.cxx
              Merger/GPUTPCGMMergerDump.cxx
              Base/GPUReconstructionLibrary.cxx
              Global/GPUChainTrackingClusterizer.cxx
              Global/GPUChainTrackingTransformation.cxx
              Global/GPUChainTrackingTRD.cxx
              Global/GPUChainTrackingRefit.cxx
              Global/GPUChainTrackingSectorTracker.cxx
              Global/GPUChainTrackingMerger.cxx
              Global/GPUChainTrackingCompression.cxx
              Global/GPUChainTrackingDebugAndProfiling.cxx
              Global/GPUChainTrackingIO.cxx)

set(HDRS_INSTALL
    Base/GPUConstantMem.h
    Base/GPUParam.inc
    Base/GPUParamRTC.h
    Base/GPUReconstructionIncludes.h
    Base/GPUReconstructionThreading.h
    Base/GPUReconstructionIO.h
    Base/GPUReconstructionIncludesITS.h
    Base/GPUReconstructionKernelMacros.h
    Base/GPUReconstructionKernels.h
    DataCompression/GPUTPCClusterRejection.h
    DataCompression/GPUTPCCompressionKernels.inc
    DataCompression/TPCClusterDecompressionCore.inc
    DataTypes/GPUdEdxInfo.h
    DataTypes/GPUHostDataTypes.h
    DataTypes/GPUO2DataTypes.h
    DataTypes/GPUOutputControl.h
    DataTypes/GPUSettings.h
    DataTypes/GPUTPCGeometry.h
    DataTypes/GPUTPCGMMergedTrackHit.h
    DataTypes/GPUTRDDef.h
    DataTypes/GPUTRDInterfaceO2Track.h
    DataTypes/GPUTriggerOutputs.h
    Debug/GPUROOTDump.h
    Definitions/GPUDefConstantsAndSettings.h
    Definitions/GPUDefGPUParameters.h
    Definitions/GPUDef.h
    Definitions/GPUDefMacros.h
    Definitions/GPULogging.h
    Definitions/GPUSettingsList.h
    Global/GPUChainTrackingDefs.h
    Global/GPUErrorCodes.h
    Merger/GPUTPCGMBorderTrack.h
    Merger/GPUTPCGMMergedTrack.h
    Merger/GPUTPCGMMergerTypes.h
    qa/GPUQAHelper.h
    qconfigoptions.h
    Refit/GPUTrackParamConvert.h
    SectorTracker/GPUTPCBaseTrackParam.h
    SectorTracker/GPUTPCClusterData.h
    SectorTracker/GPUTPCDef.h
    SectorTracker/GPUTPCHit.h
    SectorTracker/GPUTPCHitId.h
    SectorTracker/GPUTPCMCInfo.h
    SectorTracker/GPUTPCSectorOutCluster.h
    SectorTracker/GPUTPCTracklet.h
    SectorTracker/GPUTPCTrackLinearisation.h
    TPCConvert/GPUTPCConvertImpl.h
    TRDTracking/GPUTRDGeometry.h
    TRDTracking/GPUTRDInterfaces.h
    TRDTracking/GPUTRDSpacePoint.h
    TRDTracking/GPUTRDTrackData.h
    TRDTracking/GPUTRDTrackerDebug.h
    TRDTracking/GPUTRDTrackletLabels.h
    TRDTracking/GPUTRDTrackPoint.h
    TRDTracking/GPUTRDTrackPoint.h
    DataTypes/GPUTPCGMPolynomialField.h
)

set(SRCS_NO_CINT ${SRCS_NO_CINT} display/GPUDisplayInterface.cxx)

set(SRCS_NO_CINT ${SRCS_NO_CINT}
    Global/GPUChainITS.cxx
    ITS/GPUITSFitter.cxx
    ITS/GPUITSFitterKernels.cxx
    dEdx/GPUdEdx.cxx
    TPCConvert/GPUTPCConvert.cxx
    TPCConvert/GPUTPCConvertKernel.cxx
    DataCompression/GPUTPCCompression.cxx
    DataCompression/GPUTPCCompressionTrackModel.cxx
    DataCompression/GPUTPCCompressionKernels.cxx
    DataCompression/GPUTPCDecompression.cxx
    DataCompression/GPUTPCDecompressionKernels.cxx
    DataCompression/TPCClusterDecompressor.cxx
    DataCompression/GPUTPCClusterStatistics.cxx
    TPCClusterFinder/GPUTPCClusterFinder.cxx
    TPCClusterFinder/ClusterAccumulator.cxx
    TPCClusterFinder/MCLabelAccumulator.cxx
    TPCClusterFinder/GPUTPCCFCheckPadBaseline.cxx
    TPCClusterFinder/GPUTPCCFStreamCompaction.cxx
    TPCClusterFinder/GPUTPCCFChargeMapFiller.cxx
    TPCClusterFinder/GPUTPCCFPeakFinder.cxx
    TPCClusterFinder/GPUTPCCFNoiseSuppression.cxx
    TPCClusterFinder/GPUTPCCFClusterizer.cxx
    TPCClusterFinder/GPUTPCCFDeconvolution.cxx
    TPCClusterFinder/GPUTPCCFMCLabelFlattener.cxx
    TPCClusterFinder/GPUTPCCFDecodeZS.cxx
    TPCClusterFinder/GPUTPCCFGather.cxx
    Refit/GPUTrackingRefit.cxx
    Refit/GPUTrackingRefitKernel.cxx
    Merger/GPUTPCGMO2Output.cxx)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  list(APPEND SRCS_NO_CINT TPCClusterFinder/GPUTPCNNClusterizerKernels.cxx TPCClusterFinder/GPUTPCNNClusterizer.cxx TPCClusterFinder/GPUTPCNNClusterizerHost.cxx)
endif()

set(SRCS_DATATYPES
    ${SRCS_DATATYPES}
    DataTypes/TPCPadGainCalib.cxx
    DataTypes/TPCPadBitMap.cxx
    DataTypes/TPCZSLinkMapping.cxx
    DataTypes/CalibdEdxContainer.cxx
    DataTypes/CalibdEdxTrackTopologyPol.cxx
    DataTypes/CalibdEdxTrackTopologySpline.cxx
    DataTypes/GPUTRDTrackO2.cxx)

set(SRCS_NO_H ${SRCS_NO_H}
    TPCClusterFinder/GPUTPCClusterFinderDump.cxx)

set(HDRS_INSTALL ${HDRS_INSTALL}
                  ITS/GPUITSTrack.h
                  TPCClusterFinder/Array2D.h
                  TPCClusterFinder/CfConsts.h
                  TPCClusterFinder/CfFragment.h
                  TPCClusterFinder/CfUtils.h
                  TPCClusterFinder/ChargePos.h
                  Definitions/clusterFinderDefs.h
                  TPCClusterFinder/PackedCharge.h
                  TPCClusterFinder/GPUTPCCFChainContext.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(SRCS_DATATYPES
      ${SRCS_DATATYPES}
      Interface/GPUO2InterfaceConfigurableParam.cxx)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include_gpu_onthefly)
file(GENERATE
     OUTPUT include_gpu_onthefly/GPUReconstructionKernelList.h
     INPUT Base/GPUReconstructionKernelList.template.h
)
file(GENERATE
     OUTPUT include_gpu_onthefly/GPUReconstructionKernelIncludes.h
     INPUT Base/GPUReconstructionKernelIncludes.template.h
)
file(GENERATE
     OUTPUT include_gpu_onthefly/GPUReconstructionIncludesDeviceAll.h
     INPUT Base/GPUReconstructionIncludesDeviceAll.template.h
)
file(GENERATE
     OUTPUT include_gpu_onthefly/GPUNoFastMathKernels.h
     INPUT cmake/GPUNoFastMathKernels.template.h
)
if(NOT ALIGPU_BUILD_TYPE STREQUAL "O2")
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include_gpu_onthefly)
endif()
set(HDRS_INSTALL ${HDRS_INSTALL} ${CMAKE_CURRENT_BINARY_DIR}/include_gpu_onthefly/GPUReconstructionKernelList.h)
include(kernels.cmake)

# Optional sources depending on optional dependencies
if(GPUCA_EVENT_DISPLAY)
  add_subdirectory(display)
endif()

if(GPUCA_QA)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} qa/GPUQA.cxx)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} qa/GPUQA.h)
endif()

# Update HDR variables with files derrived from SRC variables
string(REPLACE ".cxx" ".h" HDRS_SRCS "${SRCS}")
string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS_NO_CINT}")
set(HDRS_INSTALL ${HDRS_INSTALL} ${HDRS_TMP})
string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS_DATATYPES}")
set(HDRS_CINT_DATATYPES ${HDRS_CINT_DATATYPES} ${HDRS_TMP})
unset(HDRS_TMP)

set(INCDIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Definitions
    ${CMAKE_CURRENT_SOURCE_DIR}/DataTypes
    ${CMAKE_CURRENT_SOURCE_DIR}/Base
    ${CMAKE_CURRENT_SOURCE_DIR}/SectorTracker
    ${CMAKE_CURRENT_SOURCE_DIR}/TPCConvert
    ${CMAKE_CURRENT_SOURCE_DIR}/dEdx
    ${CMAKE_CURRENT_SOURCE_DIR}/ITS
    ${CMAKE_CURRENT_SOURCE_DIR}/TRDTracking
    ${CMAKE_CURRENT_SOURCE_DIR}/qa
    ${CMAKE_CURRENT_SOURCE_DIR}/Global
    ${CMAKE_CURRENT_SOURCE_DIR}/HLTHeaders
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface
    ${CMAKE_CURRENT_SOURCE_DIR}/Merger
    ${CMAKE_CURRENT_SOURCE_DIR}/Refit
    ${CMAKE_CURRENT_SOURCE_DIR}/Debug
    ${CMAKE_CURRENT_SOURCE_DIR}/DataCompression
    ${CMAKE_CURRENT_SOURCE_DIR}/TPCClusterFinder
    ${CMAKE_CURRENT_BINARY_DIR}/include_gpu_onthefly)

# Main CMake part for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(GPUDataTypes
                 TARGETVARNAME targetName
                 PUBLIC_INCLUDE_DIRECTORIES .
                                            Definitions
                                            DataTypes
                 PUBLIC_LINK_LIBRARIES O2::GPUUtils
                                       O2::GPUCommon
                                       O2::ReconstructionDataFormats
                                       O2::TPCFastTransformation
                 PRIVATE_LINK_LIBRARIES O2::DataFormatsTPC
                 SOURCES ${SRCS_DATATYPES})
  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB GPUCA_TPC_GEOMETRY_O2)

  o2_target_root_dictionary(GPUDataTypes
                            HEADERS ${HDRS_CINT_DATATYPES} ${HDRS_CINT_O2_ADDITIONAL}
                            LINKDEF GPUTrackingLinkDef_O2_DataTypes.h)

  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       O2::GPUUtils
                                       O2::GPUDataTypes
                                       O2::DataFormatsTPC
                                       O2::DataFormatsTOF
                                       O2::TPCBase
                                       O2::TRDBase
                                       O2::TOFBase
                                       O2::ITSBase
                                       O2::ITStracking
                                       O2::TPCFastTransformation
                                       O2::DetectorsRaw
                                       O2::Steer
                                       O2::ML
                 PUBLIC_INCLUDE_DIRECTORIES ${INCDIRS}
                 SOURCES ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})

  target_include_directories(
    ${targetName}
    PRIVATE $<TARGET_PROPERTY:O2::Framework,INTERFACE_INCLUDE_DIRECTORIES>)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB GPUCA_TPC_GEOMETRY_O2 GPUCA_HAS_ONNX=1)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT_O2} ${HDRS_CINT_O2_ADDITIONAL}
                            LINKDEF GPUTrackingLinkDef_O2.h)

  install(FILES ${HDRS_SRCS} ${HDRS_CINT_O2} ${HDRS_CINT_DATATYPES} ${HDRS_INSTALL}
          DESTINATION include/GPU)
  install(DIRECTORY utils
          DESTINATION include/GPU
          FILES_MATCHING PATTERN *.h)

  o2_add_test_root_macro(Standalone/tools/createGeo.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)
  o2_add_test_root_macro(Standalone/tools/createLUT.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)

  add_subdirectory(Interface)
endif()

# Main CMake part for Standalone
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H} ${SRCS_DATATYPES})
  set(targetName ${MODULE})
  add_library(O2::${MODULE} ALIAS ${MODULE})
  install(TARGETS ${targetName})
  target_link_libraries(${targetName} PUBLIC TPCFastTransformation)
  target_include_directories(${targetName} PUBLIC ${INCDIRS})

  if(GPUCA_CONFIG_ROOT)
    ROOT_GENERATE_DICTIONARY(G__${targetName} ${HDRS_CINT_O2} ${HDRS_CINT_DATATYPES} ${HDRS_CINT_O2_ADDITIONAL} GPUTrackingLinkDef_Standalone.h MODULE ${targetName})
    INSTALL(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/lib${targetName}_rdict.pcm
      ${CMAKE_CURRENT_BINARY_DIR}/lib${targetName}.rootmap
    DESTINATION .)
  else()
    target_compile_definitions(${targetName} PRIVATE GPUCA_NO_ROOT)
  endif()
endif()

# GPUReconstructionLibrary needs to know which GPU backends are enabled for proper error messages
configure_file(Base/GPUReconstructionAvailableBackends.template.h ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionAvailableBackends.h)
set_source_files_properties(Base/GPUReconstructionLibrary.cxx
                            PROPERTIES
                            INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")


# Add compile definitions and libraries depending on available optional dependencies
if(GPUCA_QA)
  message(STATUS "Building GPU QA")
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_QA)
endif()

target_link_libraries(${targetName} PRIVATE TBB::tbb)

target_compile_options(${targetName} PRIVATE -Wno-instantiation-after-specialization)

# Add CMake recipes for GPU Tracking librararies
if(CUDA_ENABLED OR OPENCL_ENABLED OR HIP_ENABLED)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    message(WARNING "GPU Tracking disabled on MacOS")
  else()
    if(CUDA_ENABLED)
      add_subdirectory(Base/cuda)
    endif()
    if(OPENCL_ENABLED)
      add_subdirectory(Base/opencl)
    endif()
    if(HIP_ENABLED)
      add_subdirectory(Base/hip)
    endif()
  endif()
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  if(NOT ${CMAKE_SYSTEM} MATCHES Darwin)
    add_subdirectory(Standalone/Benchmark)
  endif()
endif()

if(GPUCA_DETERMINISTIC_MODE GREATER_EQUAL ${GPUCA_DETERMINISTIC_MODE_MAP_GPU})
  target_compile_definitions(${targetName} PRIVATE GPUCA_DETERMINISTIC_MODE)
endif()

# Disable all non-deterministic floating point to make TPC track model encoding / decoding precise
set_source_files_properties(DataCompression/GPUTPCCompressionTrackModel.cxx
                            DataCompression/GPUTPCCompressionKernels.cxx
                            DataCompression/TPCClusterDecompressor.cxx
                            DataCompression/GPUTPCDecompressionKernels.cxx
                            TARGET_DIRECTORY ${targetName}
                            PROPERTIES
                            COMPILE_FLAGS "${GPUCA_CXX_NO_FAST_MATH_FLAGS}"
                            COMPILE_DEFINITIONS "GPUCA_DETERMINISTIC_MODE")
o2_gpu_kernel_set_deterministic(GPUTPCCompressionKernels_step0attached
                                GPUTPCCompressionKernels_step1unattached
                                GPUTPCDecompressionKernels_step0attached
                                GPUTPCDecompressionKernels_step1unattached)
