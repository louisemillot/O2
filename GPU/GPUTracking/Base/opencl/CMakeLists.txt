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

set(MODULE GPUTrackingOCL)
enable_language(ASM)

message(STATUS "Building GPUTracking with OpenCL support")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(CL_SRC ${GPUDIR}/Base/opencl/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode)

set(OCL_FLAGS -Dcl_clang_storage_class_specifiers -cl-std=CLC++2021 ${GPUCA_OCL_DENORMALS_FLAGS})
if(NOT GPUCA_DETERMINISTIC_MODE GREATER_EQUAL ${GPUCA_DETERMINISTIC_MODE_MAP_NO_FAST_MATH})
  set(OCL_FLAGS ${OCL_FLAGS} -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math)
else()
  set(OCL_FLAGS ${OCL_FLAGS} ${GPUCA_OCL_NO_FAST_MATH_FLAGS})
endif()
set(OCL_DEFINECL "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
            "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
            -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
            -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
            -I${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
)

set(SRCS GPUReconstructionOCL.cxx GPUReconstructionOCLKernels.cxx)
set(HDRS GPUReconstructionOCL.h GPUReconstructionOCLIncludesHost.h)

if (NOT DEFINED GPUCA_OCL_SPIRV_VERSION)
  set(GPUCA_OCL_SPIRV_VERSION 1.2)
endif()

if(OPENCL_ENABLED_SPIRV) # BUILD OpenCL intermediate code for SPIR-V target
  # executes clang to create llvm IL code
  # Add -fintegrated-objemitter once we switch to clang >= 17
  cmake_path(GET LLVM_SPIRV PARENT_PATH TMP_LLVM_SPIRV_PATH)
  add_custom_command(
      OUTPUT ${CL_BIN}.spirv
      COMMAND ${CMAKE_COMMAND} -E env "PATH=${TMP_LLVM_SPIRV_PATH}:\$$PATH" ${LLVM_CLANG}
              -O0
              --target=spirv64v${GPUCA_OCL_SPIRV_VERSION}
              -fno-integrated-objemitter
              -ferror-limit=1000 -Wno-invalid-constexpr -Wno-unused-command-line-argument
              ${OCL_FLAGS}
              ${OCL_DEFINECL}
              -o ${CL_BIN}.spirv -c ${CL_SRC}
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS
      COMMENT "Compiling OpenCL CL source file ${CL_SRC} to SPIRV ${CL_BIN}.spirv")

  create_binary_resource(${CL_BIN}.spirv ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.o)
  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.o)
endif()

if(OPENCL_ENABLED) # BUILD OpenCL source code for runtime compilation target
  # executes clang to preprocess
  add_custom_command(
      OUTPUT ${CL_BIN}.src
      COMMAND ${LLVM_CLANG}
              -Wno-unused-command-line-argument
              ${OCL_FLAGS}
              ${OCL_DEFINECL}
              -cl-no-stdinc
              -nostdinc
              -E ${CL_SRC} > ${CL_BIN}.src
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing OpenCL CL source file for run time compilation ${CL_BIN}.src")

  create_binary_resource(${CL_BIN}.src ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.o)
  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.o)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ${SRCS}
                 PUBLIC_LINK_LIBRARIES OpenCL::OpenCL O2::GPUTracking
                 TARGETVARNAME targetName)

  target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
  # the compile_defitions are not propagated automatically on purpose (they are
  # declared PRIVATE) so we are not leaking them outside of the GPU**
  # directories

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} GPUTracking OpenCL)
  install(TARGETS ${MODULE})
  set(targetName ${MODULE})
endif()

if(OPENCL_ENABLED_SPIRV)
  target_compile_definitions(${targetName} PRIVATE OPENCL_ENABLED_SPIRV)
endif()
target_compile_definitions(${targetName} PRIVATE GPUCA_OCL_BUILD_FLAGS=$<JOIN:${OCL_FLAGS},\ > GPUCA_OCL_SPIRV_VERSION=${GPUCA_OCL_SPIRV_VERSION})
