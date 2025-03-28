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

# This is the configuration file for the standalone build
# Its options do not affect the O2 build !!!!

set(ENABLE_CUDA AUTO)
set(ENABLE_HIP AUTO)
set(ENABLE_OPENCL AUTO)
set(GPUCA_CONFIG_VC 1)
set(GPUCA_CONFIG_FMT 1)
set(GPUCA_CONFIG_ROOT 1)
set(GPUCA_BUILD_EVENT_DISPLAY 1)
set(GPUCA_BUILD_EVENT_DISPLAY_FREETYPE 1)
set(GPUCA_BUILD_EVENT_DISPLAY_VULKAN 1)
set(GPUCA_BUILD_EVENT_DISPLAY_WAYLAND 1)
set(GPUCA_BUILD_EVENT_DISPLAY_QT 1)
set(GPUCA_CONFIG_GL3W 0)
set(GPUCA_CONFIG_O2 1)
set(GPUCA_BUILD_DEBUG 0)
set(GPUCA_BUILD_DEBUG_SANITIZE 0)
set(GPUCA_DETERMINISTIC_MODE 0)             # OFF / NO_FAST_MATH / OPTO2 / GPU / WHOLEO2
#set(GPUCA_CUDA_GCCBIN c++-14)
#set(GPUCA_OPENCL_CLANGBIN clang-19)
set(HIP_AMDGPUTARGET "default")             # "gfx906;gfx908;gfx90a"
set(CUDA_COMPUTETARGET "default")           # 86 89
#set(GPUCA_CUDA_COMPILE_MODE perkernel)     # onefile / perkernel / rtc
#set(GPUCA_HIP_COMPILE_MODE perkernel)
#set(GPUCA_KERNEL_RESOURCE_USAGE_VERBOSE 1)
#set(GPUCA_CONFIG_COMPILER gcc)             # gcc / clang
#add_definitions(-DGPUCA_GPU_DEBUG_PRINT)
