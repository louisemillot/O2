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

/// \file GPUReconstructionIO.h
/// \author David Rohr

#if !defined(GPURECONSTRUCTIONIO_H)
#define GPURECONSTRUCTIONIO_H

#include "GPUReconstruction.h"

namespace o2::gpu
{

template <class T, class S>
inline uint32_t GPUReconstruction::DumpData(FILE* fp, const T* const* entries, const S* num, InOutPointerType type)
{
  int32_t count = getNIOTypeMultiplicity(type);
  uint32_t numTotal = 0;
  for (int32_t i = 0; i < count; i++) {
    numTotal += num[i];
  }
  if (numTotal == 0) {
    return 0;
  }
  fwrite(&type, sizeof(type), 1, fp);
  for (int32_t i = 0; i < count; i++) {
    fwrite(&num[i], sizeof(num[i]), 1, fp);
    if (num[i]) {
      fwrite(entries[i], sizeof(*entries[i]), num[i], fp);
    }
  }
  if (mProcessingSettings.debugLevel >= 2) {
    GPUInfo("Dumped %ld %s", (int64_t)numTotal, IOTYPENAMES[type]);
  }
  return numTotal;
}

template <class T, class S>
inline size_t GPUReconstruction::ReadData(FILE* fp, const T** entries, S* num, std::unique_ptr<T[]>* mem, InOutPointerType type, T** nonConstPtrs)
{
  if (feof(fp)) {
    return 0;
  }
  InOutPointerType inType;
  size_t r, pos = ftell(fp);
  r = fread(&inType, sizeof(inType), 1, fp);
  if (r != 1 || inType != type) {
    fseek(fp, pos, SEEK_SET);
    return 0;
  }

  int32_t count = getNIOTypeMultiplicity(type);
  size_t numTotal = 0;
  for (int32_t i = 0; i < count; i++) {
    r = fread(&num[i], sizeof(num[i]), 1, fp);
    T* m = AllocateIOMemoryHelper(num[i], entries[i], mem[i]);
    if (nonConstPtrs) {
      nonConstPtrs[i] = m;
    }
    if (num[i]) {
      r = fread(m, sizeof(*entries[i]), num[i], fp);
    }
    numTotal += num[i];
  }
  (void)r;
  if (mProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %ld %s", (int64_t)numTotal, IOTYPENAMES[type]);
  }
  return numTotal;
}

template <class T>
inline void GPUReconstruction::DumpFlatObjectToFile(const T* obj, const char* file)
{
  FILE* fp = fopen(file, "w+b");
  if (fp == nullptr) {
    return;
  }
  size_t size[2] = {sizeof(*obj), obj->getFlatBufferSize()};
  fwrite(size, sizeof(size[0]), 2, fp);
  fwrite(obj, 1, size[0], fp);
  fwrite(obj->getFlatBufferPtr(), 1, size[1], fp);
  fclose(fp);
}

template <class T>
inline std::unique_ptr<T> GPUReconstruction::ReadFlatObjectFromFile(const char* file)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return nullptr;
  }
  size_t size[2] = {0}, r;
  r = fread(size, sizeof(size[0]), 2, fp);
  if (r == 0 || size[0] != sizeof(T)) {
    fclose(fp);
    GPUError("ERROR reading %s, invalid size: %ld (%ld expected)", file, (int64_t)size[0], (int64_t)sizeof(T));
    throw std::runtime_error("invalid size");
  }
  std::unique_ptr<T> retVal(new T);
  retVal->destroy();
  char* buf = new char[size[1]]; // Not deleted as ownership is transferred to FlatObject
  r = fread((void*)retVal.get(), 1, size[0], fp);
  r = fread(buf, 1, size[1], fp);
  fclose(fp);
  if (mProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %ld bytes from %s", (int64_t)r, file);
  }
  retVal->clearInternalBufferPtr();
  retVal->setActualBufferAddress(buf);
  retVal->adoptInternalBuffer(buf);
  return retVal;
}

template <class T>
inline void GPUReconstruction::DumpStructToFile(const T* obj, const char* file)
{
  FILE* fp = fopen(file, "w+b");
  if (fp == nullptr) {
    return;
  }
  size_t size = sizeof(*obj);
  fwrite(&size, sizeof(size), 1, fp);
  fwrite(obj, 1, size, fp);
  fclose(fp);
}

template <class T>
inline std::unique_ptr<T> GPUReconstruction::ReadStructFromFile(const char* file)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return nullptr;
  }
  size_t size, r;
  r = fread(&size, sizeof(size), 1, fp);
  if (r == 0 || size != sizeof(T)) {
    fclose(fp);
    GPUError("ERROR reading %s, invalid size: %ld (%ld expected)", file, (int64_t)size, (int64_t)sizeof(T));
    throw std::runtime_error("invalid size");
  }
  std::unique_ptr<T> newObj(new T);
  r = fread(newObj.get(), 1, size, fp);
  fclose(fp);
  if (mProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %ld bytes from %s", (int64_t)r, file);
  }
  return newObj;
}

template <class T>
inline int32_t GPUReconstruction::ReadStructFromFile(const char* file, T* obj)
{
  FILE* fp = fopen(file, "rb");
  if (fp == nullptr) {
    return 1;
  }
  size_t size, r;
  r = fread(&size, sizeof(size), 1, fp);
  if (r == 0) {
    fclose(fp);
    return 1;
  }
  r = fread(obj, 1, size, fp);
  fclose(fp);
  if (mProcessingSettings.debugLevel >= 2) {
    GPUInfo("Read %ld bytes from %s", (int64_t)r, file);
  }
  return 0;
}

} // namespace o2::gpu

#endif
