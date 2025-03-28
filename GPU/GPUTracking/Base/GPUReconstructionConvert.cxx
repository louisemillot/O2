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

/// \file GPUReconstructionConvert.cxx
/// \author David Rohr

#ifdef GPUCA_O2_LIB
#include "DetectorsRaw/RawFileWriter.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Mapper.h"
#endif

#include "GPUReconstructionConvert.h"
#include "TPCFastTransform.h"
#include "GPUTPCClusterData.h"
#include "GPUO2DataTypes.h"
#include "GPUDataTypes.h"
#include "GPUTPCGeometry.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUParam.h"
#include "GPULogging.h"
#include <algorithm>
#include <vector>

#include "clusterFinderDefs.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
#include "DataFormatsTPC/Constants.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/CRU.h"
#include "DetectorsRaw/RDHUtils.h"

#include <oneapi/tbb.h>

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::tpc::constants;
using namespace std::string_literals;

void GPUReconstructionConvert::ConvertNativeToClusterData(o2::tpc::ClusterNativeAccess* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, uint32_t* nClusters, const TPCFastTransform* transform, int32_t continuousMaxTimeBin)
{
  memset(nClusters, 0, NSECTORS * sizeof(nClusters[0]));
  uint32_t offset = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    uint32_t nClSector = 0;
    for (int32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      nClSector += native->nClusters[i][j];
    }
    nClusters[i] = nClSector;
    clusters[i].reset(new GPUTPCClusterData[nClSector]);
    nClSector = 0;
    for (int32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      for (uint32_t k = 0; k < native->nClusters[i][j]; k++) {
        const auto& clin = native->clusters[i][j][k];
        float x = 0, y = 0, z = 0;
        if (continuousMaxTimeBin == 0) {
          transform->Transform(i, j, clin.getPad(), clin.getTime(), x, y, z);
        } else {
          transform->TransformInTimeFrame(i, j, clin.getPad(), clin.getTime(), x, y, z, continuousMaxTimeBin);
        }
        auto& clout = clusters[i].get()[nClSector];
        clout.x = x;
        clout.y = y;
        clout.z = z;
        clout.row = j;
        clout.amp = clin.qTot;
        clout.flags = clin.getFlags();
        clout.id = offset + k;
        nClSector++;
      }
      native->clusterOffset[i][j] = offset;
      offset += native->nClusters[i][j];
    }
  }
}

void GPUReconstructionConvert::ConvertRun2RawToNative(o2::tpc::ClusterNativeAccess& native, std::unique_ptr<ClusterNative[]>& nativeBuffer, const AliHLTTPCRawCluster** rawClusters, uint32_t* nRawClusters)
{
  memset((void*)&native, 0, sizeof(native));
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < nRawClusters[i]; j++) {
      native.nClusters[i][rawClusters[i][j].GetPadRow()]++;
    }
    native.nClustersTotal += nRawClusters[i];
  }
  nativeBuffer.reset(new ClusterNative[native.nClustersTotal]);
  native.clustersLinear = nativeBuffer.get();
  native.setOffsetPtrs();
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      native.nClusters[i][j] = 0;
    }
    for (uint32_t j = 0; j < nRawClusters[i]; j++) {
      const AliHLTTPCRawCluster& org = rawClusters[i][j];
      int32_t row = org.GetPadRow();
      ClusterNative& c = nativeBuffer[native.clusterOffset[i][row] + native.nClusters[i][row]++];
      c.setTimeFlags(org.GetTime(), org.GetFlags());
      c.setPad(org.GetPad());
      c.setSigmaTime(CAMath::Sqrt(org.GetSigmaTime2()));
      c.setSigmaPad(CAMath::Sqrt(org.GetSigmaPad2()));
      c.qMax = org.GetQMax();
      c.qTot = org.GetCharge();
    }
  }
}

int32_t GPUReconstructionConvert::GetMaxTimeBin(const ClusterNativeAccess& native)
{
  float retVal = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      for (uint32_t k = 0; k < native.nClusters[i][j]; k++) {
        if (native.clusters[i][j][k].getTime() > retVal) {
          retVal = native.clusters[i][j][k].getTime();
        }
      }
    }
  }
  return ceil(retVal);
}

int32_t GPUReconstructionConvert::GetMaxTimeBin(const GPUTrackingInOutDigits& digits)
{
  float retVal = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t k = 0; k < digits.nTPCDigits[i]; k++) {
      if (digits.tpcDigits[i][k].getTimeStamp() > retVal) {
        retVal = digits.tpcDigits[i][k].getTimeStamp();
      }
    }
  }
  return ceil(retVal);
}

int32_t GPUReconstructionConvert::GetMaxTimeBin(const GPUTrackingInOutZS& zspages)
{
  float retVal = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    int32_t firstHBF = zspages.sector[i].count[0] ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)zspages.sector[i].zsPtr[0][0]) : 0;
    for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      for (uint32_t k = 0; k < zspages.sector[i].count[j]; k++) {
        const char* page = (const char*)zspages.sector[i].zsPtr[j][k];
        for (uint32_t l = 0; l < zspages.sector[i].nZSPtr[j][k]; l++) {
          o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)(page + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          TPCZSHDR* hdr = (TPCZSHDR*)(page + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
          int32_t nTimeBinSpan = hdr->nTimeBinSpan;
          if (hdr->version >= o2::tpc::ZSVersion::ZSVersionDenseLinkBased) {
            TPCZSHDRV2* hdr2 = (TPCZSHDRV2*)hdr;
            if (hdr2->flags & TPCZSHDRV2::ZSFlags::nTimeBinSpanBit8) {
              nTimeBinSpan += 256;
            }
          }
          uint32_t timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN + nTimeBinSpan;
          if (timeBin > retVal) {
            retVal = timeBin;
          }
        }
      }
    }
  }
  return ceil(retVal);
}

// ------------------------------------------------- TPC ZS -------------------------------------------------

#ifdef GPUCA_TPC_GEOMETRY_O2
namespace o2::gpu
{
namespace // anonymous
{

// ------------------------------------------------- TPC ZS General -------------------------------------------------

typedef std::array<int64_t, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(int64_t)> zsPage;

struct zsEncoder {
  int32_t curRegion = 0, outputRegion = 0;
  uint32_t encodeBits = 0;
  uint32_t zsVersion = 0;
  uint32_t iSector = 0;
  o2::raw::RawFileWriter* raw = nullptr;
  const o2::InteractionRecord* ir = nullptr;
  const GPUParam* param = nullptr;
  bool padding = false;
  int32_t lastEndpoint = -2, lastTime = -1, lastRow = GPUCA_ROW_COUNT;
  int32_t endpoint = 0, outputEndpoint = 0;
  int64_t hbf = -1, nexthbf = 0;
  zsPage* page = nullptr;
  uint8_t* pagePtr = nullptr;
  int32_t bcShiftInFirstHBF = 0;
  int32_t firstTimebinInPage = -1;
  float encodeBitsFactor = 0;
  bool needAnotherPage = false;
  uint32_t packetCounter = 0;
  uint32_t pageCounter = 0;
  void ZSfillEmpty(void* ptr, int32_t shift, uint32_t feeId, int32_t orbit, int32_t linkid);
  static void ZSstreamOut(uint16_t* bufIn, uint32_t& lenIn, uint8_t* bufOut, uint32_t& lenOut, uint32_t nBits);
  int64_t getHbf(int64_t timestamp) { return (timestamp * LHCBCPERTIMEBIN + bcShiftInFirstHBF) / o2::constants::lhc::LHCMaxBunches; }
};

inline void zsEncoder::ZSfillEmpty(void* ptr, int32_t shift, uint32_t feeId, int32_t orbit, int32_t linkid)
{
  o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)ptr;
  o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, orbit);
  o2::raw::RDHUtils::setHeartBeatBC(*rdh, shift);
  o2::raw::RDHUtils::setMemorySize(*rdh, sizeof(o2::header::RAWDataHeader));
  o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
  o2::raw::RDHUtils::setFEEID(*rdh, feeId);
  o2::raw::RDHUtils::setDetectorField(*rdh, 2);
  o2::raw::RDHUtils::setLinkID(*rdh, linkid);
  o2::raw::RDHUtils::setPacketCounter(*rdh, packetCounter++);
  o2::raw::RDHUtils::setPageCounter(*rdh, pageCounter++);
}

inline void zsEncoder::ZSstreamOut(uint16_t* bufIn, uint32_t& lenIn, uint8_t* bufOut, uint32_t& lenOut, uint32_t nBits)
{
  uint32_t byte = 0, bits = 0;
  uint32_t mask = (1 << nBits) - 1;
  for (uint32_t i = 0; i < lenIn; i++) {
    byte |= (bufIn[i] & mask) << bits;
    bits += nBits;
    while (bits >= 8) {
      bufOut[lenOut++] = (uint8_t)(byte & 0xFF);
      byte = byte >> 8;
      bits -= 8;
    }
  }
  if (bits) {
    bufOut[lenOut++] = byte;
  }
  lenIn = 0;
}

static inline auto ZSEncoderGetDigits(const GPUTrackingInOutDigits& in, int32_t i) { return in.tpcDigits[i]; }
static inline auto ZSEncoderGetNDigits(const GPUTrackingInOutDigits& in, int32_t i) { return in.nTPCDigits[i]; }
#ifdef GPUCA_O2_LIB
using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::Sector::MAXSECTOR>;
static inline auto ZSEncoderGetDigits(const DigitArray& in, int32_t i) { return in[i].data(); }
static inline auto ZSEncoderGetNDigits(const DigitArray& in, int32_t i) { return in[i].size(); }
#endif // GPUCA_O2_LIB

// ------------------------------------------------- TPC ZS Original Row-based ZS -------------------------------------------------

struct zsEncoderRow : public zsEncoder {
  std::array<uint16_t, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer = {};
  std::array<uint8_t, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer8 = {};
  TPCZSHDR* hdr = nullptr;
  TPCZSTBHDR* curTBHdr = nullptr;
  uint8_t* nSeq = nullptr;
  int32_t seqLen = 0;
  int32_t endpointStart = 0;
  int32_t nRowsInTB = 0;
  uint32_t streamSize = 0, streamSize8 = 0;
  constexpr static int32_t RAWLNK = rdh_utils::UserLogicLinkID;

  bool checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  bool writeSubPage();
  void init() { encodeBits = zsVersion == 2 ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1; }
  void initPage() {}
  uint32_t encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);

  bool sort(const o2::tpc::Digit a, const o2::tpc::Digit b);
  void decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* page, uint32_t endpoint, uint32_t firstOrbit, uint32_t triggerBC = 0);
};

inline bool zsEncoderRow::sort(const o2::tpc::Digit a, const o2::tpc::Digit b)
{
  int32_t endpointa = GPUTPCGeometry::GetRegion(a.getRow());
  int32_t endpointb = GPUTPCGeometry::GetRegion(b.getRow());
  endpointa = 2 * endpointa + (a.getRow() >= GPUTPCGeometry::GetRegionStart(endpointa) + GPUTPCGeometry::GetRegionRows(endpointa) / 2);
  endpointb = 2 * endpointb + (b.getRow() >= GPUTPCGeometry::GetRegionStart(endpointb) + GPUTPCGeometry::GetRegionRows(endpointb) / 2);
  if (endpointa != endpointb) {
    return endpointa <= endpointb;
  }
  if (a.getTimeStamp() != b.getTimeStamp()) {
    return a.getTimeStamp() < b.getTimeStamp();
  }
  if (a.getRow() != b.getRow()) {
    return a.getRow() < b.getRow();
  }
  return a.getPad() < b.getPad();
}

bool zsEncoderRow::checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  seqLen = 1;
  if (lastRow != tmpBuffer[k].getRow()) {
    endpointStart = GPUTPCGeometry::GetRegionStart(curRegion);
    endpoint = curRegion * 2;
    if (tmpBuffer[k].getRow() >= endpointStart + GPUTPCGeometry::GetRegionRows(curRegion) / 2) {
      endpoint++;
      endpointStart += GPUTPCGeometry::GetRegionRows(curRegion) / 2;
    }
  }
  for (uint32_t l = k + 1; l < tmpBuffer.size(); l++) {
    if (tmpBuffer[l].getRow() == tmpBuffer[k].getRow() && tmpBuffer[l].getTimeStamp() == tmpBuffer[k].getTimeStamp() && tmpBuffer[l].getPad() == tmpBuffer[l - 1].getPad() + 1) {
      seqLen++;
    } else {
      break;
    }
  }
  if (lastEndpoint >= 0 && lastTime != -1 && (int32_t)hdr->nTimeBinSpan + tmpBuffer[k].getTimeStamp() - lastTime >= 256) {
    lastEndpoint = -1;
  }
  if (endpoint == lastEndpoint) {
    uint32_t sizeChk = (uint32_t)(pagePtr - reinterpret_cast<uint8_t*>(page));                                                      // already written
    sizeChk += 2 * (nRowsInTB + (tmpBuffer[k].getRow() != lastRow && tmpBuffer[k].getTimeStamp() == lastTime));                     // TB HDR
    sizeChk += streamSize8;                                                                                                         // in stream buffer
    sizeChk += (lastTime != tmpBuffer[k].getTimeStamp()) && ((sizeChk + (streamSize * encodeBits + 7) / 8) & 1);                    // time bin alignment
    sizeChk += (tmpBuffer[k].getTimeStamp() != lastTime || tmpBuffer[k].getRow() != lastRow) ? 3 : 0;                               // new row overhead
    sizeChk += (lastTime != -1 && tmpBuffer[k].getTimeStamp() > lastTime) ? ((tmpBuffer[k].getTimeStamp() - lastTime - 1) * 2) : 0; // empty time bins
    sizeChk += 2;                                                                                                                   // sequence metadata
    const uint32_t streamSizeChkBits = streamSize * encodeBits + ((lastTime != tmpBuffer[k].getTimeStamp() && (streamSize * encodeBits) % 8) ? (8 - (streamSize * encodeBits) % 8) : 0);
    if (sizeChk + (encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      lastEndpoint = -1;
    } else if (sizeChk + (seqLen * encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      seqLen = ((TPCZSHDR::TPC_ZS_PAGE_SIZE - sizeChk) * 8 - streamSizeChkBits) / encodeBits;
    }
    // sizeChk += (seqLen * encodeBits + streamSizeChkBits + 7) / 8;
    // printf("Endpoint %d (%d), Pos %d, Chk %d, Len %d, rows %d, StreamSize %d %d, time %d (%d), row %d (%d), pad %d\n", endpoint, lastEndpoint, (int32_t) (pagePtr - reinterpret_cast<uint8_t*>(page)), sizeChk, seqLen, nRowsInTB, streamSize8, streamSize, (int32_t)tmpBuffer[k].getTimeStamp(), lastTime, (int32_t)tmpBuffer[k].getRow(), lastRow, tmpBuffer[k].getPad());
  }
  return endpoint != lastEndpoint || tmpBuffer[k].getTimeStamp() != lastTime;
}

bool zsEncoderRow::writeSubPage()
{
  if (pagePtr != reinterpret_cast<uint8_t*>(page)) {
    pagePtr += 2 * nRowsInTB;
    ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
    pagePtr = std::copy(streamBuffer8.data(), streamBuffer8.data() + streamSize8, pagePtr);
    if (pagePtr - reinterpret_cast<uint8_t*>(page) > 8192) {
      throw std::runtime_error("internal error during ZS encoding");
    }
    streamSize8 = 0;
    for (int32_t l = 1; l < nRowsInTB; l++) {
      curTBHdr->rowAddr1()[l - 1] += 2 * nRowsInTB;
    }
  }
  return endpoint != lastEndpoint;
}

uint32_t zsEncoderRow::encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  if (tmpBuffer[k].getTimeStamp() != lastTime) {
    if (lastTime != -1) {
      hdr->nTimeBinSpan += tmpBuffer[k].getTimeStamp() - lastTime - 1;
      pagePtr += (tmpBuffer[k].getTimeStamp() - lastTime - 1) * 2;
    }
    hdr->nTimeBinSpan++;
    if ((pagePtr - reinterpret_cast<uint8_t*>(page)) & 1) {
      pagePtr++;
    }
    curTBHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
    curTBHdr->rowMask |= (endpoint & 1) << 15;
    nRowsInTB = 0;
    lastRow = GPUCA_ROW_COUNT;
  }
  if (tmpBuffer[k].getRow() != lastRow) {
    curTBHdr->rowMask |= 1 << (tmpBuffer[k].getRow() - endpointStart);
    ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
    if (nRowsInTB) {
      curTBHdr->rowAddr1()[nRowsInTB - 1] = (pagePtr - reinterpret_cast<uint8_t*>(page)) + streamSize8;
    }
    nRowsInTB++;
    nSeq = streamBuffer8.data() + streamSize8++;
    *nSeq = 0;
  }
  (*nSeq)++;
  streamBuffer8[streamSize8++] = tmpBuffer[k].getPad();
  streamBuffer8[streamSize8++] = streamSize + seqLen;
  for (int32_t l = 0; l < seqLen; l++) {
    streamBuffer[streamSize++] = (uint16_t)(tmpBuffer[k + l].getChargeFloat() * encodeBitsFactor + 0.5f);
  }
  return seqLen;
}

void zsEncoderRow::decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* decPage, uint32_t decEndpoint, uint32_t firstOrbit, uint32_t triggerBC)
{
  const uint8_t* decPagePtr = reinterpret_cast<const uint8_t*>(decPage);
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)decPagePtr;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  decPagePtr += sizeof(o2::header::RAWDataHeader);
  const TPCZSHDR* decHDR = reinterpret_cast<const TPCZSHDR*>(decPagePtr);
  decPagePtr += sizeof(*decHDR);
  if (decHDR->version != 1 && decHDR->version != 2) {
    throw std::runtime_error("invalid ZS version "s + std::to_string(decHDR->version) + " (1 or 2 expected)"s);
  }
  const float decodeBitsFactor = 1.f / (1 << (encodeBits - 10));
  uint32_t mask = (1 << encodeBits) - 1;
  int32_t cruid = decHDR->cruID;
  uint32_t sector = cruid / 10;
  if (sector != iSector) {
    throw std::runtime_error("invalid TPC sector");
  }
  int32_t region = cruid % 10;
  if ((uint32_t)region != decEndpoint / 2) {
    throw std::runtime_error("CRU ID / endpoint mismatch");
  }
  int32_t nRowsRegion = GPUTPCGeometry::GetRegionRows(region);

  int32_t timeBin = (decHDR->timeOffset + (uint64_t)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
  for (int32_t l = 0; l < decHDR->nTimeBinSpan; l++) {
    if ((decPagePtr - reinterpret_cast<const uint8_t*>(decPage)) & 1) {
      decPagePtr++;
    }
    const TPCZSTBHDR* tbHdr = reinterpret_cast<const TPCZSTBHDR*>(decPagePtr);
    bool upperRows = tbHdr->rowMask & 0x8000;
    if (tbHdr->rowMask != 0 && ((upperRows) ^ ((decEndpoint & 1) != 0))) {
      throw std::runtime_error("invalid endpoint");
    }
    const int32_t rowOffset = GPUTPCGeometry::GetRegionStart(region) + (upperRows ? (nRowsRegion / 2) : 0);
    const int32_t nRows = upperRows ? (nRowsRegion - nRowsRegion / 2) : (nRowsRegion / 2);
    const int32_t nRowsUsed = __builtin_popcount((uint32_t)(tbHdr->rowMask & 0x7FFF));
    decPagePtr += nRowsUsed ? (2 * nRowsUsed) : 2;
    int32_t rowPos = 0;
    for (int32_t m = 0; m < nRows; m++) {
      if ((tbHdr->rowMask & (1 << m)) == 0) {
        continue;
      }
      const uint8_t* rowData = rowPos == 0 ? decPagePtr : (reinterpret_cast<const uint8_t*>(decPage) + tbHdr->rowAddr1()[rowPos - 1]);
      const int32_t nSeqRead = *rowData;
      const uint8_t* adcData = rowData + 2 * nSeqRead + 1;
      int32_t nADC = (rowData[2 * nSeqRead] * encodeBits + 7) / 8;
      decPagePtr += 1 + 2 * nSeqRead + nADC;
      uint32_t byte = 0, bits = 0, posXbits = 0;
      std::array<uint16_t, TPCZSHDR::TPC_ZS_PAGE_SIZE> decBuffer;
      for (int32_t n = 0; n < nADC; n++) {
        byte |= *(adcData++) << bits;
        bits += 8;
        while (bits >= encodeBits) {
          decBuffer[posXbits++] = byte & mask;
          byte = byte >> encodeBits;
          bits -= encodeBits;
        }
      }
      posXbits = 0;
      for (int32_t n = 0; n < nSeqRead; n++) {
        const int32_t decSeqLen = rowData[(n + 1) * 2] - (n ? rowData[n * 2] : 0);
        for (int32_t o = 0; o < decSeqLen; o++) {
          outputBuffer.emplace_back(o2::tpc::Digit{cruid, decBuffer[posXbits++] * decodeBitsFactor, (tpccf::Row)(rowOffset + m), (tpccf::Pad)(rowData[n * 2 + 1] + o), timeBin + l});
        }
      }
      rowPos++;
    }
  }
}

// ------------------------------------------------- TPC ZS Link Based ZS -------------------------------------------------

#ifdef GPUCA_O2_LIB
struct zsEncoderLinkBased : public zsEncoder {
  TPCZSHDRV2* hdr = nullptr;
  TPCZSHDRV2 hdrBuffer;
  int32_t inverseChannelMapping[5][32];
  int32_t nSamples = 0;
  int32_t link = 0;
  bool finishPage = false;
  std::vector<uint16_t> adcValues = {};
  std::bitset<80> bitmask = {};

  void createBitmask(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  void init();
  bool sort(const o2::tpc::Digit a, const o2::tpc::Digit b);
};

void zsEncoderLinkBased::init()
{
  encodeBits = TPCZSHDRV2::TPC_ZS_NBITS_V34;
  for (int32_t i = 0; i < 5; i++) {
    for (int32_t j = 0; j < 32; j++) {
      inverseChannelMapping[i][j] = -1;
    }
  }
  for (int32_t iCRU = 0; iCRU < 2; iCRU++) {
    for (int32_t iChannel = 0; iChannel < 80; iChannel++) {
      int32_t sampaOnFEC = 0, channelOnSAMPA = 0;
      Mapper::getSampaAndChannelOnFEC(iCRU, iChannel, sampaOnFEC, channelOnSAMPA);
      if (inverseChannelMapping[sampaOnFEC][channelOnSAMPA] != -1 && inverseChannelMapping[sampaOnFEC][channelOnSAMPA] != iChannel) {
        GPUError("ERROR: Channel conflict: %d %d: %d vs %d", sampaOnFEC, channelOnSAMPA, inverseChannelMapping[sampaOnFEC][channelOnSAMPA], iChannel);
        throw std::runtime_error("ZS error");
      }
      inverseChannelMapping[sampaOnFEC][channelOnSAMPA] = iChannel;
    }
  }
  for (int32_t i = 0; i < 5; i++) {
    for (int32_t j = 0; j < 32; j++) {
      if (inverseChannelMapping[i][j] == -1) {
        GPUError("ERROR: Map missing for sampa %d channel %d", i, j);
        throw std::runtime_error("ZS error");
      }
    }
  }
}

void zsEncoderLinkBased::createBitmask(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  const auto& mapper = Mapper::instance();
  nSamples = 0;
  adcValues.clear();
  bitmask.reset();
  uint32_t l;
  for (l = k; l < tmpBuffer.size(); l++) {
    const auto& a = tmpBuffer[l];
    int32_t cruinsector = GPUTPCGeometry::GetRegion(a.getRow());
    o2::tpc::GlobalPadNumber pad = mapper.globalPadNumber(o2::tpc::PadPos(a.getRow(), a.getPad()));
    o2::tpc::FECInfo fec = mapper.fecInfo(pad);
    o2::tpc::CRU cru = cruinsector;
    int32_t fecInPartition = fec.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();
    int32_t tmpEndpoint = 2 * cruinsector + (fecInPartition >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
    if (l == k) {
      link = fecInPartition;
      endpoint = tmpEndpoint;
    } else if (endpoint != tmpEndpoint || link != fecInPartition || tmpBuffer[l].getTimeStamp() != tmpBuffer[k].getTimeStamp()) {
      break;
    }
    int32_t channel = inverseChannelMapping[fec.getSampaChip()][fec.getSampaChannel()];
    bitmask[channel] = 1;
    adcValues.emplace_back((uint16_t)(a.getChargeFloat() * encodeBitsFactor + 0.5f));
  }
  nSamples = l - k;
}

bool zsEncoderLinkBased::sort(const o2::tpc::Digit a, const o2::tpc::Digit b)
{
  // Fixme: this is blasphemy... one shoult precompute all values and sort an index array
  int32_t cruinsectora = GPUTPCGeometry::GetRegion(a.getRow());
  int32_t cruinsectorb = GPUTPCGeometry::GetRegion(b.getRow());
  if (cruinsectora != cruinsectorb) {
    return cruinsectora < cruinsectorb;
  }
  const auto& mapper = Mapper::instance();
  o2::tpc::GlobalPadNumber pada = mapper.globalPadNumber(o2::tpc::PadPos(a.getRow(), a.getPad()));
  o2::tpc::GlobalPadNumber padb = mapper.globalPadNumber(o2::tpc::PadPos(b.getRow(), b.getPad()));
  o2::tpc::FECInfo feca = mapper.fecInfo(pada);
  o2::tpc::FECInfo fecb = mapper.fecInfo(padb);
  o2::tpc::CRU cru = cruinsectora;
  int32_t fecInPartitiona = feca.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();
  int32_t fecInPartitionb = fecb.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();

  int32_t endpointa = 2 * cruinsectora + (fecInPartitiona >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
  int32_t endpointb = 2 * cruinsectorb + (fecInPartitionb >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
  if (endpointa != endpointb) {
    return endpointa < endpointb;
  }
  if (a.getTimeStamp() != b.getTimeStamp()) {
    return a.getTimeStamp() < b.getTimeStamp();
  }
  if (fecInPartitiona != fecInPartitionb) {
    return fecInPartitiona < fecInPartitionb;
  }
  return inverseChannelMapping[feca.getSampaChip()][feca.getSampaChannel()] < inverseChannelMapping[fecb.getSampaChip()][fecb.getSampaChannel()];
}

// ------------------------------------------------- TPC Improved Link Based ZS -------------------------------------------------

struct zsEncoderImprovedLinkBased : public zsEncoderLinkBased {
  bool checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  uint32_t encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  void decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* page, uint32_t endpoint, uint32_t firstOrbit, uint32_t triggerBC = 0);
  bool writeSubPage();
  void initPage();

  constexpr static int32_t RAWLNK = rdh_utils::ILBZSLinkID;
};

bool zsEncoderImprovedLinkBased::checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  createBitmask(tmpBuffer, k);
  finishPage = endpoint != lastEndpoint;
  if (firstTimebinInPage != -1 && tmpBuffer[k].getTimeStamp() - firstTimebinInPage >= 1 << (sizeof(hdr->nTimeBinSpan) * 8)) {
    finishPage = true;
  }
  if (!finishPage) {
    uint32_t sizeChk = (uint32_t)(pagePtr - reinterpret_cast<uint8_t*>(page));
    sizeChk += sizeof(o2::tpc::zerosupp_link_based::CommonHeader);
    if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
      sizeChk += (nSamples * TPCZSHDRV2::TPC_ZS_NBITS_V34 + 127) / 128 * 16;
    } else {
      sizeChk += (nSamples + 2 * TPCZSHDRV2::SAMPLESPER64BIT - 1) / (2 * TPCZSHDRV2::SAMPLESPER64BIT) * 16;
    }
    if (sizeChk > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      finishPage = true;
    }
  }
  return finishPage;
}

uint32_t zsEncoderImprovedLinkBased::encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  o2::tpc::zerosupp_link_based::CommonHeader* tbHdr = (o2::tpc::zerosupp_link_based::CommonHeader*)pagePtr;
  pagePtr += sizeof(*tbHdr);
  tbHdr->bunchCrossing = (tmpBuffer[k].getTimeStamp() - firstTimebinInPage) * LHCBCPERTIMEBIN;
  tbHdr->magicWord = o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZS;
  tbHdr->bitMaskHigh = (bitmask >> 64).to_ulong();
  tbHdr->bitMaskLow = (bitmask & std::bitset<80>(0xFFFFFFFFFFFFFFFFlu)).to_ulong();
  tbHdr->syncOffsetBC = 0;
  tbHdr->fecInPartition = link;
  hdr->nTimeBinSpan = tmpBuffer[k].getTimeStamp() - firstTimebinInPage;
  hdr->nTimebinHeaders++;
  if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
    tbHdr->numWordsPayload = (nSamples * TPCZSHDRV2::TPC_ZS_NBITS_V34 + 127) / 128; // tightly packed ADC samples
    uint32_t tmp = 0;
    uint32_t tmpIn = nSamples;
    ZSstreamOut(adcValues.data(), tmpIn, pagePtr, tmp, encodeBits);
  } else {
    tbHdr->numWordsPayload = (nSamples + 2 * TPCZSHDRV2::SAMPLESPER64BIT - 1) / (2 * TPCZSHDRV2::SAMPLESPER64BIT);
    uint64_t* payloadPtr = (uint64_t*)pagePtr;
    for (uint32_t i = 0; i < 2 * tbHdr->numWordsPayload; i++) {
      payloadPtr[i] = 0;
    }
    for (uint32_t i = 0; i < nSamples; i++) {
      payloadPtr[i / TPCZSHDRV2::SAMPLESPER64BIT] |= ((uint64_t)adcValues[i]) << ((i % TPCZSHDRV2::SAMPLESPER64BIT) * TPCZSHDRV2::TPC_ZS_NBITS_V34);
    }
  }
  pagePtr += tbHdr->numWordsPayload * 16;
  return nSamples;
}

bool zsEncoderImprovedLinkBased::writeSubPage()
{
  return finishPage;
}

void zsEncoderImprovedLinkBased::initPage()
{
  hdr->magicWord = o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader;
  hdr->nTimebinHeaders = 0;
  hdr->firstZSDataOffset = 0;
}

void zsEncoderImprovedLinkBased::decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* decPage, uint32_t decEndpoint, uint32_t firstOrbit, uint32_t triggerBC)
{
  const auto& mapper = Mapper::instance();
  const uint8_t* decPagePtr = reinterpret_cast<const uint8_t*>(decPage);
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)decPagePtr;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  decPagePtr += sizeof(o2::header::RAWDataHeader);
  const TPCZSHDRV2* decHDR = reinterpret_cast<const TPCZSHDRV2*>(decPagePtr);
  decPagePtr += sizeof(*decHDR);
  if (decHDR->version != ZSVersion::ZSVersionLinkBasedWithMeta) {
    throw std::runtime_error("invalid ZS version "s + std::to_string(decHDR->version) + " ("s + std::to_string(ZSVersion::ZSVersionLinkBasedWithMeta) + " expected)"s);
  }
  if (decHDR->magicWord != o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader) {
    throw std::runtime_error("Magic word missing");
  }
  const float decodeBitsFactor = 1.f / (1 << (encodeBits - 10));
  uint32_t mask = (1 << encodeBits) - 1;
  int32_t cruid = decHDR->cruID;
  uint32_t sector = cruid / 10;
  if (sector != iSector) {
    throw std::runtime_error("invalid TPC sector");
  }
  int32_t region = cruid % 10;
  decPagePtr += decHDR->firstZSDataOffset * 16;
  for (uint32_t i = 0; i < decHDR->nTimebinHeaders; i++) {
    const o2::tpc::zerosupp_link_based::Header* tbHdr = (const o2::tpc::zerosupp_link_based::Header*)decPagePtr;
#if 0 // Decoding using the function for the original linkZS
    o2::tpc::CRU cru = cruid % 10;
    const int32_t feeLink = tbHdr->fecInPartition - (decEndpoint & 1) * ((mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
    auto fillADC = [&outputBuffer](int32_t cru, int32_t rowInSector, int32_t padInRow, int32_t timeBin, float adcValue) {
      outputBuffer.emplace_back(o2::tpc::Digit{cruid, adcValue, rowInSector, padInRow, timeBin});
      return true;
    };
    size_t size = sizeof(*tbHdr) + tbHdr->numWordsPayload * 16;
    raw_processing_helpersa::processZSdata((const char*)decPagePtr, size, rdh_utils::getFEEID(cruid, decEndpoint & 1, feeLink), o2::raw::RDHUtils::getHeartBeatOrbit(*rdh), firstOrbit, decHDR->timeOffset, fillADC);
#else // Decoding directly
    if (!tbHdr->isLinkZS()) {
      throw std::runtime_error("ZS TB Hdr does not have linkZS magic word");
    }
    int32_t timeBin = (int32_t(decHDR->timeOffset) + int32_t(tbHdr->bunchCrossing) + (int32_t)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches - triggerBC) / LHCBCPERTIMEBIN;
    if (timeBin < 0) {
      LOGP(debug, "zsEncoderImprovedLinkBased::decodePage skipping digits hdr->tOff {} + hdr->bc {} + (orbit {} - firstOrbit {}) * maxBunch {} - triggerBC {} = {} < 0", decHDR->timeOffset, tbHdr->bunchCrossing, o2::raw::RDHUtils::getHeartBeatOrbit(*rdh), firstOrbit, o2::constants::lhc::LHCMaxBunches, triggerBC, timeBin);
      continue;
    }
    const uint8_t* adcData = (const uint8_t*)(decPagePtr + sizeof(*tbHdr));
    const auto& bitmask = tbHdr->getChannelBits();
    int32_t nADC = bitmask.count();
    std::vector<uint16_t> decBuffer(nADC);
    if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
      uint32_t byte = 0, bits = 0, posXbits = 0;
      while (posXbits < nADC) {
        byte |= *(adcData++) << bits;
        bits += 8;
        while (bits >= encodeBits) {
          decBuffer[posXbits++] = byte & mask;
          byte = byte >> encodeBits;
          bits -= encodeBits;
        }
      }
    } else {
      const uint64_t* adcData64 = (const uint64_t*)adcData;
      for (int32_t j = 0; j < nADC; j++) {
        decBuffer[j] = (adcData64[j / TPCZSHDRV2::SAMPLESPER64BIT] >> ((j % TPCZSHDRV2::SAMPLESPER64BIT) * TPCZSHDRV2::TPC_ZS_NBITS_V34)) & mask;
      }
    }
    for (int32_t j = 0, k = 0; j < bitmask.size(); j++) {
      if (bitmask[j]) {
        int32_t sampaOnFEC = 0, channelOnSAMPA = 0;
        mapper.getSampaAndChannelOnFEC(cruid, j, sampaOnFEC, channelOnSAMPA);
        const auto padSecPos = mapper.padSecPos(cruid, tbHdr->fecInPartition, sampaOnFEC, channelOnSAMPA);
        const auto& padPos = padSecPos.getPadPos();
        outputBuffer.emplace_back(o2::tpc::Digit{cruid, decBuffer[k++] * decodeBitsFactor, (tpccf::Row)padPos.getRow(), (tpccf::Pad)padPos.getPad(), timeBin});
      }
    }
#endif
    decPagePtr += sizeof(*tbHdr) + tbHdr->numWordsPayload * 16;
  }
}

// ------------------------------------------------- TPC ZS Dense Link Based ZS -------------------------------------------------

struct zsEncoderDenseLinkBased : public zsEncoderLinkBased {
  bool checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  uint32_t encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k);
  void decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* page, uint32_t endpoint, uint32_t firstOrbit, uint32_t triggerBC = 0);
  bool writeSubPage();
  void initPage();
  void amendPageErrorMessage(std::ostringstream& oss, const o2::header::RAWDataHeader* rdh, const TPCZSHDRV2* decHDR, const uint8_t* payloadEnd, const uint8_t* decPagePtr, uint32_t nOutput);

  uint16_t curTimeBin = 0;
  std::vector<uint8_t> sequenceBuffer;
  std::vector<uint16_t> sequenceBufferADC;

  constexpr static int32_t RAWLNK = rdh_utils::DLBZSLinkID;
  constexpr static int32_t v2nbits = 10;
};

bool zsEncoderDenseLinkBased::checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  createBitmask(tmpBuffer, k);
  finishPage = endpoint != lastEndpoint;
  uint16_t newTimeBin = tmpBuffer[k].getTimeStamp() - firstTimebinInPage;
  bool retVall = finishPage || newTimeBin != curTimeBin;
  return retVall;
}

uint32_t zsEncoderDenseLinkBased::encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, uint32_t k)
{
  if (sequenceBuffer.size() == 0) {
    uint16_t bc = (int64_t)tmpBuffer[k].getTimeStamp() * LHCBCPERTIMEBIN - (int64_t)hbf * o2::constants::lhc::LHCMaxBunches;
    if (zsVersion == ZSVersion::ZSVersionDenseLinkBasedV2) {
      bc &= 0xFFC;
    }
    sequenceBuffer.emplace_back(bc << 4);
    sequenceBuffer.emplace_back(bc >> 4);
    curTimeBin = tmpBuffer[k].getTimeStamp() - firstTimebinInPage;
    hdr->nTimeBinSpan = curTimeBin & 0xFF;
    if (curTimeBin & 0x100) {
      hdr->flags |= TPCZSHDRV2::ZSFlags::nTimeBinSpanBit8;
    }
    hdr->nTimebinHeaders++;
  }
  sequenceBuffer[0]++;

  sequenceBuffer.emplace_back(link);
  uint8_t* plink = &sequenceBuffer.back();

  std::bitset<10> bitmaskL2;
  for (int32_t i = 9; i >= 0; i--) {
    bitmaskL2.set(i, ((bitmask >> (i * 8)) & std::bitset<80>(0xFF)).any());
  }
  if (bitmaskL2.all()) {
    *plink |= 0b00100000;
  } else {
    *plink |= (bitmaskL2.to_ulong() >> 2) & 0b11000000;
    sequenceBuffer.emplace_back(bitmaskL2.to_ulong() & 0xFF);
  }

  for (int32_t i = 0; i < 10; i++) {
    if (bitmaskL2.test(i)) {
      sequenceBuffer.emplace_back(((bitmask >> (i * 8)) & std::bitset<80>(0xFF)).to_ulong());
    }
  }

  static_assert(TPCZSHDRV2::TPC_ZS_NBITS_V34 == 12);
  if (nSamples) {
    sequenceBufferADC.insert(sequenceBufferADC.end(), adcValues.begin(), adcValues.end());
  }

  return nSamples;
}

bool zsEncoderDenseLinkBased::writeSubPage()
{
  uint32_t offset = sequenceBuffer.size();
  if (sequenceBufferADC.size()) {
    bool need12bit = zsVersion != ZSVersion::ZSVersionDenseLinkBasedV2;
    uint32_t needNow = 0;
    if (zsVersion == ZSVersion::ZSVersionDenseLinkBasedV2) {
      for (uint32_t i = 0; i < sequenceBufferADC.size(); i++) {
        if (sequenceBufferADC[i] >= (1 << v2nbits)) {
          need12bit = true;
          break;
        }
      }
    }
    uint32_t encodeBitsBlock = encodeBits;
    if (!need12bit) {
      encodeBitsBlock = v2nbits;
      sequenceBuffer[0] |= 0x10;
    }
    sequenceBuffer.resize(offset + (sequenceBufferADC.size() * encodeBitsBlock + 7) / 8);
    uint32_t tmp = 0;
    uint32_t tmpIn = sequenceBufferADC.size();
    ZSstreamOut(sequenceBufferADC.data(), tmpIn, sequenceBuffer.data() + offset, tmp, encodeBitsBlock);
    sequenceBufferADC.clear();
  }

  if (sequenceBuffer.size()) {
    uint32_t sizeLeft = TPCZSHDR::TPC_ZS_PAGE_SIZE - (pagePtr - (uint8_t*)page) - sizeof(TPCZSHDRV2) - (hdr->flags & TPCZSHDRV2::ZSFlags::TriggerWordPresent ? TPCZSHDRV2::TRIGGER_WORD_SIZE : 0);
    uint32_t size = sequenceBuffer.size();
    uint32_t fill = std::min(sizeLeft, size);
    memcpy(pagePtr, sequenceBuffer.data(), fill);
    pagePtr += fill;
    if (size != fill) {
      hdr->flags |= o2::tpc::TPCZSHDRV2::ZSFlags::payloadExtendsToNextPage;
      sequenceBuffer.erase(sequenceBuffer.begin(), sequenceBuffer.begin() + fill);
    } else {
      sequenceBuffer.clear();
    }
    finishPage = finishPage || size >= sizeLeft || needAnotherPage;
  }

  return finishPage;
}

void zsEncoderDenseLinkBased::initPage()
{
  hdr->magicWord = o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader;
  hdr->nTimebinHeaders = 0;
  memcpy(pagePtr, sequenceBuffer.data(), sequenceBuffer.size());
  hdr->firstZSDataOffset = sequenceBuffer.size() + sizeof(o2::header::RAWDataHeader);
  pagePtr += sequenceBuffer.size();
  sequenceBuffer.clear();
  hdr->flags = 0;
}

void zsEncoderDenseLinkBased::decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* decPage, uint32_t decEndpoint, uint32_t firstOrbit, uint32_t triggerBC)
{
  const auto& mapper = Mapper::instance();
  const uint8_t* decPagePtr = reinterpret_cast<const uint8_t*>(decPage);
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)decPagePtr;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  const TPCZSHDRV2* decHDR = reinterpret_cast<const TPCZSHDRV2*>(decPagePtr + o2::raw::RDHUtils::getMemorySize(*rdh) - sizeof(TPCZSHDRV2));
  decPagePtr += sizeof(o2::header::RAWDataHeader);
  if (decHDR->version < ZSVersion::ZSVersionDenseLinkBased || decHDR->version > ZSVersion::ZSVersionDenseLinkBasedV2) {
    throw std::runtime_error("invalid ZS version "s + std::to_string(decHDR->version) + " ("s + std::to_string(ZSVersion::ZSVersionDenseLinkBased) + " - "s + std::to_string(ZSVersion::ZSVersionDenseLinkBasedV2) + " expected)"s);
  }
  if (decHDR->magicWord != o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader) {
    throw std::runtime_error("Magic word missing");
  }
  const uint8_t* payloadEnd = ((const uint8_t*)decPage) + o2::raw::RDHUtils::getMemorySize(*rdh) - sizeof(TPCZSHDRV2) - ((decHDR->flags & TPCZSHDRV2::ZSFlags::TriggerWordPresent) ? TPCZSHDRV2::TRIGGER_WORD_SIZE : 0);
  const float decodeBitsFactor = 1.f / (1 << (encodeBits - 10));
  int32_t cruid = decHDR->cruID;
  uint32_t sector = cruid / 10;
  if (sector != iSector) {
    throw std::runtime_error("invalid TPC sector");
  }
  int32_t region = cruid % 10;
  decPagePtr += decHDR->firstZSDataOffset - sizeof(o2::header::RAWDataHeader);
  std::vector<uint8_t> tmpBuffer;
  bool extendFailure = false;
  uint32_t nOutput = 0;
  uint32_t minTimeBin = -1, maxTimeBin = 0;
  for (uint32_t i = 0; i < decHDR->nTimebinHeaders; i++) {
    int32_t sizeLeftInPage = payloadEnd - decPagePtr;
    if (sizeLeftInPage <= 0) {
      throw std::runtime_error("Decoding ran beyond end of page before processing extended timebin");
    }
    if (i == decHDR->nTimebinHeaders - 1 && (decHDR->flags & o2::tpc::TPCZSHDRV2::ZSFlags::payloadExtendsToNextPage)) {
      if (o2::raw::RDHUtils::getMemorySize(*rdh) != TPCZSHDR::TPC_ZS_PAGE_SIZE) {
        throw std::runtime_error("pageExtends signaled, but current page is not full");
      }

      const uint8_t* pageNext = ((const uint8_t*)decPage) + TPCZSHDR::TPC_ZS_PAGE_SIZE;
      const o2::header::RAWDataHeader* rdhNext = (const o2::header::RAWDataHeader*)pageNext;

      if ((uint16_t)(o2::raw::RDHUtils::getPageCounter(*rdh) + 1) != o2::raw::RDHUtils::getPageCounter(*rdhNext)) {
        GPUError("Incomplete HBF: Payload extended to next page, but next page missing in stream (packet counters %d %d)", (int32_t)o2::raw::RDHUtils::getPageCounter(*rdh), (int32_t)o2::raw::RDHUtils::getPageCounter(*rdhNext));
        extendFailure = true;
        decPagePtr = payloadEnd; // Next 8kb page is missing in stream, cannot decode remaining data, skip it
        break;
      }

      const TPCZSHDRV2* hdrNext = reinterpret_cast<const TPCZSHDRV2*>(pageNext + o2::raw::RDHUtils::getMemorySize(*rdhNext) - sizeof(TPCZSHDRV2));
      tmpBuffer.resize(sizeLeftInPage + hdrNext->firstZSDataOffset - sizeof(o2::header::RAWDataHeader));
      memcpy(tmpBuffer.data(), decPagePtr, sizeLeftInPage);
      memcpy(tmpBuffer.data() + sizeLeftInPage, pageNext + sizeof(o2::header::RAWDataHeader), hdrNext->firstZSDataOffset - sizeof(o2::header::RAWDataHeader));
      decPagePtr = tmpBuffer.data();
      payloadEnd = decPagePtr + tmpBuffer.size();
    }
    uint8_t linkCount = *((const uint8_t*)decPagePtr) & 0x0F;
    uint16_t linkBC = (*((const uint16_t*)decPagePtr) & 0xFFF0) >> 4;
    bool v2Flag = decHDR->version == ZSVersion::ZSVersionDenseLinkBasedV2 && *((const uint8_t*)decPagePtr) & 0x10;
    if (decHDR->version == ZSVersion::ZSVersionDenseLinkBasedV2) {
      linkBC &= 0xFFC;
    }
    decPagePtr += sizeof(uint16_t);
    std::vector<int32_t> links;
    std::vector<std::bitset<80>> bitmasks;
    uint32_t nTotalSamples = 0;
    for (uint32_t l = 0; l < linkCount; l++) {
      uint8_t decLinkX = *((const uint8_t*)decPagePtr);
      decPagePtr += sizeof(uint8_t);
      uint8_t decLink = decLinkX & 0b00011111;
      std::bitset<10> bitmaskL2;
      if (decLinkX & 0b00100000) {
        bitmaskL2.set();
      } else {
        bitmaskL2 = std::bitset<10>(((((uint16_t)decLinkX) & 0b11000000) << 2) | (uint16_t)*((const uint8_t*)decPagePtr));
        decPagePtr += sizeof(uint8_t);
      }

      std::bitset<80> bitmask(0);
      for (int32_t i = 0; i < 10; i++) {
        if (bitmaskL2.test(i)) {
          bitmask |= std::bitset<80>(*((const uint8_t*)decPagePtr)) << i * 8;
          decPagePtr += sizeof(uint8_t);
        }
      }
      links.emplace_back(decLink);
      bitmasks.emplace_back(bitmask);
      nTotalSamples += bitmask.count();
    }

    const uint8_t* adcData = (const uint8_t*)decPagePtr;
    int32_t encodeBitsBlock = v2Flag ? v2nbits : encodeBits;
    decPagePtr += (nTotalSamples * encodeBitsBlock + 7) / 8;

    // time bin might be smaller 0 due to triggerBC
    int32_t timeBin = (int32_t(linkBC) + (int32_t)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches - int32_t(triggerBC)) / LHCBCPERTIMEBIN;
    if (timeBin < 0 || nTotalSamples == 0) {
      if (timeBin < 0 && nTotalSamples > 0) {
        LOGP(debug, "zsEncoderDenseLinkBased::decodePage skipping digits (linkBC {} + orbit {} - firstOrbit {}) * maxBunch {} - triggerBC {} = {} < 0, nTotalSamples {}", linkBC, o2::raw::RDHUtils::getHeartBeatOrbit(*rdh), firstOrbit, o2::constants::lhc::LHCMaxBunches, triggerBC, timeBin, nTotalSamples);
      }
      continue;
    }
    if (timeBin > maxTimeBin) {
      maxTimeBin = timeBin;
    }
    if (timeBin < minTimeBin) {
      minTimeBin = timeBin;
    }

    std::vector<uint16_t> samples(nTotalSamples);
    uint32_t mask = (1 << encodeBitsBlock) - 1;
    uint32_t byte = 0, bits = 0, posXbits = 0;
    while (posXbits < nTotalSamples) {
      byte |= *(adcData++) << bits;
      bits += 8;
      while (bits >= encodeBitsBlock && posXbits < nTotalSamples) {
        samples[posXbits++] = byte & mask;
        byte = byte >> encodeBitsBlock;
        bits -= encodeBitsBlock;
      }
    }
    uint32_t samplePos = 0;

    for (uint32_t l = 0; l < linkCount; l++) {
      uint8_t decLink = links[l];
      const auto& bitmask = bitmasks[l];
      int32_t nADC = bitmask.count();

      for (int32_t j = 0; j < bitmask.size(); j++) {
        if (bitmask[j]) {
          int32_t sampaOnFEC = 0, channelOnSAMPA = 0;
          mapper.getSampaAndChannelOnFEC(cruid, j, sampaOnFEC, channelOnSAMPA);
          const auto padSecPos = mapper.padSecPos(cruid, decLink, sampaOnFEC, channelOnSAMPA);
          const auto& padPos = padSecPos.getPadPos();
          outputBuffer.emplace_back(o2::tpc::Digit{cruid, samples[samplePos++] * decodeBitsFactor, (tpccf::Row)padPos.getRow(), (tpccf::Pad)padPos.getPad(), timeBin});
          nOutput++;
        }
      }
    }
  }

  int32_t hdrMinTimeBin = (int32_t(decHDR->timeOffset) + int32_t(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches - triggerBC);
  if (triggerBC > 0 && hdrMinTimeBin < 0) {
    hdrMinTimeBin = 0;
  }
  hdrMinTimeBin /= LHCBCPERTIMEBIN;
  int32_t hdrMaxTimeBin = hdrMinTimeBin + decHDR->nTimeBinSpan + ((decHDR->flags & TPCZSHDRV2::ZSFlags::nTimeBinSpanBit8) ? 256 : 0);

  if (!extendFailure && nOutput != decHDR->nADCsamples) {
    std::ostringstream oss;
    oss << "Number of decoded digits " << nOutput << " does not match value from MetaInfo " << decHDR->nADCsamples;
    amendPageErrorMessage(oss, rdh, decHDR, nullptr, nullptr, nOutput);
    throw std::runtime_error(oss.str());
  }

  if (decHDR->nADCsamples && (minTimeBin < hdrMinTimeBin || maxTimeBin > hdrMaxTimeBin)) {
    std::ostringstream oss;
    oss << "Incorrect time bin range in MetaInfo, header reports " << hdrMinTimeBin << " - " << hdrMaxTimeBin << "(timeOffset: " << decHDR->timeOffset << " + (orbit: " << o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) << " - firstOrbit " << firstOrbit << ") * LHCMaxBunches - triggerBC: " << triggerBC << ", decoded data is " << minTimeBin << " - " << maxTimeBin;
    amendPageErrorMessage(oss, rdh, decHDR, payloadEnd, decPagePtr, nOutput);
    throw std::runtime_error(oss.str());
  }

  if (decHDR->nTimebinHeaders && payloadEnd - decPagePtr < 0) {
    std::ostringstream oss;
    oss << "Decoding ran over end of page";
    amendPageErrorMessage(oss, rdh, decHDR, payloadEnd, decPagePtr, nOutput);
    throw std::runtime_error(oss.str());
  }
  if (decHDR->nTimebinHeaders && payloadEnd - decPagePtr >= 2 * o2::raw::RDHUtils::GBTWord128) {
    std::ostringstream oss;
    oss << "Decoding didn't reach end of page";
    amendPageErrorMessage(oss, rdh, decHDR, payloadEnd, decPagePtr, nOutput);
    throw std::runtime_error(oss.str());
  }
}

void zsEncoderDenseLinkBased::amendPageErrorMessage(std::ostringstream& oss, const o2::header::RAWDataHeader* rdh, const TPCZSHDRV2* decHDR, const uint8_t* payloadEnd, const uint8_t* decPagePtr, uint32_t nOutput)
{
  if (payloadEnd && decPagePtr) {
    oss << " (payloadEnd " << (void*)payloadEnd << " - decPagePtr " << (void*)decPagePtr << " - " << (payloadEnd - decPagePtr) << " bytes left, " << nOutput << " of " << decHDR->nADCsamples << " digits decoded)\n";
  } else {
    oss << "\n";
  }
  constexpr size_t bufferSize = 3 * std::max(sizeof(*rdh), sizeof(*decHDR)) + 1;
  char dumpBuffer[bufferSize];
  for (size_t i = 0; i < sizeof(*rdh); i++) {
    snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int32_t)((uint8_t*)rdh)[i]);
  }
  oss << "RDH of page: " << dumpBuffer << "\n";
  for (size_t i = 0; i < sizeof(*decHDR); i++) {
    snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int32_t)((uint8_t*)decHDR)[i]);
  }
  oss << "Meta header of page: " << dumpBuffer << "\n";
}

#endif // GPUCA_O2_LIB

// ------------------------------------------------- TPC ZS Main Encoder -------------------------------------------------

template <class T>
struct zsEncoderRun : public T {
  uint32_t run(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer, size_t* totalSize = nullptr);
  size_t compare(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer);

  using T::bcShiftInFirstHBF;
  using T::checkInput;
  using T::curRegion;
  using T::decodePage;
  using T::encodeBits;
  using T::encodeBitsFactor;
  using T::encodeSequence;
  using T::endpoint;
  using T::firstTimebinInPage;
  using T::getHbf;
  using T::hbf;
  using T::hdr;
  using T::init;
  using T::initPage;
  using T::ir;
  using T::iSector;
  using T::lastEndpoint;
  using T::lastRow;
  using T::lastTime;
  using T::needAnotherPage;
  using T::nexthbf;
  using T::outputEndpoint;
  using T::outputRegion;
  using T::packetCounter;
  using T::padding;
  using T::page;
  using T::pageCounter;
  using T::pagePtr;
  using T::param;
  using T::raw;
  using T::sort;
  using T::writeSubPage;
  using T::ZSfillEmpty;
  using T::zsVersion;
};

template <class T>
inline uint32_t zsEncoderRun<T>::run(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer, size_t* totalSize)
{
  uint32_t totalPages = 0;
  zsPage singleBuffer;
#ifdef GPUCA_O2_LIB
  int32_t maxhbf = 0;
  int32_t minhbf = o2::constants::lhc::LHCMaxBunches;
#endif
  bcShiftInFirstHBF = ir ? ir->bc : 0;
  int32_t orbitShift = ir ? ir->orbit : 0;
  int32_t rawcru = 0;
  int32_t rawendpoint = 0;
  (void)(rawcru + rawendpoint); // avoid compiler warning
  encodeBitsFactor = (1 << (encodeBits - 10));

  std::sort(tmpBuffer.begin(), tmpBuffer.end(), [this](const o2::tpc::Digit a, const o2::tpc::Digit b) { return sort(a, b); });
  for (uint32_t k = 0; k <= tmpBuffer.size();) {
    bool mustWritePage = false, mustWriteSubPage = false;
    if (needAnotherPage) {
      needAnotherPage = false;
      mustWritePage = true;
    } else {
      if (k < tmpBuffer.size()) {
        if (tmpBuffer[k].getTimeStamp() != lastTime) {
          nexthbf = getHbf(tmpBuffer[k].getTimeStamp());
          if (nexthbf < 0) {
            throw std::runtime_error("Received digit before the defined first orbit");
          }
          if (hbf != nexthbf) {
            lastEndpoint = -2;
            mustWritePage = true;
          }
        }
        if (lastRow != tmpBuffer[k].getRow()) {
          curRegion = GPUTPCGeometry::GetRegion(tmpBuffer[k].getRow());
        }
        mustWriteSubPage = checkInput(tmpBuffer, k);
      } else {
        nexthbf = -1;
        mustWritePage = true;
      }
    }
    if (mustWritePage || mustWriteSubPage) {
      mustWritePage |= writeSubPage();

      if (page && mustWritePage) {
        if constexpr (std::is_same_v<T, struct zsEncoderDenseLinkBased>) {
          if ((pagePtr - (uint8_t*)page) % o2::raw::RDHUtils::GBTWord128) {
            pagePtr += o2::raw::RDHUtils::GBTWord128 - (pagePtr - (uint8_t*)page) % o2::raw::RDHUtils::GBTWord128;
          }
          uint8_t* triggerWord = nullptr;
          if (hbf != nexthbf || endpoint != lastEndpoint) {
            if ((pagePtr - (uint8_t*)page) + sizeof(TPCZSHDRV2) + o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE <= TPCZSHDR::TPC_ZS_PAGE_SIZE) {
              if ((pagePtr - (uint8_t*)page) % (2 * o2::raw::RDHUtils::GBTWord128)) {
                pagePtr += o2::raw::RDHUtils::GBTWord128; // align to 256 bit, size constrained cannot be affected by this
              }
              hdr->flags |= o2::tpc::TPCZSHDRV2::ZSFlags::TriggerWordPresent;
            } else {
              needAnotherPage = true;
            }
            if (this->sequenceBuffer.size()) {
              needAnotherPage = true;
            }
          }
          if (hdr->flags & o2::tpc::TPCZSHDRV2::TriggerWordPresent) {
            triggerWord = pagePtr;
            pagePtr += o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE;
          }
          if ((pagePtr - (uint8_t*)page) % (2 * o2::raw::RDHUtils::GBTWord128) == 0) {
            pagePtr += o2::raw::RDHUtils::GBTWord128; // align to 128bit mod 256
          }
          TPCZSHDRV2* pagehdr = (TPCZSHDRV2*)pagePtr;
          pagePtr += sizeof(TPCZSHDRV2);
          if (pagePtr - (uint8_t*)page > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
            throw std::runtime_error("TPC ZS page overflow");
          }
          memcpy(pagehdr, hdr, sizeof(*hdr));
          if (triggerWord) {
            memset(triggerWord, 0, o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE);
          }
        }
        const rdh_utils::FEEIDType rawfeeid = rdh_utils::getFEEID(rawcru, rawendpoint, this->RAWLNK);
        if (totalSize) {
          *totalSize += !std::is_same_v<T, struct zsEncoderDenseLinkBased> && (lastEndpoint == -1 || hbf == nexthbf) ? TPCZSHDR::TPC_ZS_PAGE_SIZE : (pagePtr - (uint8_t*)page);
        }
        size_t size = !std::is_same_v<T, struct zsEncoderDenseLinkBased> && (padding || lastEndpoint == -1 || hbf == nexthbf) ? TPCZSHDR::TPC_ZS_PAGE_SIZE : (pagePtr - (uint8_t*)page);
        size = CAMath::nextMultipleOf<o2::raw::RDHUtils::GBTWord128>(size);
#ifdef GPUCA_O2_LIB
        if (raw) {
          raw->addData(rawfeeid, rawcru, 0, rawendpoint, *ir + hbf * o2::constants::lhc::LHCMaxBunches, gsl::span<char>((char*)page + sizeof(o2::header::RAWDataHeader), (char*)page + size), true, 0, 2);
          maxhbf = std::max<int32_t>(maxhbf, hbf);
          minhbf = std::min<int32_t>(minhbf, hbf);
        } else
#endif
        {
          o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)page;
          o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, hbf + orbitShift);
          o2::raw::RDHUtils::setHeartBeatBC(*rdh, bcShiftInFirstHBF);
          o2::raw::RDHUtils::setMemorySize(*rdh, size);
          o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
          o2::raw::RDHUtils::setFEEID(*rdh, rawfeeid);
          o2::raw::RDHUtils::setDetectorField(*rdh, 2);
          o2::raw::RDHUtils::setLinkID(*rdh, this->RAWLNK);
          o2::raw::RDHUtils::setPacketCounter(*rdh, packetCounter++);
          o2::raw::RDHUtils::setPageCounter(*rdh, pageCounter++);
        }
      }
      if (k >= tmpBuffer.size() && !needAnotherPage) {
        break;
      }
    }
    if (mustWritePage) {
      if (!needAnotherPage) {
        if (hbf != nexthbf) {
          pageCounter = 0;
        }
        outputRegion = curRegion;
        outputEndpoint = endpoint;
        hbf = nexthbf;
        lastTime = -1;
        lastEndpoint = endpoint;
      }
      if (raw) {
        page = &singleBuffer;
      } else {
        if (buffer[outputEndpoint].size() == 0 && nexthbf > orbitShift) {
          buffer[outputEndpoint].emplace_back();
          ZSfillEmpty(&buffer[outputEndpoint].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(iSector * 10 + outputEndpoint / 2, outputEndpoint & 1, this->RAWLNK), orbitShift, this->RAWLNK); // Emplace empty page with RDH containing beginning of TF
          if (totalSize) {
            *totalSize += sizeof(o2::header::RAWDataHeader);
          }
          totalPages++;
        }
        buffer[outputEndpoint].emplace_back();
        page = &buffer[outputEndpoint].back();
      }
      pagePtr = reinterpret_cast<uint8_t*>(page);
      std::fill(page->begin(), page->end(), 0);
      pagePtr += sizeof(o2::header::RAWDataHeader);
      if constexpr (std::is_same_v<T, struct zsEncoderDenseLinkBased>) {
        hdr = &this->hdrBuffer;
      } else {
        hdr = reinterpret_cast<decltype(hdr)>(pagePtr);
        pagePtr += sizeof(*hdr);
      }
      hdr->version = zsVersion;
      hdr->cruID = iSector * 10 + outputRegion;
      hdr->nTimeBinSpan = 0;
      hdr->nADCsamples = 0;
      rawcru = iSector * 10 + outputRegion;
      rawendpoint = outputEndpoint & 1;
      hdr->timeOffset = (int64_t)(needAnotherPage ? firstTimebinInPage : tmpBuffer[k].getTimeStamp()) * LHCBCPERTIMEBIN - (int64_t)hbf * o2::constants::lhc::LHCMaxBunches;
      firstTimebinInPage = tmpBuffer[k].getTimeStamp();
      initPage();
      totalPages++;
    }
    if (needAnotherPage) {
      continue;
    }
    uint32_t nEncoded = encodeSequence(tmpBuffer, k);
    lastTime = tmpBuffer[k].getTimeStamp();
    lastRow = tmpBuffer[k].getRow();
    hdr->nADCsamples += nEncoded;
    k += nEncoded;
  }
  if (raw) {
#ifdef GPUCA_O2_LIB
    if (iSector == 0) {
      for (int32_t i = minhbf; i <= maxhbf; i++) {
        raw->addData(46208, 360, rdh_utils::SACLinkID, 0, *ir + i * o2::constants::lhc::LHCMaxBunches, gsl::span<char>((char*)&singleBuffer, (char*)&singleBuffer), true, 0, 4);
      }
    }
#endif
  } else {
    for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      if (buffer[j].size() == 0) {
        buffer[j].emplace_back();
        ZSfillEmpty(&buffer[j].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(iSector * 10 + j / 2, j & 1, this->RAWLNK), orbitShift, this->RAWLNK);
        totalPages++;
      }
    }
  }
  return totalPages;
}

template <class T>
size_t zsEncoderRun<T>::compare(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer)
{
  size_t nErrors = 0;
  std::vector<o2::tpc::Digit> compareBuffer;
  compareBuffer.reserve(tmpBuffer.size());
  for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    uint32_t firstOrbit = ir ? ir->orbit : 0;
    for (uint32_t k = 0; k < buffer[j].size(); k++) {
      zsPage* decPage = &buffer[j][k];
      decodePage(compareBuffer, decPage, j, firstOrbit);
    }
  }
  if (tmpBuffer.size() != compareBuffer.size()) {
    nErrors += tmpBuffer.size();
    printf("Number of clusters mismatch %d %d\n", (int32_t)tmpBuffer.size(), (int32_t)compareBuffer.size());
  } else {
    for (uint32_t j = 0; j < tmpBuffer.size(); j++) {
      const float decodeBitsFactor = (1 << (encodeBits - 10));
      const float c = CAMath::Round(tmpBuffer[j].getChargeFloat() * decodeBitsFactor) / decodeBitsFactor;
      int32_t ok = c == compareBuffer[j].getChargeFloat() && (int32_t)tmpBuffer[j].getTimeStamp() == (int32_t)compareBuffer[j].getTimeStamp() && (int32_t)tmpBuffer[j].getPad() == (int32_t)compareBuffer[j].getPad() && (int32_t)tmpBuffer[j].getRow() == (int32_t)compareBuffer[j].getRow();
      if (ok) {
        continue;
      }
      nErrors++;
      printf("%4u: OK %d: Charge %3d %3d Time %4d %4d Pad %3d %3d Row %3d %3d\n", j, ok,
             (int32_t)c, (int32_t)compareBuffer[j].getChargeFloat(), (int32_t)tmpBuffer[j].getTimeStamp(), (int32_t)compareBuffer[j].getTimeStamp(), (int32_t)tmpBuffer[j].getPad(), (int32_t)compareBuffer[j].getPad(), (int32_t)tmpBuffer[j].getRow(), (int32_t)compareBuffer[j].getRow());
    }
  }
  return nErrors;
}

} // anonymous namespace
} // namespace o2::gpu
#endif // GPUCA_TPC_GEOMETRY_O2

template <class S>
void GPUReconstructionConvert::RunZSEncoder(const S& in, std::unique_ptr<uint64_t[]>* outBuffer, uint32_t* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, const GPUParam& param, int32_t version, bool verify, float threshold, bool padding, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter)
{
  // Pass in either outBuffer / outSizes, to fill standalone output buffers, or raw to use RawFileWriter
  // ir is the interaction record for time bin 0
  if (((outBuffer == nullptr) ^ (outSizes == nullptr)) || ((raw != nullptr) && (ir == nullptr)) || !((outBuffer == nullptr) ^ (raw == nullptr)) || (raw && verify)) {
    throw std::runtime_error("Invalid parameters");
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::vector<zsPage> buffer[NSECTORS][GPUTrackingInOutZS::NENDPOINTS];
  struct tmpReductionResult {
    uint32_t totalPages = 0;
    size_t totalSize = 0;
    size_t nErrors = 0;
    size_t digitsInput = 0;
    size_t digitsEncoded = 0;
  };
  auto reduced = tbb::parallel_reduce(tbb::blocked_range<uint32_t>(0, NSECTORS), tmpReductionResult(), [&](const auto range, auto red) {
    for (uint32_t i = range.begin(); i < range.end(); i++) {
      std::vector<o2::tpc::Digit> tmpBuffer;
      red.digitsInput += ZSEncoderGetNDigits(in, i);
      tmpBuffer.resize(ZSEncoderGetNDigits(in, i));
      if (threshold > 0.f && !digitsFilter) {
        auto it = std::copy_if(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin(), [threshold](auto& v) { return v.getChargeFloat() >= threshold; });
        tmpBuffer.resize(std::distance(tmpBuffer.begin(), it));
      } else {
        std::copy(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin());
      }

      if (digitsFilter) {
        digitsFilter(tmpBuffer);
        if (threshold > 0.f) {
          std::vector<o2::tpc::Digit> tmpBuffer2 = std::move(tmpBuffer);
          tmpBuffer = std::vector<o2::tpc::Digit>(tmpBuffer2.size());
          auto it = std::copy_if(tmpBuffer2.begin(), tmpBuffer2.end(), tmpBuffer.begin(), [threshold](auto& v) { return v.getChargeFloat() >= threshold; });
          tmpBuffer.resize(std::distance(tmpBuffer.begin(), it));
        }
      }
      red.digitsEncoded += tmpBuffer.size();

      auto runZS = [&](auto& encoder) {
        encoder.zsVersion = version;
        encoder.init();
        red.totalPages += encoder.run(buffer[i], tmpBuffer, &red.totalSize);
        if (verify) {
          red.nErrors += encoder.compare(buffer[i], tmpBuffer); // Verification
        }
      };

      if (version >= ZSVersion::ZSVersionRowBased10BitADC && version <= ZSVersion::ZSVersionRowBased12BitADC) {
        zsEncoderRun<zsEncoderRow> enc{{{.iSector = i, .raw = raw, .ir = ir, .param = &param, .padding = padding}}};
        runZS(enc);
      } else if (version >= ZSVersion::ZSVersionLinkBasedWithMeta && version <= ZSVersion::ZSVersionDenseLinkBasedV2) {
#ifdef GPUCA_O2_LIB
        if (version == ZSVersion::ZSVersionLinkBasedWithMeta) {
          zsEncoderRun<zsEncoderImprovedLinkBased> enc{{{{.iSector = i, .raw = raw, .ir = ir, .param = &param, .padding = padding}}}};
          runZS(enc);
        } else if (version >= ZSVersion::ZSVersionDenseLinkBased && version <= ZSVersion::ZSVersionDenseLinkBasedV2) {
          zsEncoderRun<zsEncoderDenseLinkBased> enc{{{{.iSector = i, .raw = raw, .ir = ir, .param = &param, .padding = padding}}}};
          runZS(enc);
        }
#else
        throw std::runtime_error("Link based ZS encoding not supported in standalone build");
#endif
      } else {
        throw std::runtime_error("Invalid ZS version "s + std::to_string(version) + ", cannot decode"s);
      }
    }
    return red; }, [&](const auto& red1, const auto& red2) {
    auto red = red1;
    red.totalPages += red2.totalPages;
    red.totalSize += red2.totalSize;
    red.nErrors += red2.nErrors;
    red.digitsInput += red2.digitsInput;
    red.digitsEncoded += red2.digitsEncoded;
    return red; });

  if (outBuffer) {
    outBuffer->reset(new uint64_t[reduced.totalPages * TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(uint64_t)]);
    uint64_t offset = 0;
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        memcpy((char*)outBuffer->get() + offset, buffer[i][j].data(), buffer[i][j].size() * TPCZSHDR::TPC_ZS_PAGE_SIZE);
        offset += buffer[i][j].size() * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        outSizes[i * GPUTrackingInOutZS::NENDPOINTS + j] = buffer[i][j].size();
      }
    }
  }
  if (reduced.nErrors) {
    GPUError("ERROR: %lu INCORRECT SAMPLES DURING ZS ENCODING VERIFICATION!!!", reduced.nErrors);
  } else if (verify) {
    GPUInfo("ENCODING VERIFICATION PASSED");
  }
  GPUInfo("TOTAL ENCODED SIZE: %lu (%lu of %lu digits encoded)", reduced.totalSize, reduced.digitsEncoded, reduced.digitsInput);
#endif
}

template void GPUReconstructionConvert::RunZSEncoder<GPUTrackingInOutDigits>(const GPUTrackingInOutDigits&, std::unique_ptr<uint64_t[]>*, uint32_t*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, int32_t, bool, float, bool, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter);
#ifdef GPUCA_O2_LIB
template void GPUReconstructionConvert::RunZSEncoder<DigitArray>(const DigitArray&, std::unique_ptr<uint64_t[]>*, uint32_t*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, int32_t, bool, float, bool, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter);
#endif

void GPUReconstructionConvert::RunZSEncoderCreateMeta(const uint64_t* buffer, const uint32_t* sizes, void** ptrs, GPUTrackingInOutZS* out)
{
  uint64_t offset = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      ptrs[i * GPUTrackingInOutZS::NENDPOINTS + j] = (char*)buffer + offset;
      offset += sizes[i * GPUTrackingInOutZS::NENDPOINTS + j] * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      out->sector[i].zsPtr[j] = &ptrs[i * GPUTrackingInOutZS::NENDPOINTS + j];
      out->sector[i].nZSPtr[j] = &sizes[i * GPUTrackingInOutZS::NENDPOINTS + j];
      out->sector[i].count[j] = 1;
    }
  }
}

void GPUReconstructionConvert::RunZSFilter(std::unique_ptr<o2::tpc::Digit[]>* buffers, const o2::tpc::Digit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit, float threshold)
{
  for (uint32_t i = 0; i < NSECTORS; i++) {
    if (buffers[i].get() != ptrs[i] || nsb != ns) {
      throw std::runtime_error("Not owning digits");
    }
    uint32_t j = 0;
    const uint32_t decodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
    const float decodeBitsFactor = (1 << (decodeBits - 10));
    for (uint32_t k = 0; k < ns[i]; k++) {
      if (buffers[i][k].getChargeFloat() >= threshold) {
        if (k > j) {
          buffers[i][j] = buffers[i][k];
        }
        if (zs12bit) {
          buffers[i][j].setCharge(CAMath::Round(buffers[i][j].getChargeFloat() * decodeBitsFactor) / decodeBitsFactor);
        } else {
          buffers[i][j].setCharge(CAMath::Round(buffers[i][j].getChargeFloat()));
        }
        j++;
      }
    }
    nsb[i] = j;
  }
}

#ifdef GPUCA_O2_LIB
namespace o2::gpu::internal
{
template <class T>
static inline auto GetDecoder_internal(const GPUParam* param, int32_t version)
{
  std::shared_ptr<T> enc = std::make_shared<T>();
  if (param == nullptr) {
    static GPUParam dummyParam;
    param = &dummyParam;
  }
  enc->param = param;
  enc->zsVersion = version;
  enc->init();
  return [enc](std::vector<o2::tpc::Digit>& outBuffer, const void* page, uint32_t firstTfOrbit, uint32_t triggerBC = 0) {
    const o2::header::RAWDataHeader& rdh = *(const o2::header::RAWDataHeader*)page;
    if (o2::raw::RDHUtils::getMemorySize(rdh) == sizeof(o2::header::RAWDataHeader)) {
      return;
    }
    if (o2::raw::RDHUtils::getDetectorField(rdh) != 2) {
      return;
    }
    o2::tpc::CRU cru(o2::tpc::rdh_utils::getCRU(rdh));
    enc->iSector = cru.sector();
    int32_t endpoint = cru.region() * 2 + o2::tpc::rdh_utils::getEndPoint(rdh);
    enc->decodePage(outBuffer, (const zsPage*)page, endpoint, firstTfOrbit, triggerBC);
  };
}
} // namespace o2::gpu::internal

std::function<void(std::vector<o2::tpc::Digit>&, const void*, uint32_t, uint32_t)> GPUReconstructionConvert::GetDecoder(int32_t version, const GPUParam* param)
{
  if (version >= o2::tpc::ZSVersion::ZSVersionRowBased10BitADC && version <= o2::tpc::ZSVersion::ZSVersionRowBased12BitADC) {
    return o2::gpu::internal::GetDecoder_internal<zsEncoderRow>(param, version);
  } else if (version == o2::tpc::ZSVersion::ZSVersionLinkBasedWithMeta) {
    return o2::gpu::internal::GetDecoder_internal<zsEncoderImprovedLinkBased>(param, version);
  } else if (version >= o2::tpc::ZSVersion::ZSVersionDenseLinkBased && version <= o2::tpc::ZSVersion::ZSVersionDenseLinkBasedV2) {
    return o2::gpu::internal::GetDecoder_internal<zsEncoderDenseLinkBased>(param, version);
  } else {
    throw std::runtime_error("Invalid ZS version "s + std::to_string(version) + ", cannot create decoder"s);
  }
}
#endif
