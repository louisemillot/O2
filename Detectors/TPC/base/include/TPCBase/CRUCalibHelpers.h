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

#ifndef AliceO2_TPC_CRUCalibHelpers_H_
#define AliceO2_TPC_CRUCalibHelpers_H_

#include <unordered_map>
#include <string>
#include <array>
#include <map>
#include <cassert>
#include <gsl/span>
#include <filesystem>
#include <type_traits>
#include <vector>
namespace fs = std::filesystem;

#include "Rtypes.h"
#include "TFile.h"

#include "TPCBase/CalDet.h"
#include "TPCBase/Utils.h"

namespace o2::tpc::cru_calib_helpers
{

struct LinkInfo {
  LinkInfo(int cru, int link) : cru(cru), globalLinkID(link) {}
  int cru{0};
  int globalLinkID{0};

  bool operator<(const LinkInfo& other) const
  {
    if (cru < other.cru) {
      return true;
    }
    if ((cru == other.cru) && (globalLinkID < other.globalLinkID)) {
      return true;
    }
    return false;
  }

  ClassDefNV(LinkInfo, 0);
};

using ValueArrayU32 = std::array<uint32_t, 80>;
using DataMapU32 = std::map<LinkInfo, ValueArrayU32>;

using ValueArrayF = std::array<float, 80>;
using DataMapF = std::map<LinkInfo, ValueArrayF>;

void debugDiff(std::string_view file1, std::string_view file2, std::string_view objName);
void testChannelMapping(int cruID = 0);

/// return the hardware channel number as mapped in the CRU
int getHWChannel(int sampa, int channel, int regionIter);

/// convert HW mapping to sampa and channel number
std::tuple<int, int> getSampaInfo(int hwChannel, int cruID);

/// convert float to integer with fixed precision and max number of digits
template <uint32_t DataBitSizeT = 12, uint32_t SignificantBitsT = 2>
constexpr uint32_t floatToFixedSize(float value)
{
  constexpr uint32_t DataBitSize = DataBitSizeT;                       ///< number of bits of the data representation
  constexpr uint32_t SignificantBits = SignificantBitsT;               ///< number of bits used for floating point precision
  constexpr uint64_t BitMask = ((uint64_t(1) << DataBitSize) - 1);     ///< mask for bits
  constexpr float FloatConversion = 1.f / float(1 << SignificantBits); ///< conversion factor from integer representation to float

  const auto adc = uint32_t((value + 0.5f * FloatConversion) / FloatConversion) & BitMask;
  assert(std::abs(value - adc * FloatConversion) <= 0.5f * FloatConversion);

  return adc;
}

template <uint32_t SignificantBitsT = 2>
constexpr float fixedSizeToFloat(uint32_t value)
{
  constexpr uint32_t SignificantBits = SignificantBitsT;               ///< number of bits used for floating point precision
  constexpr float FloatConversion = 1.f / float(1 << SignificantBits); ///< conversion factor from integer representation to float

  return float(value) * FloatConversion;
}

/// write values of map to fileName
///
template <typename DataMap>
void writeValues(std::ostream& str, const DataMap& map, bool onlyFilled = false)
{
  for (const auto& [linkInfo, data] : map) {
    if (onlyFilled) {
      if (!std::accumulate(data.begin(), data.end(), uint32_t(0))) {
        continue;
      }
    }
    std::string values;
    for (const auto& val : data) {
      if (values.size()) {
        values += ",";
      }
      values += fmt::format("{}", val);
    }

    str << linkInfo.cru << " "
        << linkInfo.globalLinkID << " "
        << values << "\n";
  }
}

template <typename DataMap>
void writeValues(const std::string_view fileName, const DataMap& map, bool onlyFilled = false)
{
  std::ofstream str(fileName.data(), std::ofstream::out);
  writeValues(str, map, onlyFilled);
}

template <class T>
struct is_map {
  static constexpr bool value = false;
};

template <class Key, class Value>
struct is_map<std::map<Key, Value>> {
  static constexpr bool value = true;
};

/// fill cal pad object from HW data map
/// TODO: Function to be tested
template <typename DataMap, uint32_t SignificantBitsT = 0>
typename std::enable_if_t<is_map<DataMap>::value, void>
  fillCalPad(CalDet<float>& calPad, const DataMap& map)
{
  using namespace o2::tpc;
  const auto& mapper = Mapper::instance();

  for (const auto& [linkInfo, data] : map) {
    const CRU cru(linkInfo.cru);
    const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
    const int nFECs = partInfo.getNumberOfFECs();
    const int fecOffset = (nFECs + 1) / 2;
    const int fecInPartition = (linkInfo.globalLinkID < fecOffset) ? linkInfo.globalLinkID : fecOffset + linkInfo.globalLinkID % 12;

    int hwChannel{0};
    for (const auto& val : data) {
      const auto& [sampaOnFEC, channelOnSAMPA] = getSampaInfo(hwChannel, cru);
      const PadROCPos padROCPos = mapper.padROCPos(cru, fecInPartition, sampaOnFEC, channelOnSAMPA);
      if constexpr (SignificantBitsT == 0) {
        const float set = std::stof(val);
        calPad.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), set);
      } else {
        const float set = fixedSizeToFloat<SignificantBitsT>(uint32_t(std::stoi(val)));
        calPad.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), set);
      }
      ++hwChannel;
    }
  }
}

/// fill cal pad object from HW value stream
template <uint32_t SignificantBitsT = 2>
int fillCalPad(CalDet<float>& calPad, std::istream& infile)
{
  using namespace o2::tpc;
  const auto& mapper = Mapper::instance();

  int cruID{0};
  int globalLinkID{0};
  int sampaOnFEC{0};
  int channelOnSAMPA{0};
  std::string values;
  int nLines{0};

  std::string line;
  while (std::getline(infile, line)) {
    ++nLines;
    std::stringstream streamLine(line);
    streamLine >> cruID >> globalLinkID >> values;

    const CRU cru(cruID);
    const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
    const int nFECs = partInfo.getNumberOfFECs();
    const int fecOffset = (nFECs + 1) / 2;
    const int fecInPartition = (globalLinkID < fecOffset) ? globalLinkID : fecOffset + globalLinkID % 12;

    int hwChannel{0};
    for (const auto& val : utils::tokenize(values, ",")) {
      const auto& [sampaOnFEC, channelOnSAMPA] = getSampaInfo(hwChannel, cru);
      const PadROCPos padROCPos = mapper.padROCPos(cru, fecInPartition, sampaOnFEC, channelOnSAMPA);
      if constexpr (SignificantBitsT == 0) {
        const float set = std::stof(val);
        calPad.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), set);
      } else {
        const float set = fixedSizeToFloat<SignificantBitsT>(uint32_t(std::stoi(val)));
        calPad.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), set);
      }
      ++hwChannel;
    }
  }

  return nLines;
}

/// fill cal pad object from HW value buffer
template <uint32_t SignificantBitsT = 2>
int fillCalPad(CalDet<float>& calPad, gsl::span<const char> data)
{
  struct membuf : std::streambuf {
    membuf(char* base, std::ptrdiff_t n)
    {
      this->setg(base, base, base + n);
    }
  };
  membuf sbuf((char*)data.data(), data.size());
  std::istream in(&sbuf);

  return fillCalPad<SignificantBitsT>(calPad, in);
}

/// create cal pad object from HW value file
///
/// if outputFile is set, write the object to file
/// if calPadName is set use it for the object name in the file. Otherwise the basename of the fileName is used
template <uint32_t SignificantBitsT = 2>
o2::tpc::CalDet<float> getCalPad(const std::string_view fileName, const std::string_view outputFile = "", std::string_view calPadName = "")
{
  using namespace o2::tpc;

  if (!calPadName.size()) {
    calPadName = fs::path(fileName.data()).stem().c_str();
  }
  CalDet<float> calPad(calPadName);

  std::ifstream infile(fileName.data(), std::ifstream::in);
  if (!infile.is_open()) {
    LOGP(error, "could not open file {}", fileName);
    return calPad;
  }

  fillCalPad<SignificantBitsT>(calPad, infile);

  if (outputFile.size()) {
    TFile f(outputFile.data(), "recreate");
    f.WriteObject(&calPad, calPadName.data());
  }
  return calPad;
}

/// \param sigmaNoiseROCType can be either one value for all ROC types, or {IROC, OROC}, or {IROC, OROC1, OROC2, OROC3}
/// \param minADCROCType can be either one value for all ROC types, or {IROC, OROC}, or {IROC, OROC1, OROC2, OROC3}
std::unordered_map<std::string, CalPad> preparePedestalFiles(const CalPad& pedestals, const CalPad& noise, std::vector<float> sigmaNoiseROCType = {3, 3, 3, 3}, std::vector<float> minADCROCType = {2, 2, 2, 2}, float pedestalOffset = 0, bool onlyFilled = false, bool maskBad = true, float noisyChannelThreshold = 1.5, float sigmaNoiseNoisyChannels = 4, float badChannelThreshold = 6, bool fixedSize = false);

DataMapU32 getDataMap(const CalPad& calPad);
} // namespace o2::tpc::cru_calib_helpers

#endif
