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

/// \file RawDataDecoder.cxx
/// \author Roman Lietava
#include <fstream>
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "CTPReconstruction/RawDataDecoder.h"
#include "DataFormatsCTP/Configuration.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include <boost/range/adaptor/reversed.hpp>

using namespace o2::ctp;

// Inverse of Digits2Raw::makeGBTWord
void RawDataDecoder::makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld)
{
  gbtword80_t diglet = remnant;
  uint32_t i = 0;
  while (i < (NGBT - Npld)) {
    std::bitset<NGBT> masksize = 0;
    for (uint32_t j = 0; j < (Npld - size_gbt); j++) {
      masksize[j] = 1;
    }
    diglet |= (GBTWord & masksize) << (size_gbt);
    diglets.push_back(diglet);
    diglet = 0;
    i += Npld - size_gbt;
    GBTWord = GBTWord >> (Npld - size_gbt);
    size_gbt = 0;
  }
  size_gbt = NGBT - i;
  remnant = GBTWord;
}
int RawDataDecoder::addCTPDigit(uint32_t linkCRU, uint32_t orbit, gbtword80_t& diglet, gbtword80_t& pldmask, std::map<o2::InteractionRecord, CTPDigit>& digits)
{
  int ret = 0;
  gbtword80_t pld = (diglet & pldmask);
  if (pld.count() == 0) {
    return 0;
  }
  pld >>= 12;
  CTPDigit digit;
  const gbtword80_t bcidmask = 0xfff;
  uint16_t bcid = (diglet & bcidmask).to_ulong();
  LOG(debug) << bcid << "    pld:" << pld;
  o2::InteractionRecord ir = {bcid, orbit};
  if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
    int32_t BCShiftCorrectionInps = -o2::ctp::TriggerOffsetsParam::Instance().globalInputsShift;
    LOG(debug) << "InputMaskCount:" << digits[ir].CTPInputMask.count();
    LOG(debug) << "ir ir ori:" << ir;
    if ((ir.orbit <= mTFOrbit) && ((int32_t)ir.bc < BCShiftCorrectionInps)) {
      // LOG(warning) << "Loosing ir:" << ir;
      mIRRejected++;
      return 0;
    }
    ir -= BCShiftCorrectionInps;
    LOG(debug) << "ir ir corrected:" << ir;
    digit.intRecord = ir;
    if (digits.count(ir) == 0) {
      digit.setInputMask(pld);
      digits[ir] = digit;
      LOG(debug) << bcid << " inputs case 0 bcid orbit " << orbit << " pld:" << pld;
    } else if (digits.count(ir) == 1) {
      if (digits[ir].CTPInputMask.count() == 0) {
        digits[ir].setInputMask(pld);
        LOG(debug) << bcid << " inputs bcid case 1 orbit " << orbit << " pld:" << pld;
      } else {
        if (mErrorIR < mErrorMax) {
          LOG(error) << "Two CTP IRs with the same timestamp:" << ir.bc << " " << ir.orbit << " pld:" << pld << " dig:" << digits[ir];
        }
        ret = 4;
        mErrorIR++;
        mStickyError = true;
      }
    } else {
      LOG(error) << "Two digits with the same timestamp:" << ir.bc << " " << ir.orbit;
      ret = 8;
    }
  } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
    int32_t BCShiftCorrection = -o2::ctp::TriggerOffsetsParam::Instance().customOffset[o2::detectors::DetID::CTP];
    int32_t offset = BCShiftCorrection + o2::ctp::TriggerOffsetsParam::Instance().LM_L0 + o2::ctp::TriggerOffsetsParam::Instance().L0_L1_classes - 1;
    LOG(debug) << "tcr ir ori:" << ir;
    if ((ir.orbit <= mTFOrbit) && ((int32_t)ir.bc < offset)) {
      // LOG(warning) << "Loosing tclass:" << ir;
      mTCRRejected++;
      return 0;
    }
    ir -= offset;
    LOG(debug) << "tcr ir corrected:" << ir;
    digit.intRecord = ir;
    if (digits.count(ir) == 0) {
      digit.setClassMask(pld);
      digits[ir] = digit;
      LOG(debug) << bcid << " class bcid case 0 orbit " << orbit << " pld:" << pld;
    } else if (digits.count(ir) == 1) {
      if (digits[ir].CTPClassMask.count() == 0) {
        digits[ir].setClassMask(pld);
        LOG(debug) << bcid << " class bcid case 1 orbit " << orbit << " pld:" << pld;
      } else {
        if (mErrorTCR < mErrorMax) {
          LOG(error) << "Two CTP Class masks for same timestamp";
          mStickyError = true;
        }
        mErrorTCR++;
        ret = 16;
      }
    } else {
      LOG(error) << "Two digits with the same timestamp:" << ir.bc << " " << ir.orbit;
      ret = 32;
    }
  } else {
    LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
  }
  return ret;
}
//
// Decodes one page
// It is NOT assumed that CTP HBF has never more than one page.
// 1 HBF/page <= 8000kB = 8*1024*8/120 = 546 GBT words = 546 IRs/page = 5.5 MHz
int RawDataDecoder::decodeRaw(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec>& filter, o2::pmr::vector<CTPDigit>& digits, std::vector<LumiInfo>& lumiPointsHBF1)
{
  int ret = 0;
  static int nwrites = 0;
  uint64_t countsMBT = 0;
  uint64_t countsMBV = 0;
  std::map<o2::InteractionRecord, CTPDigit> digitsMap;
  //
  // using InputSpec = o2::framework::InputSpec;
  // using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  // using Lifetime = o2::framework::Lifetime;
  o2::framework::DPLRawParser parser(inputs, filter);
  uint32_t payloadCTP = 0;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  mTFOrbit = 0;
  uint32_t orbit0 = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    const o2::header::RDHAny* rdh = nullptr;
    try {
      rdh = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      mPadding = (o2::raw::RDHUtils::getDataFormat(rdh) == 0);
    } catch (std::exception& e) {
      LOG(error) << "Failed to extract RDH, abandoning TF sending dummy output, exception was: " << e.what();
      // dummyOutput();
      return 1;
    }
    // auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    uint32_t stopBit = o2::raw::RDHUtils::getStop(rdh);
    uint32_t packetCounter = o2::raw::RDHUtils::getPageCounter(rdh);
    uint32_t version = o2::raw::RDHUtils::getVersion(rdh);
    uint32_t rdhOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdh);
    uint32_t triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
    // std::cout << "diff orbits:" << triggerOrbit - rdhOrbit << std::endl;
    bool tf = (triggerType & TF_TRIGGERTYPE_MASK) && (packetCounter == 0);
    bool hb = (triggerType & HB_TRIGGERTYPE_MASK) && (packetCounter == 0);
    if (tf) {
      mTFOrbit = rdhOrbit;
      // std::cout << "tforbit==================>" << mTFOrbit << " " << std::hex << mTFOrbit << std::endl;
      mTFOrbits.push_back(mTFOrbit);
    }
    static bool prt = true;
    if (prt) {
      LOG(info) << "RDH version:" << version << " Padding:" << mPadding;
      prt = false;
    }
    auto feeID = o2::raw::RDHUtils::getFEEID(rdh); // 0 = IR, 1 = TCR
    auto linkCRU = (feeID & 0xf00) >> 8;
    // LOG(info) << "CRU link:" << linkCRU;
    if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
      payloadCTP = o2::ctp::NClassPayload;
      if (!mDoDigits) { // Do not do TCR if only lumi
        continue;
      }
    } else {
      LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
    }
    LOG(debug) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << rdhOrbit << " triggerType:" << triggerType;
    // LOG(info) << "remnant :" << remnant.count();
    gbtword80_t pldmask = 0;
    for (uint32_t i = 0; i < payloadCTP; i++) {
      pldmask[12 + i] = 1;
    }
    // std::cout << (orbit0 != rdhOrbit) << " comp " << (mTFOrbit==rdhOrbit) << std::endl;
    // if(orbit0 != rdhOrbit) {
    if (hb) {
      if ((mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) && !tf) { // create lumi per HB
        lumiPointsHBF1.emplace_back(LumiInfo{rdhOrbit, 0, 0, countsMBT, countsMBV});
        // std::cout << "hb:" << nhb << " countsMBT:" << countsMBT << std::endl;
        countsMBT = 0;
        countsMBV = 0;
        // nhb++;
      }
      remnant = 0;
      size_gbt = 0;
      orbit0 = rdhOrbit;
      // std::cout << "orbit0============>" << std::dec << orbit0 << " " << std::hex << orbit0 << std::endl;
    }
    // Create 80 bit words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord80;
    gbtWord80.set();
    int wordCount = 0;
    int wordSize = 10;
    std::vector<gbtword80_t> gbtwords80;
    // mPadding = 0;
    if (mPadding == 1) {
      wordSize = 16;
    }
    // LOG(info) << ii << " payload size:" << payload.size();
    /* if (payload.size()) {
      //LOG(info) << "payload size:" << payload.size();
      // LOG(info) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << triggerOrbit << " stopbit:" << stopBit << " packet:" << packetCounter;
      // LOGP(info, "RDH FEEid: {} CRU link: {}, Orbit: {}", feeID, linkCRU, triggerOrbit);
      std::cout << std::hex << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << rdhOrbit << std::endl;
    } */
    gbtword80_t bcmask = std::bitset<80>("111111111111");
    for (auto payloadWord : payload) {
      int wc = wordCount % wordSize;
      // LOG(info) << wordCount << ":" << wc << " payload:" << int(payloadWord);
      if ((wc == 0) && (wordCount != 0)) {
        if (gbtWord80.count() != 80) {
          gbtwords80.push_back(gbtWord80);
        }
        gbtWord80.set();
      }
      if (wc < 10) {
        for (int i = 0; i < 8; i++) {
          gbtWord80[wc * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
      }
      wordCount++;
    }
    if ((gbtWord80.count() != 80) && (gbtWord80.count() > 0)) {
      gbtwords80.push_back(gbtWord80);
    }
    // decode 80 bits payload
    for (auto word : gbtwords80) {
      std::vector<gbtword80_t> diglets;
      gbtword80_t gbtWord = word;
      makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
      for (auto diglet : diglets) {
        if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) {
          gbtword80_t pld = (diglet >> 12) & mTVXMask;
          if (pld.count() != 0) {
            countsMBT++;
          }
          pld = (diglet >> 12) & mVBAMask;
          if (pld.count() != 0) {
            countsMBV++;
          }
        }
        if (!mDoDigits) {
          continue;
        }
        LOG(debug) << "diglet:" << diglet << " " << (diglet & bcmask).to_ullong();
        ret = addCTPDigit(linkCRU, rdhOrbit, diglet, pldmask, digitsMap);
      }
    }
    if ((remnant.count() > 0) && stopBit) {
      if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) {
        gbtword80_t pld = (remnant >> 12) & mTVXMask;
        if (pld.count() != 0) {
          countsMBT++;
        }
        pld = (remnant >> 12) & mVBAMask;
        if (pld.count() != 0) {
          countsMBV++;
        }
      }
      if (!mDoDigits) {
        continue;
      }
      ret = addCTPDigit(linkCRU, rdhOrbit, remnant, pldmask, digitsMap);
      LOG(debug) << "diglet:" << remnant << " " << (remnant & bcmask).to_ullong();
      remnant = 0;
    }
  }
  if (mDoLumi) {
    lumiPointsHBF1.emplace_back(LumiInfo{orbit0, 0, 0, countsMBT, countsMBV});
    // std::cout << "last lumi:" << nhb  << std::endl;
  }
  if (mDoDigits & mDecodeInps) {
    uint64_t trgclassmask = 0xffffffffffffffff;
    if (mCTPConfig.getRunNumber() != 0) {
      trgclassmask = mCTPConfig.getTriggerClassMask();
    }
    // std::cout << "trgclassmask:" << std::hex << trgclassmask << std::dec << std::endl;
    ret = shiftInputs(digitsMap, digits, mTFOrbit, trgclassmask);
    if (mCheckConsistency) {
      ret = checkReadoutConsistentncy(digits, trgclassmask);
    }
  }
  if (mDoDigits && !mDecodeInps) {
    for (auto const& dig : digitsMap) {
      digits.push_back(dig.second);
    }
  }
  // ret = 1;
  if (mStickyError) {
    if (nwrites < mErrorMax) {
      std::string file = "dumpCTP" + std::to_string(nwrites) + ".bin";
      std::ofstream dumpctp(file.c_str(), std::ios::out | std::ios::binary);
      if (!dumpctp.good()) {
        LOGP(error, "Failed to open file {}", file);
      } else {
        LOGP(info, "CTP dump file open {}", file);
        for (auto it = parser.begin(); it != parser.end(); ++it) {
          char* dataout = (char*)(it.raw());
          dumpctp.write(dataout, it.sizeTotal());
        }
        dumpctp.close();
      }
      nwrites++;
    }
    // LOG(error) << "CTP decoding IR errors:" << mErrorIR << " TCR errors:" << mErrorTCR;
  }
  return ret;
}
//
int RawDataDecoder::decodeRawFatal(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec>& filter)
{
  o2::framework::DPLRawParser parser(inputs, filter);
  uint32_t payloadCTP = 0;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  mTFOrbit = 0;
  uint32_t orbit0 = 0;
  std::array<int, o2::ctp::CTP_NCLASSES> rates{};
  std::array<int, o2::ctp::CTP_NCLASSES> ratesC{};
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    const o2::header::RDHAny* rdh = nullptr;
    try {
      rdh = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      mPadding = (o2::raw::RDHUtils::getDataFormat(rdh) == 0);
    } catch (std::exception& e) {
      LOG(error) << "Failed to extract RDH, abandoning TF sending dummy output, exception was: " << e.what();
      // dummyOutput();
      return 1;
    }
    // auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    uint32_t stopBit = o2::raw::RDHUtils::getStop(rdh);
    uint32_t packetCounter = o2::raw::RDHUtils::getPageCounter(rdh);
    uint32_t version = o2::raw::RDHUtils::getVersion(rdh);
    uint32_t rdhOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdh);
    uint32_t triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
    // std::cout << "diff orbits:" << triggerOrbit - rdhOrbit << std::endl;
    bool tf = (triggerType & TF_TRIGGERTYPE_MASK) && (packetCounter == 0);
    bool hb = (triggerType & HB_TRIGGERTYPE_MASK) && (packetCounter == 0);
    if (tf) {
      mTFOrbit = rdhOrbit;
      // std::cout << "tforbit==================>" << mTFOrbit << " " << std::hex << mTFOrbit << std::endl;
      mTFOrbits.push_back(mTFOrbit);
    }
    static bool prt = true;
    if (prt) {
      LOG(info) << "RDH version:" << version << " Padding:" << mPadding;
      prt = false;
    }
    auto feeID = o2::raw::RDHUtils::getFEEID(rdh); // 0 = IR, 1 = TCR
    auto linkCRU = (feeID & 0xf00) >> 8;
    // LOG(info) << "CRU link:" << linkCRU;
    if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
      payloadCTP = o2::ctp::NClassPayload;
    } else {
      LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
    }
    LOG(debug) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << rdhOrbit << " triggerType:" << triggerType;
    // LOG(info) << "remnant :" << remnant.count();
    gbtword80_t pldmask = 0;
    for (uint32_t i = 0; i < payloadCTP; i++) {
      pldmask[12 + i] = 1;
    }
    // std::cout << (orbit0 != rdhOrbit) << " comp " << (mTFOrbit==rdhOrbit) << std::endl;
    // if(orbit0 != rdhOrbit) {
    if (hb) {
      remnant = 0;
      size_gbt = 0;
      orbit0 = rdhOrbit;
      // std::cout << "orbit0============>" << std::dec << orbit0 << " " << std::hex << orbit0 << std::endl;
    }
    // Create 80 bit words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord80;
    gbtWord80.set();
    int wordCount = 0;
    int wordSize = 10;
    std::vector<gbtword80_t> gbtwords80;
    // mPadding = 0;
    if (mPadding == 1) {
      wordSize = 16;
    }
    // LOG(info) << ii << " payload size:" << payload.size();
    gbtword80_t bcmask = std::bitset<80>("111111111111");
    for (auto payloadWord : payload) {
      int wc = wordCount % wordSize;
      // LOG(info) << wordCount << ":" << wc << " payload:" << int(payloadWord);
      if ((wc == 0) && (wordCount != 0)) {
        if (gbtWord80.count() != 80) {
          gbtwords80.push_back(gbtWord80);
        }
        gbtWord80.set();
      }
      if (wc < 10) {
        for (int i = 0; i < 8; i++) {
          gbtWord80[wc * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
      }
      wordCount++;
    }
    if ((gbtWord80.count() != 80) && (gbtWord80.count() > 0)) {
      gbtwords80.push_back(gbtWord80);
    }
    // decode 80 bits payload
    for (auto word : gbtwords80) {
      std::vector<gbtword80_t> diglets;
      gbtword80_t gbtWord = word;
      makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
      for (auto diglet : diglets) {
        int nbits = payloadCTP - 12;
        for (int i = 0; i < nbits; i++) {
          gbtword80_t mask = 1ull << i;
          gbtword80_t pld = (diglet >> 12) & mask;
          // LOG(info) << "diglet:" << diglet << " pld:" << pld;
          if (pld.count() != 0) {
            if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
              rates[i]++;
            } else {
              ratesC[i]++;
            }
          }
        }
        // LOG(debug) << "diglet:" << diglet << " " << (diglet & bcmask).to_ullong();
      }
    }
    if ((remnant.count() > 0) && stopBit) {
      int nbits = payloadCTP - 12;
      for (int i = 0; i < nbits; i++) {
        gbtword80_t mask = 1ull << i;
        gbtword80_t pld = (remnant >> 12) & mask;
        // LOG(info) << "diglet:" << remnant << " pld:" << pld;
        if (pld.count() != 0) {
          if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
            rates[i]++;
          } else {
            ratesC[i]++;
          }
        }
      }
      remnant = 0;
    }
  }
  // print max rates
  std::map<int, int> ratesmap;
  std::map<int, int> ratesmapC;
  for (int i = 0; i < o2::ctp::CTP_NCLASSES; i++) {
    if (rates[i]) {
      ratesmap[rates[i]] = i;
    }
    if (ratesC[i]) {
      ratesmapC[ratesC[i]] = i;
    }
  }
  auto nhb = o2::base::GRPGeomHelper::getNHBFPerTF();
  std::string message = "Ringing inputs [MHz]:";
  for (auto const& r : boost::adaptors::reverse(ratesmap)) {
    // LOG(error) << r.second;
    message += " " + o2::ctp::CTPInputsConfiguration::getInputNameFromIndex(r.second + 1) + ":" + std::to_string(r.first / 32. / o2::constants::lhc::LHCOrbitMUS);
  }
  std::string messageC = "Ringing classes [MHz]:";
  for (auto const& r : boost::adaptors::reverse(ratesmapC)) {
    messageC += " class" + std::to_string(r.second) + ":" + std::to_string(r.first / 32. / o2::constants::lhc::LHCOrbitMUS);
  }
  LOG(error) << messageC;
  LOG(fatal) << message;
  return 0;
}
//
int RawDataDecoder::decodeRaw(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec>& filter, std::vector<CTPDigit>& digits, std::vector<LumiInfo>& lumiPointsHBF1)
{
  o2::pmr::vector<CTPDigit> pmrdigits;
  int ret = decodeRaw(inputs, filter, pmrdigits, lumiPointsHBF1);
  for (auto const d : pmrdigits) {
    digits.push_back(d);
  }
  return ret;
}
//
// Not to be called with level LM
// Keeping shift in params if needed to be generalised
int RawDataDecoder::shiftNew(const o2::InteractionRecord& irin, uint32_t TFOrbit, std::bitset<48>& inpmask, int64_t shift, int level, std::map<o2::InteractionRecord, CTPDigit>& digmap)
{
  //
  if (irin.orbit > TFOrbit || irin.bc >= shift) {
    auto lxmask = L0MASKInputs;
    if (level == 1) {
      lxmask = L1MASKInputs;
    }
    auto ir = irin - shift; // add L0 to prev digit
    if (digmap.count(ir)) {
      if ((digmap[ir].CTPInputMask & lxmask).count()) {
        LOG(error) << " Overwriting LX ? X:" << level;
      }
      digmap[ir].CTPInputMask = digmap[ir].CTPInputMask | (inpmask & lxmask);
    } else {
      CTPDigit digit = {ir, inpmask & lxmask, 0};
      digmap[ir] = digit;
    }
  } else {
    LOG(info) << "LOST:" << irin << " shift:" << shift;
  }
  return 0;
}
//

int RawDataDecoder::shiftInputs(std::map<o2::InteractionRecord, CTPDigit>& digitsMap, o2::pmr::vector<CTPDigit>& digits, uint32_t TFOrbit, uint64_t trgclassmask)
{
  // int nClasswoInp = 0; // counting classes without input which should never happen
  int nLM = 0;
  int nL0 = 0;
  int nL1 = 0;
  int nTwI = 0;
  int nTwoI = 0;
  int nTwoIlost = 0;
  std::map<o2::InteractionRecord, CTPDigit> digitsMapShifted;
  auto L0shift = o2::ctp::TriggerOffsetsParam::Instance().LM_L0;
  auto L1shift = L0shift + o2::ctp::TriggerOffsetsParam::Instance().L0_L1;
  for (auto const& dig : digitsMap) {
    auto inpmask = dig.second.CTPInputMask;
    auto inpmaskLM = inpmask & LMMASKInputs;
    auto inpmaskL0 = inpmask & L0MASKInputs;
    auto inpmaskL1 = inpmask & L1MASKInputs;
    int lm = inpmaskLM.count() > 0;
    int l0 = inpmaskL0.count() > 0;
    int l1 = inpmaskL1.count() > 0;
    int lut = lm + (l0 << 1) + (l1 << 2);
    // std::cout << "L0mask:" << L0MASKInputs << std::endl;
    // std::cout << "L0:" << inpmaskL0 << std::endl;
    // std::cout << "L1:" << inpmaskL1 << std::endl;
    if (lut == 0 || lut == 1) { // no inps or LM
      digitsMapShifted[dig.first] = dig.second;
    } else if (lut == 2) { // L0
      shiftNew(dig.first, TFOrbit, inpmask, L0shift, 0, digitsMapShifted);
      if (dig.second.CTPClassMask.count()) {
        // LOG(error) << "Adding class mask without input ?";
        //  This is not needed as it can happen; Full checj done below - see next LOG(error)
        CTPDigit digi = {dig.first, 0, dig.second.CTPClassMask};
        digitsMapShifted[dig.first] = digi;
      }
    } else if (lut == 4) { // L1
      shiftNew(dig.first, TFOrbit, inpmask, L1shift, 1, digitsMapShifted);
      if (dig.second.CTPClassMask.count()) {
        CTPDigit digi = {dig.first, 0, dig.second.CTPClassMask};
        digitsMapShifted[dig.first] = digi;
      }
    } else if (lut == 6) { // L0 and L1
      shiftNew(dig.first, TFOrbit, inpmask, L0shift, 0, digitsMapShifted);
      shiftNew(dig.first, TFOrbit, inpmask, L1shift, 1, digitsMapShifted);
      if (dig.second.CTPClassMask.count()) {
        CTPDigit digi = {dig.first, 0, dig.second.CTPClassMask};
        digitsMapShifted[dig.first] = digi;
      }
    } else if (lut == 3) { // LM and L0
      shiftNew(dig.first, TFOrbit, inpmask, L0shift, 0, digitsMapShifted);
      CTPDigit digi = {dig.first, inpmask & (~L0MASKInputs), dig.second.CTPClassMask};
      // if LM level do not need to add class as LM is not shifted;
      digitsMapShifted[dig.first] = digi;
    } else if (lut == 5) { // LM and L1
      shiftNew(dig.first, TFOrbit, inpmask, L1shift, 1, digitsMapShifted);
      CTPDigit digi = {dig.first, inpmask & (~L1MASKInputs), dig.second.CTPClassMask};
      digitsMapShifted[dig.first] = digi;
    } else if (lut == 7) { // LM and L0 and L1
      shiftNew(dig.first, TFOrbit, inpmask, L0shift, 0, digitsMapShifted);
      shiftNew(dig.first, TFOrbit, inpmask, L1shift, 1, digitsMapShifted);
      CTPDigit digi = {dig.first, inpmaskLM, dig.second.CTPClassMask};
      digitsMapShifted[dig.first] = digi;
    } else {
      LOG(fatal) << "lut = " << lut;
    }
  }
  for (auto const& dig : digitsMapShifted) {
    auto d = dig.second;
    if ((d.CTPInputMask & LMMASKInputs).count()) {
      nLM++;
    }
    if ((d.CTPInputMask & L0MASKInputs).count()) {
      nL0++;
    }
    if ((d.CTPInputMask & L1MASKInputs).count()) {
      nL1++;
    }
    if ((d.CTPClassMask).to_ulong() & trgclassmask) {
      if (d.CTPInputMask.count()) {
        nTwI++;
      } else {
        if (d.intRecord.bc == (o2::constants::lhc::LHCMaxBunches - L1shift)) { // input can be lost because latency class-l1input = 1
          nTwoIlost++;
        } else {
          // LOG(error) << d.intRecord << " " << d.CTPClassMask << " " << d.CTPInputMask;
          // std::cout << "ERROR:" << std::hex << d.CTPClassMask << " " << d.CTPInputMask << std::dec << std::endl;
          nTwoI++;
        }
      }
    }
    digits.push_back(dig.second);
  }
  int ret = 0;
  if (nTwoI) { // Trigger class wo Input
    LOG(error) << "LM:" << nLM << " L0:" << nL0 << " L1:" << nL1 << " TwI:" << nTwI << " Trigger classes wo input:" << nTwoI;
    ret = 64;
  }
  if (nTwoIlost) {
    LOG(warn) << " Trigger classes wo input from diff latency 1:" << nTwoIlost;
  }
  return ret;
}
//
int RawDataDecoder::checkReadoutConsistentncy(o2::pmr::vector<CTPDigit>& digits, uint64_t trgclassmask)
{
  int ret = 0;
  int lost = 0;
  for (auto const& digit : digits) {
    // if class mask => inps
    for (int i = 0; i < digit.CTPClassMask.size(); i++) {
      if (digit.CTPClassMask[i]) {
        const CTPClass* cls = mCTPConfig.getCTPClassFromHWIndex(i);
        uint64_t clsinpmask = cls->descriptor->getInputsMask();
        uint64_t diginpmask = digit.CTPInputMask.to_ullong();
        if (!((clsinpmask & diginpmask) == clsinpmask)) {
          LOG(error) << "CTP class:" << cls->name << " inpmask:" << clsinpmask << " not compatible with inputs mask:" << diginpmask;
          ret = 128;
        }
      }
    }
    // if inps => class mask
    for (auto const& cls : mCTPConfig.getCTPClasses()) {
      uint64_t clsinpmask = cls.descriptor->getInputsMask();
      uint64_t diginpmask = digit.CTPInputMask.to_ullong();
      uint64_t digclsmask = digit.CTPClassMask.to_ullong();
      if ((clsinpmask & diginpmask) == clsinpmask) {
        if ((cls.classMask & digclsmask) == 0) {
          int32_t BCShiftCorrection = -o2::ctp::TriggerOffsetsParam::Instance().customOffset[o2::detectors::DetID::CTP];
          int32_t offset = BCShiftCorrection + o2::ctp::TriggerOffsetsParam::Instance().LM_L0 + o2::ctp::TriggerOffsetsParam::Instance().L0_L1_classes - 1;
          offset = o2::constants::lhc::LHCMaxBunches - offset;
          if (digit.intRecord.bc < offset) {
            LOG(error) << "CTP class:" << cls.name << " inpmask:" << clsinpmask << " cls mask:" << cls.classMask << " not found in digit:" << digit;
            ret = 256;
          } else {
            lost++;
          }
        }
      }
    }
  }
  if (lost) {
    LOG(info) << "LOST classes because of shift:" << lost;
  }
  return ret;
}
//
int RawDataDecoder::setLumiInp(int lumiinp, std::string inp)
{
  // check if valid input
  int index = o2::ctp::CTPInputsConfiguration::getInputIndexFromName(inp);
  if (index == 0xff) {
    LOG(fatal) << "CTP raw decoder: input index not found:" << inp;
    return 0xff;
  }
  if (lumiinp == 1) {
    mTVXMask.reset();
    mTVXMask[index - 1] = true;
  } else {
    mVBAMask.reset();
    mVBAMask[index - 1] = true;
  }
  return index;
}
//
int RawDataDecoder::init()
{
  return 0;
}
