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

#ifndef ALICEO2_SIMULATIONDATAFORMAT_RUNCONTEXT_H
#define ALICEO2_SIMULATIONDATAFORMAT_RUNCONTEXT_H

#include <vector>
#include <TChain.h>
#include <TBranch.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/BunchFilling.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include <GPUCommonLogger.h>
#include <unordered_map>
#include <MathUtils/Cartesian.h>
#include <DataFormatsCalibration/MeanVertexObject.h>
#include <DataFormatsCTP/Digits.h>

namespace o2
{
namespace steer
{
// a structure describing EventPart
// (an elementary constituent of a collision)

constexpr static int QEDSOURCEID = 99;

struct EventPart {
  EventPart() = default;
  EventPart(int s, int e) : sourceID(s), entryID(e) {}
  int sourceID = 0; // the ID of the source (0->backGround; > 1 signal source)
  // the sourceID should correspond to the chain ID
  int entryID = 0; // the event/entry ID inside the chain corresponding to sourceID

  static bool isSignal(EventPart e) { return e.sourceID > 1 && e.sourceID != QEDSOURCEID; }
  static bool isBackGround(EventPart e) { return !isSignal(e); }
  static bool isQED(EventPart e) { return e.sourceID == QEDSOURCEID; }
  ClassDefNV(EventPart, 1);
};

// Class fully describing the Collision context or timeframe structure.
// The context fixes things such as times (orbits and bunch crossings)
// at which collision happen inside a timeframe and how they are composed
// in terms of MC events.
class DigitizationContext
{
 public:
  DigitizationContext() : mNofEntries{0}, mMaxPartNumber{0}, mEventRecords(), mEventParts() {}

  uint32_t getFirstOrbitForSampling() const { return mFirstOrbitForSampling; }
  void setFirstOrbitForSampling(uint32_t o) { mFirstOrbitForSampling = o; }

  int getNCollisions() const { return mNofEntries; }
  void setNCollisions(int n) { mNofEntries = n; }

  void setMaxNumberParts(int maxp) { mMaxPartNumber = maxp; }
  int getMaxNumberParts() const { return mMaxPartNumber; }

  std::vector<o2::InteractionTimeRecord>& getEventRecords(bool withQED = false) { return withQED ? mEventRecordsWithQED : mEventRecords; }
  std::vector<std::vector<o2::steer::EventPart>>& getEventParts(bool withQED = false) { return withQED ? mEventPartsWithQED : mEventParts; }

  const std::vector<o2::InteractionTimeRecord>& getEventRecords(bool withQED = false) const { return withQED ? mEventRecordsWithQED : mEventRecords; }
  const std::vector<std::vector<o2::steer::EventPart>>& getEventParts(bool withQED = false) const { return withQED ? mEventPartsWithQED : mEventParts; }

  // returns a collection of (first) collision indices that have this "source" included
  std::unordered_map<int, int> getCollisionIndicesForSource(int source) const;

  bool isQEDProvided() const { return !mEventRecordsWithQED.empty(); }

  void setBunchFilling(o2::BunchFilling const& bf) { mBCFilling = bf; }
  const o2::BunchFilling& getBunchFilling() const { return (const o2::BunchFilling&)mBCFilling; }

  void setMuPerBC(float m) { mMuBC = m; }
  float getMuPerBC() const { return mMuBC; }

  /// returns the main (hadronic interaction rate) associated to this digitization context
  float getCalculatedInteractionRate() const { return getMuPerBC() * getBunchFilling().getNBunches() * o2::constants::lhc::LHCRevFreq; }

  void printCollisionSummary(bool withQED = false, int truncateOutputTo = -1) const;

  // we need a method to fill the file names
  void setSimPrefixes(std::vector<std::string> const& p);
  std::vector<std::string> const& getSimPrefixes() const { return mSimPrefixes; }
  // returns the source for a given simprefix ... otherwise -1 if not found
  int findSimPrefix(std::string const& prefix) const;

  /// add QED contributions to context, giving prefix; maximal event number and qed interaction rate
  void fillQED(std::string_view QEDprefix, int max_events, double qedrate);

  /// add QED contributions to context; QEDprefix is prefix of QED production
  /// irecord is vector of QED interaction times (sampled externally)
  void fillQED(std::string_view QEDprefix, std::vector<o2::InteractionTimeRecord> const& irecord, int max_events = -1, bool fromKinematics = true);

  /// Common functions the setup input TChains for reading, given the state (prefixes) encapsulated
  /// by this context. The input vector needs to be empty otherwise nothing will be done.
  /// return boolean saying if input simchains was modified or not
  bool initSimChains(o2::detectors::DetID detid, std::vector<TChain*>& simchains) const;

  /// Common functions the setup input TChains for reading kinematics information, given the state (prefixes) encapsulated
  /// by this context. The input vector needs to be empty otherwise nothing will be done.
  /// return boolean saying if input simchains was modified or not
  bool initSimKinematicsChains(std::vector<TChain*>& simkinematicschains) const;

  /// Check collision parts for vertex consistency.
  bool checkVertexCompatibility(bool verbose = false) const;

  /// retrieves collision context for a single timeframe-id (which may be needed by simulation)
  /// (Only copies collision context without QED information. This can be added to the result with the fillQED method
  ///  in a second step. Takes as input a timeframe indices collection)
  DigitizationContext extractSingleTimeframe(int timeframeid, std::vector<std::tuple<int, int, int>> const& timeframeindices, std::vector<int> const& sources_to_offset);

  /// function reading the hits from a chain (previously initialized with initSimChains
  /// The hits pointer will be initialized (what to we do about ownership??)
  template <typename T>
  void retrieveHits(std::vector<TChain*> const& chains,
                    const char* brname,
                    int sourceID,
                    int entryID,
                    std::vector<T>* hits) const;

  /// returns the GRP object associated to this context
  o2::parameters::GRPObject const& getGRP() const;

  // apply collision number cuts and potential relabeling of eventID, (keeps collisions which fall into the orbitsEarly range for the next timeframe)
  // needs a timeframe index structure (determined by calcTimeframeIndices), which is adjusted during the process to reflect the filtering
  void applyMaxCollisionFilter(std::vector<std::tuple<int, int, int>>& timeframeindices, long startOrbit, long orbitsPerTF, int maxColl, double orbitsEarly = 0.);

  /// get timeframe structure --> index markers where timeframe starts/ends/is_influenced_by
  std::vector<std::tuple<int, int, int>> calcTimeframeIndices(long startOrbit, long orbitsPerTF, double orbitsEarly = 0.) const;

  // Sample and fix interaction vertices (according to some distribution). Makes sure that same event ids
  // have to have same vertex, as well as event ids associated to same collision.
  void sampleInteractionVertices(o2::dataformats::MeanVertexObject const& v);

  // helper functions to save and load a context
  void saveToFile(std::string_view filename) const;

  // Return the vector of interaction vertices associated with collisions
  // The vector is empty if no vertices were provided or sampled. In this case, one
  // may call "sampleInteractionVertices".
  std::vector<math_utils::Point3D<float>> const& getInteractionVertices() const { return mInteractionVertices; }

  static DigitizationContext* loadFromFile(std::string_view filename = "");

  void setCTPDigits(std::vector<o2::ctp::CTPDigit> const* ctpdigits) const
  {
    mCTPTrigger = ctpdigits;
    if (mCTPTrigger) {
      mHasTrigger = true;
    }
  }

  void setDigitizerInteractionRate(float intRate) { mDigitizerInteractionRate = intRate; }
  float getDigitizerInteractionRate() const { return mDigitizerInteractionRate; }

  std::vector<o2::ctp::CTPDigit> const* getCTPDigits() const { return mCTPTrigger; }
  bool hasTriggerInput() const { return mHasTrigger; }

 private:
  int mNofEntries = 0;
  int mMaxPartNumber = 0; // max number of parts in any given collision
  uint32_t mFirstOrbitForSampling = 0; // 1st orbit to start sampling

  float mMuBC;            // probability of hadronic interaction per bunch

  std::vector<o2::InteractionTimeRecord> mEventRecords;
  // for each collision we record the constituents (which shall not exceed mMaxPartNumber)
  std::vector<std::vector<o2::steer::EventPart>> mEventParts;

  // for each collisionstd::vector<std::tuple<int,int,int>> &timeframeindice we may record/fix the interaction vertex (to be used in event generation)
  std::vector<math_utils::Point3D<float>> mInteractionVertices;

  // the collision records **with** QED interleaved;
  std::vector<o2::InteractionTimeRecord> mEventRecordsWithQED;
  std::vector<std::vector<o2::steer::EventPart>> mEventPartsWithQED;

  o2::BunchFilling mBCFilling; // pattern of active BCs

  std::vector<std::string> mSimPrefixes;             // identifiers to the hit sim products; the key corresponds to the source ID of event record
  std::string mQEDSimPrefix;                         // prefix for QED production/contribution
  mutable o2::parameters::GRPObject* mGRP = nullptr; //!

  mutable std::vector<o2::ctp::CTPDigit> const* mCTPTrigger = nullptr; // CTP trigger info associated to this digitization context
  mutable bool mHasTrigger = false;                                    //

  // The global ALICE interaction hadronic interaction rate as applied in digitization.
  // It should be consistent with mMuPerBC but it might be easier to handle.
  // The value will be filled/inserted by the digitization workflow so that digiters can access it.
  // There is no guarantee that the value is available elsewhere.
  float mDigitizerInteractionRate{-1};

  ClassDefNV(DigitizationContext, 6);
};

/// function reading the hits from a chain (previously initialized with initSimChains
template <typename T>
inline void DigitizationContext::retrieveHits(std::vector<TChain*> const& chains,
                                              const char* brname,
                                              int sourceID,
                                              int entryID,
                                              std::vector<T>* hits) const
{
  if (chains.size() <= sourceID) {
    return;
  }
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(error) << "No branch found with name " << brname;
    return;
  }
  br->SetAddress(&hits);
  auto maxEntries = br->GetEntries();
  if (maxEntries) {
    entryID %= maxEntries;
  }
  br->GetEntry(entryID);
}

} // namespace steer
} // namespace o2

#endif
