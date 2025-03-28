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

/// @file   AODProducerWorkflowSpec.h

#ifndef O2_AODPRODUCER_WORKFLOW_SPEC
#define O2_AODPRODUCER_WORKFLOW_SPEC

#include "AODMcProducerHelpers.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "Steer/MCKinematicsReader.h"
#include "TStopwatch.h"
#include "ZDCBase/Constants.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonUtils/EnumFlags.h"

#include <cstdint>
#include <limits>
#include <set>
#include <vector>
#include <random>
using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2::aodproducer
{
/// A structure or container to organize bunch crossing data of a timeframe
/// and to facilitate fast lookup and search within bunch crossings.
class BunchCrossings
{
 public:
  /// Constructor initializes the acceleration structure
  BunchCrossings() = default;

  /// initialize this container (to be ready for lookup/search queries)
  void init(std::map<uint64_t, int> const& bcs)
  {
    clear();
    // init the structures
    for (auto& key : bcs) {
      mBCTimeVector.emplace_back(key.first);
    }
    initTimeWindows();
  }

  /// return the sorted vector of increaing BC times
  std::vector<uint64_t> const& getBCTimeVector() const { return mBCTimeVector; }

  /// Performs a "lower bound" search for timestamp within the bunch
  /// crossing data.
  ///
  /// Returns the smallest bunch crossing (index and value) equal or
  /// greater than timestamp.
  ///
  /// The functions is expected to perform much better than a binary
  /// search in the bunch crossing data directly. Expect O(1) instead
  /// of O(log(N)) at the cost of the additional memory used by this
  /// class.
  ///
  /// This is _not_ O(1).  The loop below makes it at least O(N).  The
  /// call to std::lower_bound is O(log(N)).
  std::pair<size_t, uint64_t> lower_bound(uint64_t timestamp) const
  {
    // a) determine the timewindow
    const auto NofWindows = static_cast<int>(mTimeWindows.size());
    const auto smallestBC = mBCTimeVector[0];
    const auto largestBC = mBCTimeVector.back();
    auto timeindex = std::max((int)0, (int)((timestamp - smallestBC) / mWindowSize));

    if (timeindex >= NofWindows) {
      // do extra check avoid valse positive due to machine precision
      if (timestamp > largestBC) { // there is no next greater; so the bc index is at the end of the vector
        return std::make_pair<int, uint64_t>(mBCTimeVector.size(), 0);
      }
      timeindex = int(mBCTimeVector.size() - 1);
    }

    const auto* timewindow = &mTimeWindows[timeindex];
    while (timeindex < NofWindows && (!timewindow->isOccupied() || mBCTimeVector[timewindow->to] < timestamp)) {
      timeindex = timewindow->nextOccupiedRight;
      if (timeindex < NofWindows) {
        timewindow = &mTimeWindows[timeindex];
      }
    }
    if (timeindex >= NofWindows) {
      // there is no next greater; so the bc index is at the end of the vector
      return std::make_pair<int, uint64_t>(mBCTimeVector.size(), 0);
    }
    // otherwise we actually do a search now
    std::pair<int, uint64_t> p;
    auto iter = std::lower_bound(mBCTimeVector.begin() + timewindow->from, mBCTimeVector.begin() + timewindow->to + 1, timestamp);
    int k = std::distance(mBCTimeVector.begin(), iter);
    p.first = k;
    p.second = mBCTimeVector[k];
    return p;
  }

  /// clear/reset this container
  void clear()
  {
    mBCs.clear();
    mBCTimeVector.clear();
    mTimeWindows.clear();
  }

  /// print information about this container
  void print()
  {
    LOG(info) << "Have " << mBCTimeVector.size() << " BCs";
    for (auto t : mBCTimeVector) {
      LOG(info) << t;
    }
    int twcount = 0;
    auto wsize = mWindowSize;
    for (auto& tw : mTimeWindows) {
      LOG(info) << "TimeWindow " << twcount << " [ " << wsize * twcount << ":" << wsize * (twcount + 1) << " ]  : from " << tw.from << " to " << tw.to << " nextLeft " << tw.nextOccupiedLeft << " nextRight " << tw.nextOccupiedRight;
      twcount++;
    }
  }

 private:
  std::map<uint64_t, int> mBCs;
  std::vector<uint64_t> mBCTimeVector; // simple sorted vector of BC times

  /// initialize the internal acceleration structure
  void initTimeWindows()
  {
    // on average we want say M bunch crossings per time window
    const int M = 5;
    int window_number = mBCTimeVector.size() / M;
    if (mBCTimeVector.size() % M != 0) {
      window_number += 1;
    }
    auto bcrange = (mBCTimeVector.back() + 1 - mBCTimeVector[0]);
    if (bcrange > (uint64_t(3564 * 258))) {
      LOGP(warn, "Attention: BC range {}:{} covers more than 258 orbits", mBCTimeVector[0], mBCTimeVector.back());
    }
    mWindowSize = bcrange / (1. * window_number);
    // now we go through the list of times and bucket them into the correct windows
    mTimeWindows.resize(window_number);
    for (auto bcindex = 0U; bcindex < mBCTimeVector.size(); ++bcindex) {
      auto windowindex = (int)((mBCTimeVector[bcindex] - mBCTimeVector[0]) / mWindowSize);
      // we add "bcindex" to the TimeWindow windowindex
      auto& tw = mTimeWindows[windowindex];
      if (tw.from == -1) {
        tw.from = bcindex;
      } else {
        tw.from = std::min(tw.from, static_cast<int>(bcindex));
      }
      if (tw.to == -1) {
        tw.to = bcindex;
      } else {
        tw.to = std::max(tw.to, static_cast<int>(bcindex));
      }
    }

    // now we do the neighbourhood linking of time windows
    int lastoccupied = -1;
    for (int windowindex = 0; windowindex < window_number; ++windowindex) {
      mTimeWindows[windowindex].nextOccupiedLeft = lastoccupied;
      if (mTimeWindows[windowindex].isOccupied()) {
        lastoccupied = windowindex;
      }
    }
    lastoccupied = window_number;
    for (int windowindex = window_number - 1; windowindex >= 0; --windowindex) {
      mTimeWindows[windowindex].nextOccupiedRight = lastoccupied;
      if (mTimeWindows[windowindex].isOccupied()) {
        lastoccupied = windowindex;
      }
    }
  }

  /// Internal structure to "cover" the time duration of all BCs with
  /// constant time intervals to speed up searching for a particular BC.
  /// The structure keeps indices into mBCTimeVector denoting the BCs contained within.
  struct TimeWindow {
    int from = -1;
    int to = -1;
    int nextOccupiedRight = -1; // next time window occupied to the right
    int nextOccupiedLeft = -1;  // next time window which is occupied to the left
    inline bool size() const { return to - from; }
    inline bool isOccupied() const { return size() > 0; }
  }; // end struct

  std::vector<TimeWindow> mTimeWindows; // the time window structure covering the complete duration of mBCTimeVector
  double mWindowSize;                   // the size of a single time window
}; // end internal class

// Steering bits for additional output during AOD production
enum struct AODProducerStreamerFlags : uint8_t {
  TrackQA,
};

class AODProducerWorkflowDPL : public Task
{
 public:
  AODProducerWorkflowDPL(GID::mask_t src, std::shared_ptr<DataRequest> dataRequest, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool enableSV, bool useMC = true, bool enableFITextra = false) : mUseMC(useMC), mEnableSV(enableSV), mEnableFITextra(enableFITextra), mInputSources(src), mDataRequest(dataRequest), mGGCCDBRequest(gr) {}
  ~AODProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  // takes a local vertex timing in NS and converts to a lobal BC information relative to start of timeframe
  uint64_t relativeTime_to_LocalBC(double relativeTimeStampInNS) const
  {
    return relativeTimeStampInNS > 0. ? std::round(relativeTimeStampInNS / o2::constants::lhc::LHCBunchSpacingNS) : 0;
  }
  // takes a local vertex timing in NS and converts to a global BC information
  uint64_t relativeTime_to_GlobalBC(double relativeTimeStampInNS) const
  {
    return std::uint64_t(mStartIR.toLong()) + relativeTime_to_LocalBC(relativeTimeStampInNS);
  }

  bool mThinTracks{false};
  bool mPropTracks{false};
  bool mPropMuons{false};
  float mTrackQCFraction{0.00};
  int64_t mTrackQCNTrCut{4};
  float mSqrtS{13860.};
  std::mt19937 mGenerator{}; ///< random generator for trackQA sampling
  o2::base::Propagator::MatCorrType mMatCorr{o2::base::Propagator::MatCorrType::USEMatCorrLUT};
  o2::dataformats::MeanVertexObject mVtx;
  float mMinPropR{o2::constants::geom::XTPCInnerRef + 0.1f};

  std::unordered_set<GIndex> mGIDUsedBySVtx;
  std::unordered_set<GIndex> mGIDUsedByStr;

  o2::utils::EnumFlags<AODProducerStreamerFlags> mStreamerFlags;
  std::shared_ptr<o2::utils::TreeStreamRedirector> mStreamer;

  int mNThreads = 1;
  bool mUseMC = true;
  bool mEnableSV = true; // enable secondary vertices
  bool mEnableFITextra = false;
  bool mFieldON = false;
  const float cSpeed = 0.029979246f; // speed of light in TOF units

  GID::mask_t mInputSources;
  int64_t mTFNumber{-1};
  int mRunNumber{-1};
  int mTruncate{1};
  int mRecoOnly{0};
  o2::InteractionRecord mStartIR{}; // TF 1st IR
  TString mLPMProdTag{""};
  TString mAnchorPass{""};
  TString mAnchorProd{""};
  TString mRecoPass{""};
  TStopwatch mTimer;
  bool mEMCselectLeading{false};
  uint64_t mEMCALTrgClassMask = 0;

  // unordered map connects global indices and table indices of barrel tracks
  std::unordered_map<GIndex, int> mGIDToTableID;
  int mTableTrID{0};
  // unordered map connects global indices and table indices of fwd tracks
  std::unordered_map<GIndex, int> mGIDToTableFwdID;
  int mTableTrFwdID{0};
  // unordered map connects global indices and table indices of MFT tracks
  std::unordered_map<GIndex, int> mGIDToTableMFTID;
  int mTableTrMFTID{0};
  // unordered map connects global indices and table indices of vertices
  std::unordered_map<GIndex, int> mVtxToTableCollID;
  int mTableCollID{0};
  // unordered map connects global indices and table indices of V0s (needed for cascades references)
  std::unordered_map<GIndex, int> mV0ToTableID;
  int mTableV0ID{0};

  // Strangeness tracking indices lookup tables
  std::vector<int> mVertexStrLUT;                    /// LUT for accessing strangeness tracks for each vertex
  std::vector<std::pair<int, int>> mCollisionStrTrk; /// collision index and original index of the strangeness track
  std::vector<int> mStrTrkIndices;                   /// indices of strangeness tracks in the track table

  //  std::unordered_map<int, int> mIndexTableFwd;
  std::vector<int> mIndexTableFwd;
  int mIndexFwdID{0};
  //  std::unordered_map<int, int> mIndexTableMFT;
  std::vector<int> mIndexTableMFT;
  int mIndexMFTID{0};

  BunchCrossings mBCLookup;

  // zdc helper maps to avoid a number of "if" statements
  // when filling ZDC table
  std::array<float, o2::zdc::NChannels> mZDCEnergyMap; // mapping detector id to a corresponding energy
  std::array<float, o2::zdc::NTDCChannels> mZDCTDCMap; // mapping TDC channel id to a corresponding TDC value

  std::vector<uint16_t> mITSTPCTRDTriggers; // mapping from TRD tracks ID to corresponding trigger (for tracks time extraction)
  std::vector<uint16_t> mTPCTRDTriggers;    // mapping from TRD tracks ID to corresponding trigger (for tracks time extraction)
  std::vector<uint16_t> mITSROFs;           // mapping from ITS tracks ID to corresponding ROF (for SA ITS tracks time extraction)
  std::vector<uint16_t> mMFTROFs;           // mapping from MFT tracks ID to corresponding ROF (for SA MFT tracks time extraction)
  std::vector<uint16_t> mMCHROFs;           // mapping from MCH tracks ID to corresponding ROF (for SA MCH tracks time extraction)
  double mITSROFrameHalfLengthNS = -1;      // ITS ROF half length
  double mMFTROFrameHalfLengthNS = -1;      // ITS ROF half length
  double mITSROFBiasNS = 0;                 // ITS ROF start bias
  double mMFTROFBiasNS = 0;                 // ITS ROF start bias
  double mNSigmaTimeTrack = -1;             // number track errors sigmas (for gaussian errors only) used in track-vertex matching
  double mTimeMarginTrackTime = -1;         // safety margin in NS used for track-vertex matching (additive to track uncertainty)
  double mTPCBinNS = -1;                    // inverse TPC time-bin in ns

  // Container used to mark MC particles to store/transfer to AOD.
  // Mapping of eventID, sourceID, trackID to some integer.
  // The first two indices are not sparse whereas the trackID index is sparse which explains
  // the combination of vector and map
  std::vector<std::vector<std::unordered_map<int, int>>> mToStore;
  o2::steer::MCKinematicsReader* mMCKineReader = nullptr; //!

  // production metadata
  std::vector<TString> mMetaDataKeys;
  std::vector<TString> mMetaDataVals;

  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;

  o2::globaltracking::MatchGlobalFwd mMatching;

  static constexpr int TOFTimePrecPS = 16; // required max error in ps for TOF tracks
  // truncation is enabled by default
  uint32_t mCollisionPosition = 0xFFFFFFF0;    // 19 bits mantissa
  uint32_t mCollisionPositionCov = 0xFFFFE000; // 10 bits mantissa
  uint32_t mTrackX = 0xFFFFFFF0;               // 19 bits
  uint32_t mTrackAlpha = 0xFFFFFFF0;           // 19 bits
  uint32_t mTrackSnp = 0xFFFFFF00;             // 15 bits
  uint32_t mTrackTgl = 0xFFFFFF00;             // 15 bits
  uint32_t mTrack1Pt = 0xFFFFFC00;             // 13 bits
  uint32_t mTrackCovDiag = 0xFFFFFF00;         // 15 bits
  uint32_t mTrackChi2 = 0xFFFF0000;            // 7 bits
  uint32_t mTrackCovOffDiag = 0xFFFF0000;      // 7 bits
  uint32_t mTrackSignal = 0xFFFFFF00;          // 15 bits
  uint32_t mTrackTime = 0xFFFFFFFF;            // use full float precision for time
  uint32_t mTPCTime0 = 0xFFFFFFE0;             // 18 bits, providing 14256./(1<<19) = 0.027 TB precision e.g., ~0.13 mm in z
  uint32_t mTrackTimeError = 0xFFFFFF00;       // 15 bits
  uint32_t mTrackPosEMCAL = 0xFFFFFF00;        // 15 bits
  uint32_t mTracklets = 0xFFFFFF00;            // 15 bits
  uint32_t mMcParticleW = 0xFFFFFFF0;          // 19 bits
  uint32_t mMcParticlePos = 0xFFFFFFF0;        // 19 bits
  uint32_t mMcParticleMom = 0xFFFFFFF0;        // 19 bits
  uint32_t mCaloAmp = 0xFFFFFF00;              // 15 bits todo check which truncation should actually be used
  uint32_t mCaloTime = 0xFFFFFF00;             // 15 bits todo check which truncation should actually be used
  uint32_t mCPVPos = 0xFFFFF800;               // 12 bits
  uint32_t mCPVAmpl = 0xFFFFFF00;              // 15 bits
  uint32_t mMuonTr1P = 0xFFFFFC00;             // 13 bits
  uint32_t mMuonTrThetaX = 0xFFFFFF00;         // 15 bits
  uint32_t mMuonTrThetaY = 0xFFFFFF00;         // 15 bits
  uint32_t mMuonTrZmu = 0xFFFFFFF0;            // 19 bits
  uint32_t mMuonTrBend = 0xFFFFFFF0;           // 19 bits
  uint32_t mMuonTrNonBend = 0xFFFFFFF0;        // 19 bits
  uint32_t mMuonTrCov = 0xFFFF0000;            // 7 bits
  uint32_t mMuonCl = 0xFFFFFF00;               // 15 bits
  uint32_t mMuonClErr = 0xFFFF0000;            // 7 bits
  uint32_t mV0Time = 0xFFFFF000;               // 11 bits
  uint32_t mV0ChannelTime = 0xFFFFFF00;        // 15 bits
  uint32_t mFDDTime = 0xFFFFF000;              // 11 bits
  uint32_t mFDDChannelTime = 0xFFFFFF00;       // 15 bits
  uint32_t mT0Time = 0xFFFFFF00;               // 15 bits
  uint32_t mT0ChannelTime = 0xFFFFFFF0;        // 19 bits
  uint32_t mV0Amplitude = 0xFFFFF000;          // 11 bits
  uint32_t mFDDAmplitude = 0xFFFFF000;         // 11 bits
  uint32_t mT0Amplitude = 0xFFFFF000;          // 11 bits
  int mCTPReadout = 0;                         // 0 = use CTP readout from CTP; 1 = create CTP readout
  bool mCTPConfigPerRun = false;               // 0 = use common CTPconfig as for MC; 1 = run dependent CTP config
  // helper struct for extra info in fillTrackTablesPerCollision()
  struct TrackExtraInfo {
    float tpcInnerParam = 0.f;
    uint32_t flags = 0;
    uint32_t itsClusterSizes = 0u;
    uint8_t itsClusterMap = 0;
    uint8_t tpcNClsFindable = 0;
    int8_t tpcNClsFindableMinusFound = 0;
    int8_t tpcNClsFindableMinusPID = 0;
    int8_t tpcNClsFindableMinusCrossedRows = 0;
    uint8_t tpcNClsShared = 0;
    uint8_t trdPattern = 0;
    float itsChi2NCl = -999.f;
    float tpcChi2NCl = -999.f;
    float trdChi2 = -999.f;
    float tofChi2 = -999.f;
    float tpcSignal = -999.f;
    float trdSignal = -999.f;
    float length = -999.f;
    float tofExpMom = -999.f;
    float trackEtaEMCAL = -999.f;
    float trackPhiEMCAL = -999.f;
    float trackTime = -999.f;
    float trackTimeRes = -999.f;
    int diffBCRef = 0; // offset of time reference BC from the start of the orbit
    int bcSlice[2] = {-1, -1};
    bool isTPConly = false; // not to be written out
  };

  struct TrackQA {
    GID trackID;
    float tpcTime0{};
    int16_t tpcdcaR{};
    int16_t tpcdcaZ{};
    uint8_t tpcClusterByteMask{};
    uint8_t tpcdEdxMax0R{};
    uint8_t tpcdEdxMax1R{};
    uint8_t tpcdEdxMax2R{};
    uint8_t tpcdEdxMax3R{};
    uint8_t tpcdEdxTot0R{};
    uint8_t tpcdEdxTot1R{};
    uint8_t tpcdEdxTot2R{};
    uint8_t tpcdEdxTot3R{};
    int8_t dRefContY{std::numeric_limits<int8_t>::min()};
    int8_t dRefContZ{std::numeric_limits<int8_t>::min()};
    int8_t dRefContSnp{std::numeric_limits<int8_t>::min()};
    int8_t dRefContTgl{std::numeric_limits<int8_t>::min()};
    int8_t dRefContQ2Pt{std::numeric_limits<int8_t>::min()};
    int8_t dRefGloY{std::numeric_limits<int8_t>::min()};
    int8_t dRefGloZ{std::numeric_limits<int8_t>::min()};
    int8_t dRefGloSnp{std::numeric_limits<int8_t>::min()};
    int8_t dRefGloTgl{std::numeric_limits<int8_t>::min()};
    int8_t dRefGloQ2Pt{std::numeric_limits<int8_t>::min()};
    int8_t dTofdX{std::numeric_limits<int8_t>::min()};
    int8_t dTofdZ{std::numeric_limits<int8_t>::min()};
  };

  // helper struct for addToFwdTracksTable()
  struct FwdTrackInfo {
    uint8_t trackTypeId = 0;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float rabs = 0.f;
    float phi = 0.f;
    float tanl = 0.f;
    float invqpt = 0.f;
    float chi2 = 0.f;
    float pdca = 0.f;
    int nClusters = -1;
    float chi2matchmchmid = -1.0;
    float chi2matchmchmft = -1.0;
    float matchscoremchmft = -1.0;
    int matchmfttrackid = -1;
    int matchmchtrackid = -1;
    uint16_t mchBitMap = 0;
    uint8_t midBitMap = 0;
    uint32_t midBoards = 0;
    float trackTime = -999.f;
    float trackTimeRes = -999.f;
  };

  // helper struct for addToFwdTracksTable()
  struct FwdTrackCovInfo {
    float sigX = 0.f;
    float sigY = 0.f;
    float sigPhi = 0.f;
    float sigTgl = 0.f;
    float sig1Pt = 0.f;
    int8_t rhoXY = 0;
    int8_t rhoPhiX = 0;
    int8_t rhoPhiY = 0;
    int8_t rhoTglX = 0;
    int8_t rhoTglY = 0;
    int8_t rhoTglPhi = 0;
    int8_t rho1PtX = 0;
    int8_t rho1PtY = 0;
    int8_t rho1PtPhi = 0;
    int8_t rho1PtTgl = 0;
  };

  // helper struct for mc track labels
  // using -1 as dummies for AOD
  struct MCLabels {
    uint32_t labelID = -1;
    uint16_t labelMask = 0;
    uint8_t fwdLabelMask = 0;
  };

  // counters for TPC clusters
  struct TPCCounters {
    uint8_t shared = 0;
    uint8_t found = 0;
    uint8_t crossed = 0;
  };
  std::vector<TPCCounters> mTPCCounters;

  void updateTimeDependentParams(ProcessingContext& pc);

  void addRefGlobalBCsForTOF(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices,
                             const o2::globaltracking::RecoContainer& data, std::map<uint64_t, int>& bcsMap);
  void createCTPReadout(const o2::globaltracking::RecoContainer& recoData, std::vector<o2::ctp::CTPDigit>& ctpDigits, ProcessingContext& pc);
  void collectBCs(const o2::globaltracking::RecoContainer& data,
                  const std::vector<o2::InteractionTimeRecord>& mcRecords,
                  std::map<uint64_t, int>& bcsMap);

  template <typename TracksCursorType, typename TracksCovCursorType>
  void addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                        const o2::track::TrackParCov& track, int collisionID, aod::track::TrackTypeEnum type = aod::track::TrackIU);

  template <typename TracksExtraCursorType>
  void addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor, TrackExtraInfo& extraInfoHolder);

  template <typename TracksQACursorType>
  void addToTracksQATable(TracksQACursorType& tracksQACursor, TrackQA& trackQAInfoHolder);

  template <typename mftTracksCursorType, typename AmbigMFTTracksCursorType>
  void addToMFTTracksTable(mftTracksCursorType& mftTracksCursor, AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                           GIndex trackID, const o2::globaltracking::RecoContainer& data, int collisionID,
                           std::uint64_t collisionBC, const std::map<uint64_t, int>& bcsMap);

  template <typename fwdTracksCursorType, typename fwdTracksCovCursorType, typename AmbigFwdTracksCursorType, typename mftTracksCovCursorType>
  void addToFwdTracksTable(fwdTracksCursorType& fwdTracksCursor, fwdTracksCovCursorType& fwdTracksCovCursor, AmbigFwdTracksCursorType& ambigFwdTracksCursor, mftTracksCovCursorType& mftTracksCovCursor,
                           GIndex trackID, const o2::globaltracking::RecoContainer& data, int collisionID, std::uint64_t collisionBC, const std::map<uint64_t, int>& bcsMap);

  TrackExtraInfo processBarrelTrack(int collisionID, std::uint64_t collisionBC, GIndex trackIndex, const o2::globaltracking::RecoContainer& data, const std::map<uint64_t, int>& bcsMap);
  TrackQA processBarrelTrackQA(int collisionID, std::uint64_t collisionBC, GIndex trackIndex, const o2::globaltracking::RecoContainer& data, const std::map<uint64_t, int>& bcsMap);

  bool propagateTrackToPV(o2::track::TrackParametrizationWithError<float>& trackPar, const o2::globaltracking::RecoContainer& data, int colID);
  void extrapolateToCalorimeters(TrackExtraInfo& extraInfoHolder, const o2::track::TrackPar& track);
  void cacheTriggers(const o2::globaltracking::RecoContainer& recoData);

  // helper for track tables
  // * fills tables collision by collision
  // * interaction time is for TOF information
  template <typename TracksCursorType, typename TracksCovCursorType, typename TracksExtraCursorType, typename TracksQACursorType, typename AmbigTracksCursorType,
            typename MFTTracksCursorType, typename MFTTracksCovCursorType, typename AmbigMFTTracksCursorType,
            typename FwdTracksCursorType, typename FwdTracksCovCursorType, typename AmbigFwdTracksCursorType, typename FwdTrkClsCursorType>
  void fillTrackTablesPerCollision(int collisionID,
                                   std::uint64_t collisionBC,
                                   const o2::dataformats::VtxTrackRef& trackRef,
                                   const gsl::span<const GIndex>& GIndices,
                                   const o2::globaltracking::RecoContainer& data,
                                   TracksCursorType& tracksCursor,
                                   TracksCovCursorType& tracksCovCursor,
                                   TracksExtraCursorType& tracksExtraCursor,
                                   TracksQACursorType& tracksQACursor,
                                   AmbigTracksCursorType& ambigTracksCursor,
                                   MFTTracksCursorType& mftTracksCursor,
                                   MFTTracksCovCursorType& mftTracksCovCursor,
                                   AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                                   FwdTracksCursorType& fwdTracksCursor,
                                   FwdTracksCovCursorType& fwdTracksCovCursor,
                                   AmbigFwdTracksCursorType& ambigFwdTracksCursor,
                                   FwdTrkClsCursorType& fwdTrkClsCursor,
                                   const std::map<uint64_t, int>& bcsMap);

  template <typename FwdTrkClsCursorType>
  void addClustersToFwdTrkClsTable(const o2::globaltracking::RecoContainer& recoData, FwdTrkClsCursorType& fwdTrkClsCursor, GIndex trackID, int fwdTrackId);

  void fillIndexTablesPerCollision(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices, const o2::globaltracking::RecoContainer& data);

  template <typename V0CursorType, typename CascadeCursorType, typename Decay3bodyCursorType>
  void fillSecondaryVertices(const o2::globaltracking::RecoContainer& data, V0CursorType& v0Cursor, CascadeCursorType& cascadeCursor, Decay3bodyCursorType& decay3bodyCursor);

  template <typename HMPCursorType>
  void fillHMPID(const o2::globaltracking::RecoContainer& recoData, HMPCursorType& hmpCursor);

  void prepareStrangenessTracking(const o2::globaltracking::RecoContainer& recoData);
  template <typename V0C, typename CC, typename D3BC>
  void fillStrangenessTrackingTables(const o2::globaltracking::RecoContainer& data, V0C& v0Cursor, CC& cascadeCursor, D3BC& decay3bodyCursor);

  /** some other types we will use */
  using MCCollisionCursor = aodmchelpers::CollisionCursor;
  using XSectionCursor = aodmchelpers::XSectionCursor;
  using PdfInfoCursor = aodmchelpers::PdfInfoCursor;
  using HeavyIonCursor = aodmchelpers::HeavyIonCursor;
  using MCParticlesCursor = aodmchelpers::ParticleCursor;
  using HepMCUpdate = aodmchelpers::HepMCUpdate;
  using MCEventHeader = dataformats::MCEventHeader;
  /** Rules for when to update HepMC tables */
  HepMCUpdate mXSectionUpdate = HepMCUpdate::anyKey;
  HepMCUpdate mPdfInfoUpdate = HepMCUpdate::anyKey;
  HepMCUpdate mHeavyIonUpdate = HepMCUpdate::anyKey;
  /**
   * Update the header (collision and HepMC aux) information.
   *
   * When updating the HepMC aux tables, we take the relevant policies
   * into account (mXSectionUpdate, mPdfInfoUpdate, mHeavyIonUpdate).
   *
   * - If a policy is "never", then the corresponding table is never
   *   updated.
   *
   * - If the policy is "always", then the table is always
   *   update.
   *
   * - If the policy is either "anyKey" or "allKeys", _and_
   *   this is the first event, then we check if any or all keys,
   *   respectively are present in the header.
   *
   *   - If that check fails, then we do not update and set the
   *     corresponding policy to be "never".
   *
   *   - If the check succeeds, then we do update the table, and set
   *     the corresponding policty to "always".
   *
   *   In this way, we will let the first event decide what to do for
   *   subsequent events and thus avoid too many string comparisions.
   *
   * @param collisionCursor Cursor over aod::McCollisions
   * @param xSectionCursor Cursor over aod::HepMCXSections
   * @param pdfInfoCursor Cursor over aod::HepMCPdfInfos
   * @param heavyIonCursor Cursor over aod::HepMCHeavyIons
   * @param header Header to read information from
   * @param collisionID Index of collision in the table
   * @param bcID Current event identifier (bcID)
   * @param time Time of event
   * @param generatorID Generator identifier, if any
   * @param sourceID Source identifier
   *
   */
  void updateMCHeader(MCCollisionCursor& collisionCursor,
                      XSectionCursor& xSectionCursor,
                      PdfInfoCursor& pdfInfoCursor,
                      HeavyIonCursor& heavyIonCursor,
                      const MCEventHeader& header,
                      int collisionID,
                      int bcID,
                      float time,
                      short generatorID,
                      int sourceID);

  void fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                            MCParticlesCursor& mcParticlesCursor,
                            const gsl::span<const o2::dataformats::VtxTrackRef>& primVer2TRefs,
                            const gsl::span<const GIndex>& GIndices,
                            const o2::globaltracking::RecoContainer& data,
                            const std::vector<std::vector<int>>& mcColToEvSrc);

  template <typename MCTrackLabelCursorType, typename MCMFTTrackLabelCursorType, typename MCFwdTrackLabelCursorType>
  void fillMCTrackLabelsTable(MCTrackLabelCursorType& mcTrackLabelCursor,
                              MCMFTTrackLabelCursorType& mcMFTTrackLabelCursor,
                              MCFwdTrackLabelCursorType& mcFwdTrackLabelCursor,
                              const o2::dataformats::VtxTrackRef& trackRef,
                              const gsl::span<const GIndex>& primVerGIs,
                              const o2::globaltracking::RecoContainer& data,
                              int vertexId = -1);

  std::uint64_t fillBCSlice(int (&slice)[2], double tmin, double tmax, const std::map<uint64_t, int>& bcsMap) const;

  std::vector<uint8_t> fillBCFlags(const o2::globaltracking::RecoContainer& data, std::map<uint64_t, int>& bcsMap) const;

  // helper for tpc clusters
  void countTPCClusters(const o2::globaltracking::RecoContainer& data);

  // helper for trd pattern
  uint8_t getTRDPattern(const o2::trd::TrackTRD& track);

  template <typename TCaloHandler, typename TCaloCursor, typename TCaloTRGCursor, typename TMCCaloLabelCursor>
  void addToCaloTable(TCaloHandler& caloHandler, TCaloCursor& caloCellCursor, TCaloTRGCursor& caloTRGCursor,
                      TMCCaloLabelCursor& mcCaloCellLabelCursor, int eventID, int bcID, int8_t caloType);

  template <typename TCaloCursor, typename TCaloTRGCursor, typename TMCCaloLabelCursor>
  void fillCaloTable(TCaloCursor& caloCellCursor, TCaloTRGCursor& caloTRGCursor,
                     TMCCaloLabelCursor& mcCaloCellLabelCursor, const std::map<uint64_t, int>& bcsMap,
                     const o2::globaltracking::RecoContainer& data);

  std::set<uint64_t> filterEMCALIncomplete(const gsl::span<const o2::emcal::TriggerRecord> triggers);
};

/// create a processor spec
framework::DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool enableSV, bool enableST, bool useMC, bool CTPConfigPerRun, bool enableFITextra);

// helper interface for calo cells to "befriend" emcal and phos cells
class CellHelper
{
 public:
  static int8_t getTriggerBits(const o2::emcal::Cell& /*cell*/)
  {
    return 0; // dummy value
  }

  static int8_t getTriggerBits(const o2::phos::Cell& cell)
  {
    return (cell.getType() == o2::phos::ChannelType_t::TRU2x2) ? 0 : 1;
  }

  static int16_t getCellNumber(const o2::emcal::Cell& cell)
  {
    return cell.getTower();
  }

  static int16_t getCellNumber(const o2::phos::Cell& cell)
  {
    if (cell.getTRU()) {
      return cell.getTRUId();
    }
    return cell.getAbsId();
  }
  // If this cell - trigger one?
  static bool isTRU(const o2::emcal::Cell& cell)
  {
    return cell.getTRU();
  }

  static bool isTRU(const o2::phos::Cell& cell)
  {
    return cell.getTRU();
  }

  static int16_t getFastOrAbsID(const o2::emcal::Cell& /*cell*/)
  {
    return 0; // dummy value
  }

  static int16_t getFastOrAbsID(const o2::phos::Cell& cell)
  {
    return cell.getTRUId();
  }

  static float getAmplitude(const o2::emcal::Cell& cell)
  {
    return cell.getAmplitude();
  }

  static float getAmplitude(const o2::phos::Cell& cell)
  {
    return cell.getEnergy();
  }

  static int16_t getLnAmplitude(const o2::emcal::Cell& /*cell*/)
  {
    return 0; // dummy value
  }

  static int16_t getLnAmplitude(const o2::phos::Cell& cell)
  {
    return cell.getEnergy(); // dummy value
  }

  static float getTimeStamp(const o2::emcal::Cell& cell)
  {
    return cell.getTimeStamp();
  }

  static float getTimeStamp(const o2::phos::Cell& cell)
  {
    return cell.getTime();
  }
};

} // namespace o2::aodproducer

#endif /* O2_AODPRODUCER_WORKFLOW_SPEC */
