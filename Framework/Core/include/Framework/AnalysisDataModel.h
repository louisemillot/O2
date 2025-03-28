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
#ifndef O2_FRAMEWORK_ANALYSISDATAMODEL_H_
#define O2_FRAMEWORK_ANALYSISDATAMODEL_H_

#include "Framework/ASoA.h"

#include <cmath>
#include <bitset>
#include <limits>
#include <numeric>

#include "Framework/DataTypes.h"
#include "CommonConstants/MathConstants.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonConstants/GeomConstants.h"
#include "CommonConstants/ZDCConstants.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "Framework/PID.h"

namespace o2
{
namespace aod
{
namespace bc
{
DECLARE_SOA_COLUMN(RunNumber, runNumber, int);          //! Run number
DECLARE_SOA_COLUMN(GlobalBC, globalBC, uint64_t);       //! Bunch crossing number (globally unique in this run)
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint64_t); //! CTP trigger mask
DECLARE_SOA_COLUMN(InputMask, inputMask, uint64_t);     //! CTP input mask
DECLARE_SOA_COLUMN(Flags, flags, uint8_t);              //! BC flags (e.g. tagging of UPC tracking settings, etc)
} // namespace bc

DECLARE_SOA_TABLE(BCs_000, "AOD", "BC", //! Root of data model for tables pointing to a bunch crossing
                  o2::soa::Index<>,
                  bc::RunNumber, bc::GlobalBC,
                  bc::TriggerMask);
DECLARE_SOA_TABLE_VERSIONED(BCs_001, "AOD", "BC", 1, //! Root of data model for tables pointing to a bunch crossing, version 1
                            o2::soa::Index<>,
                            bc::RunNumber, bc::GlobalBC,
                            bc::TriggerMask, bc::InputMask);
DECLARE_SOA_TABLE(BCFlags, "AOD", "BCFLAG", //! flag for tagging UPCs, joinable with BCs
                  bc::Flags);

using BCs = BCs_001; // current version
using BC = BCs::iterator;

namespace timestamp
{
DECLARE_SOA_COLUMN(Timestamp, timestamp, uint64_t); //! Timestamp of a BC in ms (epoch style)
} // namespace timestamp

DECLARE_SOA_TABLE(Timestamps, "AOD", "TIMESTAMPS", //! Table which holds the timestamp of a BC
                  timestamp::Timestamp);

using BCsWithTimestamps = soa::Join<aod::BCs, aod::Timestamps>;
} // namespace aod

namespace soa
{
extern template struct JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::BCs, aod::Timestamps>;
}
namespace aod
{
namespace collision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                              //! Most probably BC to where this collision has occured
DECLARE_SOA_COLUMN(PosX, posX, float);                         //! X Vertex position in cm
DECLARE_SOA_COLUMN(PosY, posY, float);                         //! Y Vertex position in cm
DECLARE_SOA_COLUMN(PosZ, posZ, float);                         //! Z Vertex position in cm
DECLARE_SOA_COLUMN(CovXX, covXX, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovXY, covXY, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovXZ, covXZ, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovYY, covYY, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovYZ, covYZ, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovZZ, covZZ, float);                       //! Vertex covariance matrix
DECLARE_SOA_COLUMN(Flags, flags, uint16_t);                    //! Run 2: see CollisionFlagsRun2 | Run 3: see Vertex::Flags
DECLARE_SOA_COLUMN(Chi2, chi2, float);                         //! Chi2 of vertex fit
DECLARE_SOA_COLUMN(NumContrib, numContrib, uint16_t);          //! Number of tracks used for the vertex
DECLARE_SOA_COLUMN(CollisionTime, collisionTime, float);       //! Collision time in ns relative to BC stored in bc()
DECLARE_SOA_COLUMN(CollisionTimeRes, collisionTimeRes, float); //! Resolution of collision time
} // namespace collision

DECLARE_SOA_TABLE(Collisions_000, "AOD", "COLLISION", //! Time and vertex information of collision
                  o2::soa::Index<>, collision::BCId,
                  collision::PosX, collision::PosY, collision::PosZ,
                  collision::CovXX, collision::CovXY, collision::CovXZ, collision::CovYY, collision::CovYZ, collision::CovZZ,
                  collision::Flags, collision::Chi2, collision::NumContrib,
                  collision::CollisionTime, collision::CollisionTimeRes);

DECLARE_SOA_TABLE_VERSIONED(Collisions_001, "AOD", "COLLISION", 1, //! Time and vertex information of collision
                            o2::soa::Index<>, collision::BCId,
                            collision::PosX, collision::PosY, collision::PosZ,
                            collision::CovXX, collision::CovXY, collision::CovYY, collision::CovXZ, collision::CovYZ, collision::CovZZ,
                            collision::Flags, collision::Chi2, collision::NumContrib,
                            collision::CollisionTime, collision::CollisionTimeRes);

using Collisions = Collisions_001; // current version
using Collision = Collisions::iterator;

// NOTE Relation between Collisions and BC table
// (important for pp in case of ambiguous assignment)
// A collision entry points to the entry in the BC table based on the calculated BC from the collision time
// To study other compatible triggers with the collision time, check the tutorial: compatibleBCs.cxx

namespace track
{
// TRACKPAR TABLE definition
DECLARE_SOA_INDEX_COLUMN(Collision, collision);    //! Collision to which this track belongs
DECLARE_SOA_COLUMN(TrackType, trackType, uint8_t); //! Type of track. See enum TrackTypeEnum. This cannot be used to decide which detector has contributed to this track. Use hasITS, hasTPC, etc.
DECLARE_SOA_COLUMN(X, x, float);                   //!
DECLARE_SOA_COLUMN(Alpha, alpha, float);           //!
DECLARE_SOA_COLUMN(Y, y, float);                   //!
DECLARE_SOA_COLUMN(Z, z, float);                   //!
DECLARE_SOA_COLUMN(Snp, snp, float);               //!
DECLARE_SOA_COLUMN(Tgl, tgl, float);               //!
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float);   //! (sign of charge)/Pt in c/GeV. Use pt() and sign() instead
DECLARE_SOA_EXPRESSION_COLUMN(Phi, phi, float,     //! Phi of the track, in radians within [0, 2pi)
                              ifnode(nasin(aod::track::snp) + aod::track::alpha < 0.0f, nasin(aod::track::snp) + aod::track::alpha + o2::constants::math::TwoPI,
                                     ifnode(nasin(aod::track::snp) + aod::track::alpha >= o2::constants::math::TwoPI, nasin(aod::track::snp) + aod::track::alpha - o2::constants::math::TwoPI,
                                            nasin(aod::track::snp) + aod::track::alpha)));
DECLARE_SOA_EXPRESSION_COLUMN(Eta, eta, float, //! Pseudorapidity
                              -1.f * nlog(ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::track::tgl))));
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //! Transverse momentum of the track in GeV/c
                              ifnode(nabs(aod::track::signed1Pt) <= o2::constants::math::Almost0, o2::constants::math::VeryBig, nabs(1.f / aod::track::signed1Pt)));
DECLARE_SOA_DYNAMIC_COLUMN(IsWithinBeamPipe, isWithinBeamPipe, //! Is the track within the beam pipe (= successfully propagated to a collision vertex)
                           [](float x) -> bool { return (std::fabs(x) < o2::constants::geom::XBeamPipeOuterRef); });
DECLARE_SOA_DYNAMIC_COLUMN(Sign, sign, //! Charge: positive: 1, negative: -1
                           [](float signed1Pt) -> short { return (signed1Pt > 0) ? 1 : -1; });
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, //! Momentum in x-direction in GeV/c
                           [](float signed1Pt, float snp, float alpha) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             // FIXME: GCC & clang should optimize to sincosf
                             float cs = cosf(alpha), sn = sinf(alpha);
                             auto r = std::sqrt((1.f - snp) * (1.f + snp));
                             return pt * (r * cs - snp * sn);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //! Momentum in y-direction in GeV/c
                           [](float signed1Pt, float snp, float alpha) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             // FIXME: GCC & clang should optimize to sincosf
                             float cs = cosf(alpha), sn = sinf(alpha);
                             auto r = std::sqrt((1.f - snp) * (1.f + snp));
                             return pt * (snp * cs + r * sn);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //! Momentum in z-direction in GeV/c
                           [](float signed1Pt, float tgl) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             return pt * tgl;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(PVector, pVector, //! Momentum vector in x,y,z-directions in GeV/c
                           [](float signed1Pt, float snp, float alpha, float tgl) -> std::array<float, 3> {
                             const auto pt = 1.f / std::abs(signed1Pt);
                             // FIXME: GCC & clang should optimize to sincosf
                             const float cs = cosf(alpha), sn = sinf(alpha);
                             const auto r = std::sqrt((1.f - snp) * (1.f + snp));
                             const auto px = pt * (r * cs - snp * sn);
                             const auto py = pt * (snp * cs + r * sn);
                             const auto pz = pt * tgl;
                             return std::array<float, 3>{px, py, pz};
                           });
DECLARE_SOA_EXPRESSION_COLUMN(P, p, float, //! Momentum in Gev/c
                              ifnode(nabs(aod::track::signed1Pt) <= o2::constants::math::Almost0, o2::constants::math::VeryBig, 0.5f * (ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::track::tgl)) + 1.f / ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::track::tgl))) / nabs(aod::track::signed1Pt)));
DECLARE_SOA_DYNAMIC_COLUMN(Energy, energy, //! Track energy, computed under the mass assumption given as input
                           [](float signed1Pt, float tgl, float mass) -> float {
                             const auto pt = 1.f / std::abs(signed1Pt);
                             const auto p = 0.5f * (std::tan(o2::constants::math::PIQuarter - 0.5f * std::atan(tgl)) + 1.f / std::tan(o2::constants::math::PIQuarter - 0.5f * std::atan(tgl))) * pt;
                             return std::sqrt(p * p + mass * mass);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Rapidity, rapidity, //! Track rapidity, computed under the mass assumption given as input
                           [](float signed1Pt, float tgl, float mass) -> float {
                             const auto pt = 1.f / std::abs(signed1Pt);
                             const auto pz = pt * tgl;
                             const auto p = 0.5f * (std::tan(o2::constants::math::PIQuarter - 0.5f * std::atan(tgl)) + 1.f / std::tan(o2::constants::math::PIQuarter - 0.5f * std::atan(tgl))) * pt;
                             const auto energy = std::sqrt(p * p + mass * mass);
                             return 0.5f * std::log((energy + pz) / (energy - pz));
                           });

// TRACKPARCOV TABLE definition
DECLARE_SOA_COLUMN(SigmaY, sigmaY, float);        //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaZ, sigmaZ, float);        //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaSnp, sigmaSnp, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaTgl, sigmaTgl, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(Sigma1Pt, sigma1Pt, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(RhoZY, rhoZY, int8_t);         //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoSnpY, rhoSnpY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoSnpZ, rhoSnpZ, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglY, rhoTglY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglZ, rhoTglZ, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglSnp, rhoTglSnp, int8_t); //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtY, rho1PtY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtZ, rho1PtZ, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtSnp, rho1PtSnp, int8_t); //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtTgl, rho1PtTgl, int8_t); //! Covariance matrix in compressed form

DECLARE_SOA_EXPRESSION_COLUMN(CYY, cYY, float, //!
                              aod::track::sigmaY* aod::track::sigmaY);
DECLARE_SOA_EXPRESSION_COLUMN(CZY, cZY, float, //!
                              (aod::track::rhoZY / 128.f) * (aod::track::sigmaZ * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CZZ, cZZ, float, //!
                              aod::track::sigmaZ* aod::track::sigmaZ);
DECLARE_SOA_EXPRESSION_COLUMN(CSnpY, cSnpY, float, //!
                              (aod::track::rhoSnpY / 128.f) * (aod::track::sigmaSnp * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CSnpZ, cSnpZ, float, //!
                              (aod::track::rhoSnpZ / 128.f) * (aod::track::sigmaSnp * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(CSnpSnp, cSnpSnp, float, //!
                              aod::track::sigmaSnp* aod::track::sigmaSnp);
DECLARE_SOA_EXPRESSION_COLUMN(CTglY, cTglY, float, //!
                              (aod::track::rhoTglY / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CTglZ, cTglZ, float, //!
                              (aod::track::rhoTglZ / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(CTglSnp, cTglSnp, float, //!
                              (aod::track::rhoTglSnp / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaSnp));
DECLARE_SOA_EXPRESSION_COLUMN(CTglTgl, cTglTgl, float, //!
                              aod::track::sigmaTgl* aod::track::sigmaTgl);
DECLARE_SOA_EXPRESSION_COLUMN(C1PtY, c1PtY, float, //!
                              (aod::track::rho1PtY / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtZ, c1PtZ, float, //!
                              (aod::track::rho1PtZ / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtSnp, c1PtSnp, float, //!
                              (aod::track::rho1PtSnp / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaSnp));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtTgl, c1PtTgl, float, //!
                              (aod::track::rho1PtTgl / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaTgl));
DECLARE_SOA_EXPRESSION_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float, //!
                              aod::track::sigma1Pt* aod::track::sigma1Pt);

// TRACKEXTRA TABLE definition
DECLARE_SOA_COLUMN(TPCInnerParam, tpcInnerParam, float);                                      //! Momentum at inner wall of the TPC
DECLARE_SOA_COLUMN(Flags, flags, uint32_t);                                                   //! Track flags. Run 2: see TrackFlagsRun2Enum | Run 3: see TrackFlags
DECLARE_SOA_COLUMN(ITSClusterSizes, itsClusterSizes, uint32_t);                               //! Clusters sizes, four bits per a layer, starting from the innermost
DECLARE_SOA_COLUMN(ITSClusterMap, itsClusterMap, uint8_t);                                    //! Old cluster ITS cluster map, kept for version 0 compatibility
DECLARE_SOA_COLUMN(TPCNClsFindable, tpcNClsFindable, uint8_t);                                //! Findable TPC clusters for this track geometry
DECLARE_SOA_COLUMN(TPCNClsFindableMinusFound, tpcNClsFindableMinusFound, int8_t);             //! TPC Clusters: Findable - Found
DECLARE_SOA_COLUMN(TPCNClsFindableMinusPID, tpcNClsFindableMinusPID, int8_t);                 //! TPC Clusters: Findable - Found clusters used for PID
DECLARE_SOA_COLUMN(TPCNClsFindableMinusCrossedRows, tpcNClsFindableMinusCrossedRows, int8_t); //! TPC Clusters: Findable - crossed rows
DECLARE_SOA_COLUMN(TPCNClsShared, tpcNClsShared, uint8_t);                                    //! Number of shared TPC clusters
DECLARE_SOA_COLUMN(ITSSharedClusterMap, itsSharedClusterMap, uint8_t);                        //! shared ITS cluster map (Run 2)
DECLARE_SOA_COLUMN(TRDPattern, trdPattern, uint8_t);                                          //! Contributor to the track on TRD layer in bits 0-5, starting from the innermost, bit 6 indicates a potentially split tracklet, bit 7 if the track crossed a padrow
DECLARE_SOA_COLUMN(ITSChi2NCl, itsChi2NCl, float);                                            //! Chi2 / cluster for the ITS track segment
DECLARE_SOA_COLUMN(TPCChi2NCl, tpcChi2NCl, float);                                            //! Chi2 / cluster for the TPC track segment
DECLARE_SOA_COLUMN(TRDChi2, trdChi2, float);                                                  //! Chi2 for the TRD track segment
DECLARE_SOA_COLUMN(TOFChi2, tofChi2, float);                                                  //! Chi2 for the TOF track segment
DECLARE_SOA_COLUMN(ITSSignal, itsSignal, float);                                              //! dE/dx signal in the ITS (Run 2)
DECLARE_SOA_COLUMN(TPCSignal, tpcSignal, float);                                              //! dE/dx signal in the TPC
DECLARE_SOA_COLUMN(TRDSignal, trdSignal, float);                                              //! PID signal in the TRD
DECLARE_SOA_COLUMN(Length, length, float);                                                    //! Track length
DECLARE_SOA_COLUMN(TOFExpMom, tofExpMom, float);                                              //! TOF expected momentum obtained in tracking, used to compute the expected times
DECLARE_SOA_COLUMN(TrackEtaEMCAL, trackEtaEmcal, float);                                      //!
DECLARE_SOA_COLUMN(TrackPhiEMCAL, trackPhiEmcal, float);                                      //!
DECLARE_SOA_COLUMN(TrackTime, trackTime, float);                                              //! Estimated time of the track in ns wrt collision().bc() or ambiguoustrack.bcSlice()[0]
DECLARE_SOA_COLUMN(TrackTimeRes, trackTimeRes, float);                                        //! Resolution of the track time in ns (see TrackFlags::TrackTimeResIsRange)

// expression columns changing between versions have to be declared in different namespaces

DECLARE_SOA_EXPRESSION_COLUMN(DetectorMap, detectorMap, uint8_t, //! Detector map: see enum DetectorMapEnum
                              ifnode(aod::track::itsClusterMap > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::ITS), (uint8_t)0x0) |
                                ifnode(aod::track::tpcNClsFindable > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TPC), (uint8_t)0x0) |
                                ifnode(aod::track::trdPattern > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TRD), (uint8_t)0x0) |
                                ifnode((aod::track::tofChi2 >= 0.f) && (aod::track::tofExpMom > 0.f), static_cast<uint8_t>(o2::aod::track::TOF), (uint8_t)0x0));

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeEl, tofExpTimeEl, //! Expected time for the track to reach the TOF under the electron hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassElectron * o2::constants::physics::MassElectron;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeMu, tofExpTimeMu, //! Expected time for the track to reach the TOF under the muon hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassMuon * o2::constants::physics::MassMuon;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimePi, tofExpTimePi, //! Expected time for the track to reach the TOF under the pion hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassPionCharged * o2::constants::physics::MassPionCharged;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeKa, tofExpTimeKa, //! Expected time for the track to reach the TOF under the kaon hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassKaonCharged * o2::constants::physics::MassKaonCharged;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimePr, tofExpTimePr, //! Expected time for the track to reach the TOF under the proton hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassProton * o2::constants::physics::MassProton;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeDe, tofExpTimeDe, //! Expected time for the track to reach the TOF under the deuteron hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassDeuteron * o2::constants::physics::MassDeuteron;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeTr, tofExpTimeTr, //! Expected time for the track to reach the TOF under the triton hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassTriton * o2::constants::physics::MassTriton;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeHe, tofExpTimeHe, //! Expected time for the track to reach the TOF under the helium3 hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassHelium3 * o2::constants::physics::MassHelium3;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TOFExpTimeAl, tofExpTimeAl, //! Expected time for the track to reach the TOF under the helium4 hypothesis
                           [](float length, float tofExpMom) -> float {
                             constexpr float massSquared = o2::constants::physics::MassAlpha * o2::constants::physics::MassAlpha;
                             return o2::framework::pid::tof::MassToExpTime(tofExpMom, length, massSquared);
                           });

namespace v001
{
DECLARE_SOA_EXPRESSION_COLUMN(DetectorMap, detectorMap, uint8_t, //! Detector map version 1, see enum DetectorMapEnum
                              ifnode(aod::track::itsClusterSizes > (uint32_t)0, static_cast<uint8_t>(o2::aod::track::ITS), (uint8_t)0x0) |
                                ifnode(aod::track::tpcNClsFindable > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TPC), (uint8_t)0x0) |
                                ifnode(aod::track::trdPattern > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TRD), (uint8_t)0x0) |
                                ifnode((aod::track::tofChi2 >= 0.f) && (aod::track::tofExpMom > 0.f), static_cast<uint8_t>(o2::aod::track::TOF), (uint8_t)0x0));
DECLARE_SOA_DYNAMIC_COLUMN(ITSClusterMap, itsClusterMap, //! ITS cluster map, one bit per a layer, starting from the innermost
                           [](uint32_t itsClusterSizes) -> uint8_t {
                             uint8_t clmap = 0;
                             for (unsigned int layer = 0; layer < 7; layer++) {
                               if ((itsClusterSizes >> (layer * 4)) & 0xf) {
                                 clmap |= (1 << layer);
                               }
                             }
                             return clmap;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNCls, itsNCls, //! Number of ITS clusters
                           [](uint32_t itsClusterSizes) -> uint8_t {
                             uint8_t itsNcls = 0;
                             for (int layer = 0; layer < 7; layer++) {
                               if ((itsClusterSizes >> (layer * 4)) & 0xf)
                                 itsNcls++;
                             }
                             return itsNcls;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNClsInnerBarrel, itsNClsInnerBarrel, //! Number of ITS clusters in the Inner Barrel
                           [](uint32_t itsClusterSizes) -> uint8_t {
                             uint8_t itsNclsInnerBarrel = 0;
                             for (int layer = 0; layer < 3; layer++) {
                               if ((itsClusterSizes >> (layer * 4)) & 0xf)
                                 itsNclsInnerBarrel++;
                             }
                             return itsNclsInnerBarrel;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSClsSizeInLayer, itsClsSizeInLayer, //! Size of the ITS cluster in a given layer
                           [](uint32_t itsClusterSizes, int layer) -> uint8_t {
                             if (layer >= 7 || layer < 0) {
                               return 0;
                             }
                             return (itsClusterSizes >> (layer * 4)) & 0xf;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(IsITSAfterburner, isITSAfterburner, //! If the track used the afterburner in the ITS
                           [](uint8_t detectorMap, float itsChi2Ncl) -> bool {
                             return (detectorMap & o2::aod::track::ITS) && (itsChi2Ncl < 0.f);
                           });

namespace extensions
{
using TPCTimeErrEncoding = o2::aod::track::extensions::TPCTimeErrEncoding;
DECLARE_SOA_DYNAMIC_COLUMN(TPCDeltaTFwd, tpcDeltaTFwd, //! Delta Forward of track time in TPC time bis
                           [](float timeErr, uint32_t trackType) -> float {
                             if (!(trackType & TrackFlags::TrackTimeAsym)) {
                               return TPCTimeErrEncoding::invalidValue;
                             }
                             TPCTimeErrEncoding enc;
                             enc.encoding.timeErr = timeErr;
                             return enc.getDeltaTFwd();
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCDeltaTBwd, tpcDeltaTBwd, //! Delta Backward of track time in TPC time bis
                           [](float timeErr, uint32_t trackType) -> float {
                             if (!(trackType & TrackFlags::TrackTimeAsym)) {
                               return TPCTimeErrEncoding::invalidValue;
                             }
                             TPCTimeErrEncoding p;
                             p.encoding.timeErr = timeErr;
                             return p.getDeltaTBwd();
                           });
} // namespace extensions

} // namespace v001

DECLARE_SOA_DYNAMIC_COLUMN(HasITS, hasITS, //! Flag to check if track has a ITS match
                           [](uint8_t detectorMap) -> bool { return detectorMap & o2::aod::track::ITS; });
DECLARE_SOA_DYNAMIC_COLUMN(HasTPC, hasTPC, //! Flag to check if track has a TPC match
                           [](uint8_t detectorMap) -> bool { return detectorMap & o2::aod::track::TPC; });
DECLARE_SOA_DYNAMIC_COLUMN(HasTRD, hasTRD, //! Flag to check if track has a TRD match
                           [](uint8_t detectorMap) -> bool { return detectorMap & o2::aod::track::TRD; });
DECLARE_SOA_DYNAMIC_COLUMN(HasTOF, hasTOF, //! Flag to check if track has a TOF measurement
                           [](uint8_t detectorMap) -> bool { return detectorMap & o2::aod::track::TOF; });
DECLARE_SOA_DYNAMIC_COLUMN(IsPVContributor, isPVContributor, //! Run 3: Has this track contributed to the collision vertex fit
                           [](uint8_t flags) -> bool { return (flags & o2::aod::track::PVContributor) == o2::aod::track::PVContributor; });
DECLARE_SOA_DYNAMIC_COLUMN(PIDForTracking, pidForTracking, //! PID hypothesis used during tracking. See the constants in the class PID in PID.h
                           [](uint32_t flags) -> uint32_t { return flags >> 28; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsFound, tpcNClsFound, //! Number of found TPC clusters
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusFound) -> int16_t { return (int16_t)tpcNClsFindable - tpcNClsFindableMinusFound; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsPID, tpcNClsPID, //! Number of found TPC clusters used for PID
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusPID) -> int16_t { return (int16_t)tpcNClsFindable - tpcNClsFindableMinusPID; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsCrossedRows, tpcNClsCrossedRows, //! Number of crossed TPC Rows
                           [](uint8_t tpcNClsFindable, int8_t TPCNClsFindableMinusCrossedRows) -> int16_t { return (int16_t)tpcNClsFindable - TPCNClsFindableMinusCrossedRows; });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNCls, itsNCls, //! Number of ITS clusters
                           [](uint8_t itsClusterMap) -> uint8_t {
                             uint8_t itsNcls = 0;
                             constexpr uint8_t bit = 1;
                             for (int layer = 0; layer < 7; layer++) {
                               if (itsClusterMap & (bit << layer))
                                 itsNcls++;
                             }
                             return itsNcls;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNClsInnerBarrel, itsNClsInnerBarrel, //! Number of ITS clusters in the Inner Barrel
                           [](uint8_t itsClusterMap) -> uint8_t {
                             uint8_t itsNclsInnerBarrel = 0;
                             constexpr uint8_t bit = 1;
                             for (int layer = 0; layer < 3; layer++) {
                               if (itsClusterMap & (bit << layer))
                                 itsNclsInnerBarrel++;
                             }
                             return itsNclsInnerBarrel;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNSharedCls, itsNSharedCls, //! Number of shared ITS clusters (Run 2)
                           [](uint8_t itsSharedClusterMap) -> uint8_t {
                             uint8_t itsNSharedCls = 0;
                             constexpr uint8_t bit = 1;
                             for (int layer = 0; layer < 6; layer++) { // ITS1: 6 layers
                               if (itsSharedClusterMap & (bit << layer))
                                 itsNSharedCls++;
                             }
                             return itsNSharedCls;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(TPCFoundOverFindableCls, tpcFoundOverFindableCls, //! Ratio of found over findable clusters
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusFound) -> float {
                             int16_t tpcNClsFound = (int16_t)tpcNClsFindable - tpcNClsFindableMinusFound;
                             return (float)tpcNClsFound / (float)tpcNClsFindable;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCCrossedRowsOverFindableCls, tpcCrossedRowsOverFindableCls, //! Ratio  crossed rows over findable clusters
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusCrossedRows) -> float {
                             int16_t tpcNClsCrossedRows = (int16_t)tpcNClsFindable - tpcNClsFindableMinusCrossedRows;
                             return (float)tpcNClsCrossedRows / (float)tpcNClsFindable;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCFractionSharedCls, tpcFractionSharedCls, //! Fraction of shared TPC clusters
                           [](uint8_t tpcNClsShared, uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusFound) -> float {
                             int16_t tpcNClsFound = (int16_t)tpcNClsFindable - tpcNClsFindableMinusFound;
                             return (float)tpcNClsShared / (float)tpcNClsFound;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TRDHasNeighbor, trdPattern, //! Flag to check if at least one tracklet of a TRD Track has a neighboring tracklet
                           [](uint8_t trdPattern) -> bool { return trdPattern & o2::aod::track::HasNeighbor; });

DECLARE_SOA_DYNAMIC_COLUMN(TRDHasCrossing, trdPattern, //! Flag to check if at least one tracklet of a TRD Track crossed a padrow
                           [](uint8_t trdPattern) -> bool { return trdPattern & o2::aod::track::HasCrossing; });

DECLARE_SOA_DYNAMIC_COLUMN(TRDNLayers, trdPattern, //! Number of TRD tracklets in a Track
                           [](uint8_t trdPattern) -> std::size_t { return std::bitset<6>(trdPattern).count(); });
} // namespace track

DECLARE_SOA_TABLE_FULL(StoredTracks, "Tracks", "AOD", "TRACK", //! On disk version of the track parameters at collision vertex
                       o2::soa::Index<>, track::CollisionId, track::TrackType,
                       track::X, track::Alpha,
                       track::Y, track::Z, track::Snp, track::Tgl,
                       track::Signed1Pt, track::IsWithinBeamPipe<track::X>,
                       track::Px<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Py<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Pz<track::Signed1Pt, track::Tgl>,
                       track::PVector<track::Signed1Pt, track::Snp, track::Alpha, track::Tgl>,
                       track::Energy<track::Signed1Pt, track::Tgl>,
                       track::Rapidity<track::Signed1Pt, track::Tgl>,
                       track::Sign<track::Signed1Pt>,
                       o2::soa::Marker<1>);

DECLARE_SOA_EXTENDED_TABLE(Tracks, StoredTracks, "EXTRACK", 0, //! Track parameters at collision vertex
                           aod::track::Pt,
                           aod::track::P,
                           aod::track::Eta,
                           aod::track::Phi);

DECLARE_SOA_TABLE_FULL(StoredTracksIU, "Tracks_IU", "AOD", "TRACK_IU", //! On disk version of the track parameters at inner most update (e.g. ITS) as it comes from the tracking
                       o2::soa::Index<>, track::CollisionId, track::TrackType,
                       track::X, track::Alpha,
                       track::Y, track::Z, track::Snp, track::Tgl,
                       track::Signed1Pt, track::IsWithinBeamPipe<track::X>,
                       track::Px<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Py<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Pz<track::Signed1Pt, track::Tgl>,
                       track::PVector<track::Signed1Pt, track::Snp, track::Alpha, track::Tgl>,
                       track::Energy<track::Signed1Pt, track::Tgl>,
                       track::Rapidity<track::Signed1Pt, track::Tgl>,
                       track::Sign<track::Signed1Pt>,
                       o2::soa::Marker<2>);

DECLARE_SOA_EXTENDED_TABLE(TracksIU, StoredTracksIU, "EXTRACK_IU", 0, //! Track parameters at inner most update (e.g. ITS) as it comes from the tracking
                           aod::track::Pt,
                           aod::track::P,
                           aod::track::Eta,
                           aod::track::Phi);

DECLARE_SOA_TABLE_FULL(StoredTracksCov, "TracksCov", "AOD", "TRACKCOV", //! On disk version of the TracksCov table at collision vertex
                       track::SigmaY, track::SigmaZ, track::SigmaSnp, track::SigmaTgl, track::Sigma1Pt,
                       track::RhoZY, track::RhoSnpY, track::RhoSnpZ, track::RhoTglY, track::RhoTglZ,
                       track::RhoTglSnp, track::Rho1PtY, track::Rho1PtZ, track::Rho1PtSnp, track::Rho1PtTgl, o2::soa::Marker<1>);

DECLARE_SOA_EXTENDED_TABLE(TracksCov, StoredTracksCov, "EXTRACKCOV", 0, //! Track covariance matrix at collision vertex
                           aod::track::CYY,
                           aod::track::CZY,
                           aod::track::CZZ,
                           aod::track::CSnpY,
                           aod::track::CSnpZ,
                           aod::track::CSnpSnp,
                           aod::track::CTglY,
                           aod::track::CTglZ,
                           aod::track::CTglSnp,
                           aod::track::CTglTgl,
                           aod::track::C1PtY,
                           aod::track::C1PtZ,
                           aod::track::C1PtSnp,
                           aod::track::C1PtTgl,
                           aod::track::C1Pt21Pt2);

DECLARE_SOA_TABLE_FULL(StoredTracksCovIU, "TracksCov_IU", "AOD", "TRACKCOV_IU", //! On disk version of the TracksCov table at inner most update (e.g. ITS) as it comes from the tracking
                       track::SigmaY, track::SigmaZ, track::SigmaSnp, track::SigmaTgl, track::Sigma1Pt,
                       track::RhoZY, track::RhoSnpY, track::RhoSnpZ, track::RhoTglY, track::RhoTglZ,
                       track::RhoTglSnp, track::Rho1PtY, track::Rho1PtZ, track::Rho1PtSnp, track::Rho1PtTgl, o2::soa::Marker<2>);

DECLARE_SOA_EXTENDED_TABLE(TracksCovIU, StoredTracksCovIU, "EXTRACKCOV_IU", 0, //! Track covariance matrix at inner most update (e.g. ITS) as it comes from the tracking
                           aod::track::CYY,
                           aod::track::CZY,
                           aod::track::CZZ,
                           aod::track::CSnpY,
                           aod::track::CSnpZ,
                           aod::track::CSnpSnp,
                           aod::track::CTglY,
                           aod::track::CTglZ,
                           aod::track::CTglSnp,
                           aod::track::CTglTgl,
                           aod::track::C1PtY,
                           aod::track::C1PtZ,
                           aod::track::C1PtSnp,
                           aod::track::C1PtTgl,
                           aod::track::C1Pt21Pt2);

DECLARE_SOA_TABLE_FULL(StoredTracksExtra_000, "TracksExtra", "AOD", "TRACKEXTRA", //! On disk version of TracksExtra, version 0
                       track::TPCInnerParam, track::Flags, track::ITSClusterMap,
                       track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                       track::TPCNClsShared, track::TRDPattern, track::ITSChi2NCl,
                       track::TPCChi2NCl, track::TRDChi2, track::TOFChi2,
                       track::TPCSignal, track::TRDSignal, track::Length, track::TOFExpMom,
                       track::PIDForTracking<track::Flags>,
                       track::IsPVContributor<track::Flags>,
                       track::HasITS<track::DetectorMap>, track::HasTPC<track::DetectorMap>,
                       track::HasTRD<track::DetectorMap>, track::HasTOF<track::DetectorMap>,
                       track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                       track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                       track::TOFExpTimeEl<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeMu<track::Length, track::TOFExpMom>,
                       track::TOFExpTimePi<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeKa<track::Length, track::TOFExpMom>,
                       track::TOFExpTimePr<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeDe<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeTr<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeHe<track::Length, track::TOFExpMom>,
                       track::TOFExpTimeAl<track::Length, track::TOFExpMom>,
                       track::ITSNCls<track::ITSClusterMap>, track::ITSNClsInnerBarrel<track::ITSClusterMap>,
                       track::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                       track::TPCFoundOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                       track::TPCFractionSharedCls<track::TPCNClsShared, track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                       track::TrackEtaEMCAL, track::TrackPhiEMCAL, track::TrackTime, track::TrackTimeRes);

DECLARE_SOA_TABLE_FULL_VERSIONED(StoredTracksExtra_001, "TracksExtra", "AOD", "TRACKEXTRA", 1, // On disk version of TracksExtra, version 1
                                 track::TPCInnerParam, track::Flags, track::ITSClusterSizes,
                                 track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                                 track::TPCNClsShared, track::v001::extensions::TPCDeltaTFwd<track::TrackTimeRes, track::Flags>, track::v001::extensions::TPCDeltaTBwd<track::TrackTimeRes, track::Flags>,
                                 track::TRDPattern, track::ITSChi2NCl, track::TPCChi2NCl, track::TRDChi2, track::TOFChi2,
                                 track::TPCSignal, track::TRDSignal, track::Length, track::TOFExpMom,
                                 track::PIDForTracking<track::Flags>,
                                 track::IsPVContributor<track::Flags>,
                                 track::HasITS<track::v001::DetectorMap>, track::HasTPC<track::v001::DetectorMap>,
                                 track::HasTRD<track::v001::DetectorMap>, track::HasTOF<track::v001::DetectorMap>,
                                 track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                                 track::v001::ITSClusterMap<track::ITSClusterSizes>, track::v001::ITSNCls<track::ITSClusterSizes>, track::v001::ITSNClsInnerBarrel<track::ITSClusterSizes>,
                                 track::v001::ITSClsSizeInLayer<track::ITSClusterSizes>,
                                 track::v001::IsITSAfterburner<track::v001::DetectorMap, track::ITSChi2NCl>,
                                 track::TOFExpTimeEl<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeMu<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimePi<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeKa<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimePr<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeDe<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeTr<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeHe<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeAl<track::Length, track::TOFExpMom>,
                                 track::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                                 track::TPCFoundOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TPCFractionSharedCls<track::TPCNClsShared, track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TrackEtaEMCAL, track::TrackPhiEMCAL, track::TrackTime, track::TrackTimeRes);

DECLARE_SOA_TABLE_FULL_VERSIONED(StoredTracksExtra_002, "TracksExtra", "AOD", "TRACKEXTRA", 2, // On disk version of TracksExtra, version 2
                                 track::TPCInnerParam, track::Flags, track::ITSClusterSizes,
                                 track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusPID, track::TPCNClsFindableMinusCrossedRows,
                                 track::TPCNClsShared, track::v001::extensions::TPCDeltaTFwd<track::TrackTimeRes, track::Flags>, track::v001::extensions::TPCDeltaTBwd<track::TrackTimeRes, track::Flags>,
                                 track::TRDPattern, track::ITSChi2NCl, track::TPCChi2NCl, track::TRDChi2, track::TOFChi2,
                                 track::TPCSignal, track::TRDSignal, track::Length, track::TOFExpMom,
                                 track::PIDForTracking<track::Flags>,
                                 track::IsPVContributor<track::Flags>,
                                 track::HasITS<track::v001::DetectorMap>, track::HasTPC<track::v001::DetectorMap>,
                                 track::HasTRD<track::v001::DetectorMap>, track::HasTOF<track::v001::DetectorMap>,
                                 track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                                 track::v001::ITSClusterMap<track::ITSClusterSizes>, track::v001::ITSNCls<track::ITSClusterSizes>, track::v001::ITSNClsInnerBarrel<track::ITSClusterSizes>,
                                 track::v001::ITSClsSizeInLayer<track::ITSClusterSizes>,
                                 track::v001::IsITSAfterburner<track::v001::DetectorMap, track::ITSChi2NCl>,
                                 track::TOFExpTimeEl<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeMu<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimePi<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeKa<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimePr<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeDe<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeTr<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeHe<track::Length, track::TOFExpMom>,
                                 track::TOFExpTimeAl<track::Length, track::TOFExpMom>,
                                 track::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                                 track::TPCFoundOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TPCFractionSharedCls<track::TPCNClsShared, track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                                 track::TrackEtaEMCAL, track::TrackPhiEMCAL, track::TrackTime, track::TrackTimeRes);

DECLARE_SOA_EXTENDED_TABLE(TracksExtra_000, StoredTracksExtra_000, "EXTRACKEXTRA", 0, //! Additional track information (clusters, PID, etc.)
                           track::DetectorMap);
DECLARE_SOA_EXTENDED_TABLE(TracksExtra_001, StoredTracksExtra_001, "EXTRACKEXTRA", 1, //! Additional track information (clusters, PID, etc.)
                           track::v001::DetectorMap);
DECLARE_SOA_EXTENDED_TABLE(TracksExtra_002, StoredTracksExtra_002, "EXTRACKEXTRA", 2, //! Additional track information (clusters, PID, etc.)
                           track::v001::DetectorMap);

DECLARE_SOA_TABLE(Run2TrackExtras_000, "AOD", "RUN2TRACKEXTRA",
                  track::ITSSignal);
DECLARE_SOA_TABLE_VERSIONED(Run2TrackExtras_001, "AOD", "RUN2TRACKEXTRA", 1, //! adds ITS shared cluster map
                            track::ITSSignal, track::ITSSharedClusterMap,
                            track::ITSNSharedCls<track::ITSSharedClusterMap>);

using StoredTracksExtra = StoredTracksExtra_002;
using TracksExtra = TracksExtra_002;

using Track = Tracks::iterator;
using TrackIU = TracksIU::iterator;
using TrackCov = TracksCov::iterator;
using TrackCovIU = TracksCovIU::iterator;
using TrackExtra = TracksExtra::iterator;
using Run2TrackExtras = Run2TrackExtras_001;
using Run2TrackExtra = Run2TrackExtras::iterator;

} // namespace aod
namespace soa
{
extern template struct soa::JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::Tracks, aod::TracksExtra>;
extern template struct soa::JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::Tracks, aod::TracksCov, aod::TracksExtra>;
extern template struct soa::JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::TracksExtension, aod::StoredTracks>;
} // namespace soa
namespace aod
{
using FullTracks = soa::Join<Tracks, TracksExtra>;
using FullTrack = FullTracks::iterator;

namespace trackqa
{
// TRACKQA TABLE COLUMNS
DECLARE_SOA_INDEX_COLUMN(Track, track);                                   //! track to which this QA information belongs
DECLARE_SOA_COLUMN(TPCTime0, tpcTime0, float);                            //! tpc only time0 (mTime0 in TPC track)
DECLARE_SOA_COLUMN(TPCDCAR, tpcdcaR, int16_t);                            //! tpc only DCAr
DECLARE_SOA_COLUMN(TPCDCAZ, tpcdcaZ, int16_t);                            //! tpc only DCAz
DECLARE_SOA_COLUMN(TPCClusterByteMask, tpcClusterByteMask, uint8_t);      //! tracklet bitmask - track defining 8 tracklets (152=8*19 rows) bit set if nCluster>thr (default 5)
DECLARE_SOA_COLUMN(TPCdEdxMax0R, tpcdEdxMax0R, uint8_t);                  //! TPC dEdxQMax -ROC0/dEdx
DECLARE_SOA_COLUMN(TPCdEdxMax1R, tpcdEdxMax1R, uint8_t);                  //! TPC dEdxQMax -ROC1/dEdx
DECLARE_SOA_COLUMN(TPCdEdxMax2R, tpcdEdxMax2R, uint8_t);                  //! TPC dEdxQMax -ROC2/dEdx
DECLARE_SOA_COLUMN(TPCdEdxMax3R, tpcdEdxMax3R, uint8_t);                  //! TPC dEdxQMax -ROC3/dEdx
DECLARE_SOA_COLUMN(TPCdEdxTot0R, tpcdEdxTot0R, uint8_t);                  //! TPC dEdxQtot -ROC0/dEdx
DECLARE_SOA_COLUMN(TPCdEdxTot1R, tpcdEdxTot1R, uint8_t);                  //! TPC dEdxQtot -ROC1/dEdx
DECLARE_SOA_COLUMN(TPCdEdxTot2R, tpcdEdxTot2R, uint8_t);                  //! TPC dEdxQtot -ROC2/dEdx
DECLARE_SOA_COLUMN(TPCdEdxTot3R, tpcdEdxTot3R, uint8_t);                  //! TPC dEdxQtot -ROC3/dEdx
DECLARE_SOA_COLUMN(DeltaRefContParamY, deltaRefContParamY, int8_t);       //! Normalized delta of contributor tracks at reference point in the same frame Y
DECLARE_SOA_COLUMN(DeltaRefContParamZ, deltaRefITSParamZ, int8_t);        //! Normalized delta of contributor tracks at reference point in the same frame Z
DECLARE_SOA_COLUMN(DeltaRefContParamSnp, deltaRefContParamSnp, int8_t);   //! Normalized delta of contributor tracks at reference point in the same frame Snp
DECLARE_SOA_COLUMN(DeltaRefContParamTgl, deltaRefContParamTgl, int8_t);   //! Normalized delta of contributor tracks at reference point in the same frame Tgl
DECLARE_SOA_COLUMN(DeltaRefContParamQ2Pt, deltaRefContParamQ2Pt, int8_t); //! Normalized delta of contributor tracks at reference point in the same frame Q2Pt
DECLARE_SOA_COLUMN(DeltaRefGloParamY, deltaRefGloParamY, int8_t);         //! Normalized delta of global track to average contributors matched tracks at reference point in the same frame Y
DECLARE_SOA_COLUMN(DeltaRefGloParamZ, deltaRefGloParamZ, int8_t);         //! Normalized delta of global track to average contributors matched tracks at reference point in the same frame Z
DECLARE_SOA_COLUMN(DeltaRefGloParamSnp, deltaRefGloParamSnp, int8_t);     //! Normalized delta of global track to average contributors matched tracks at reference point in the same frame Snp
DECLARE_SOA_COLUMN(DeltaRefGloParamTgl, deltaRefGloParamTgl, int8_t);     //! Normalized delta of global track to average contributors matched tracks at reference point in the same frame Tgl
DECLARE_SOA_COLUMN(DeltaRefGloParamQ2Pt, deltaRefGloParamQ2Pt, int8_t);   //! Normalized delta of global track to average contributors matched tracks at reference point in the same frame Q2Pt
DECLARE_SOA_COLUMN(DeltaTOFdX, deltaTOFdX, int8_t);                       //!
DECLARE_SOA_COLUMN(DeltaTOFdZ, deltaTOFdZ, int8_t);                       //!

DECLARE_SOA_DYNAMIC_COLUMN(IsDummy, isDummy, //! indicates if the propagation of the contrib. tracks was successful and residuals are available
                           [](int8_t cY, int8_t cZ, int8_t cSnp, int8_t cTgl, int8_t cQ2Pt, int8_t gY, int8_t gZ, int8_t gSnp, int8_t gTgl, int8_t gQ2Pt) -> bool {
                             constexpr int8_t m = std::numeric_limits<int8_t>::min();
                             return (cY == m && cZ == m && cSnp == m && cTgl == m && cQ2Pt == m && gY == m && gZ == m && gSnp == m && gTgl == m && gQ2Pt == m);
                           });
} // namespace trackqa

DECLARE_SOA_TABLE(TracksQA_000, "AOD", "TRACKQA", //! trackQA information - sampled QA information currently for the TPC - version 0
                  o2::soa::Index<>, trackqa::TrackId, trackqa::TPCTime0, trackqa::TPCDCAR, trackqa::TPCDCAZ, trackqa::TPCClusterByteMask,
                  trackqa::TPCdEdxMax0R, trackqa::TPCdEdxMax1R, trackqa::TPCdEdxMax2R, trackqa::TPCdEdxMax3R,
                  trackqa::TPCdEdxTot0R, trackqa::TPCdEdxTot1R, trackqa::TPCdEdxTot2R, trackqa::TPCdEdxTot3R);
//                  o2::soa::Index<>, trackqa::TrackId, trackqa::TPCDCAR, trackqa::TPCDCAZ, trackqa::TPCClusterByteMask,

DECLARE_SOA_TABLE_VERSIONED(TracksQA_001, "AOD", "TRACKQA", 1, //! trackQA information - version 1 - including contributor residuals of matched tracks at reference radius
                            o2::soa::Index<>, trackqa::TrackId, trackqa::TPCTime0, trackqa::TPCDCAR, trackqa::TPCDCAZ, trackqa::TPCClusterByteMask,
                            trackqa::TPCdEdxMax0R, trackqa::TPCdEdxMax1R, trackqa::TPCdEdxMax2R, trackqa::TPCdEdxMax3R,
                            trackqa::TPCdEdxTot0R, trackqa::TPCdEdxTot1R, trackqa::TPCdEdxTot2R, trackqa::TPCdEdxTot3R,
                            trackqa::DeltaRefContParamY, trackqa::DeltaRefContParamZ, trackqa::DeltaRefContParamSnp, trackqa::DeltaRefContParamTgl, trackqa::DeltaRefContParamQ2Pt,
                            trackqa::DeltaRefGloParamY, trackqa::DeltaRefGloParamZ, trackqa::DeltaRefGloParamSnp, trackqa::DeltaRefGloParamTgl, trackqa::DeltaRefGloParamQ2Pt,
                            trackqa::IsDummy<trackqa::DeltaRefContParamY, trackqa::DeltaRefContParamZ, trackqa::DeltaRefContParamSnp, trackqa::DeltaRefContParamTgl, trackqa::DeltaRefContParamQ2Pt,
                                             trackqa::DeltaRefGloParamY, trackqa::DeltaRefGloParamZ, trackqa::DeltaRefGloParamSnp, trackqa::DeltaRefGloParamTgl, trackqa::DeltaRefGloParamQ2Pt>);

DECLARE_SOA_TABLE_VERSIONED(TracksQA_002, "AOD", "TRACKQA", 2, //! trackQA information - version 2 - including contributor residuals of matched tracks at reference radius + TOF delta information
                            o2::soa::Index<>, trackqa::TrackId, trackqa::TPCTime0, trackqa::TPCDCAR, trackqa::TPCDCAZ, trackqa::TPCClusterByteMask,
                            trackqa::TPCdEdxMax0R, trackqa::TPCdEdxMax1R, trackqa::TPCdEdxMax2R, trackqa::TPCdEdxMax3R,
                            trackqa::TPCdEdxTot0R, trackqa::TPCdEdxTot1R, trackqa::TPCdEdxTot2R, trackqa::TPCdEdxTot3R,
                            trackqa::DeltaRefContParamY, trackqa::DeltaRefContParamZ, trackqa::DeltaRefContParamSnp, trackqa::DeltaRefContParamTgl, trackqa::DeltaRefContParamQ2Pt,
                            trackqa::DeltaRefGloParamY, trackqa::DeltaRefGloParamZ, trackqa::DeltaRefGloParamSnp, trackqa::DeltaRefGloParamTgl, trackqa::DeltaRefGloParamQ2Pt,
                            trackqa::DeltaTOFdX, trackqa::DeltaTOFdZ,
                            trackqa::IsDummy<trackqa::DeltaRefContParamY, trackqa::DeltaRefContParamZ, trackqa::DeltaRefContParamSnp, trackqa::DeltaRefContParamTgl, trackqa::DeltaRefContParamQ2Pt,
                                             trackqa::DeltaRefGloParamY, trackqa::DeltaRefGloParamZ, trackqa::DeltaRefGloParamSnp, trackqa::DeltaRefGloParamTgl, trackqa::DeltaRefGloParamQ2Pt>);

using TracksQAVersion = TracksQA_002;
using TracksQA = TracksQAVersion::iterator;

namespace fwdtrack
{
// FwdTracks and MFTTracks Columns definitions
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                                              //!
DECLARE_SOA_COLUMN(TrackType, trackType, uint8_t);                                           //! Type of track. See enum ForwardTrackTypeEnum
DECLARE_SOA_COLUMN(X, x, float);                                                             //! TrackParFwd parameter x
DECLARE_SOA_COLUMN(Y, y, float);                                                             //! TrackParFwd parameter y
DECLARE_SOA_COLUMN(Z, z, float);                                                             //! TrackParFwd propagation parameter z
DECLARE_SOA_COLUMN(Phi, phi, float);                                                         //! TrackParFwd parameter phi; (i.e. pt pointing direction)
DECLARE_SOA_COLUMN(Tgl, tgl, float);                                                         //! TrackParFwd parameter tan(\lamba); (\lambda = 90 - \theta_{polar})
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float);                                             //! TrackParFwd parameter: charged inverse transverse momentum; (q/pt)
DECLARE_SOA_COLUMN(NClusters, nClusters, int8_t);                                            //! Number of clusters
DECLARE_SOA_COLUMN(MFTClusterSizesAndTrackFlags, mftClusterSizesAndTrackFlags, uint64_t);    //! Cluster sizes per track, stored per layer (each 6 bits). Remaining 4 bits for MFT flags
DECLARE_SOA_COLUMN(Chi2, chi2, float);                                                       //! Track chi^2
DECLARE_SOA_COLUMN(PDca, pDca, float);                                                       //! PDca for MUONStandalone
DECLARE_SOA_COLUMN(RAtAbsorberEnd, rAtAbsorberEnd, float);                                   //! RAtAbsorberEnd for MUONStandalone tracks and GlobalMuonTrackstracks
DECLARE_SOA_COLUMN(Chi2MatchMCHMID, chi2MatchMCHMID, float);                                 //! MCH-MID Match Chi2 for MUONStandalone tracks
DECLARE_SOA_COLUMN(Chi2MatchMCHMFT, chi2MatchMCHMFT, float);                                 //! MCH-MFT Match Chi2 for GlobalMuonTracks
DECLARE_SOA_COLUMN(MatchScoreMCHMFT, matchScoreMCHMFT, float);                               //! MCH-MFT Machine Learning Matching Score for GlobalMuonTracks
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(MCHTrack, matchMCHTrack, int, "FwdTracks_MatchMCHTrack"); //! Index of matching MCH track for GlobalMuonTracks and GlobalForwardTracks
DECLARE_SOA_COLUMN(MCHBitMap, mchBitMap, uint16_t);                                          //! Fired muon trackig chambers bitmap
DECLARE_SOA_COLUMN(MIDBitMap, midBitMap, uint8_t);                                           //! MID bitmap: non-bending plane (4bit), bending plane (4bit)
DECLARE_SOA_COLUMN(MIDBoards, midBoards, uint32_t);                                          //! Local boards on each MID plane (8 bits per plane)
DECLARE_SOA_COLUMN(TrackTime, trackTime, float);                                             //! Estimated time of the track in ns wrt collision().bc() or ambiguoustrack.bcSlice()[0]
DECLARE_SOA_COLUMN(TrackTimeRes, trackTimeRes, float);                                       //! Resolution of the track time in ns
DECLARE_SOA_DYNAMIC_COLUMN(Sign, sign,                                                       //! Sign of the track eletric charge
                           [](float signed1Pt) -> short { return (signed1Pt > 0) ? 1 : -1; });
DECLARE_SOA_DYNAMIC_COLUMN(IsCA, isCA, //! Returns true if used track-finding algorithm was Cellular Automaton
                           [](uint64_t mftClusterSizesAndTrackFlags) -> bool { return mftClusterSizesAndTrackFlags & (0x1ULL << 60); });
DECLARE_SOA_EXPRESSION_COLUMN(Eta, eta, float, //!
                              -1.f * nlog(ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::fwdtrack::tgl))));
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //!
                              ifnode(nabs(aod::fwdtrack::signed1Pt) < o2::constants::math::Almost0, o2::constants::math::VeryBig, nabs(1.f / aod::fwdtrack::signed1Pt)));
DECLARE_SOA_EXPRESSION_COLUMN(P, p, float, //!
                              ifnode((nabs(aod::fwdtrack::signed1Pt) < o2::constants::math::Almost0) || (nabs(o2::constants::math::PIQuarter - 0.5f * natan(aod::fwdtrack::tgl)) < o2::constants::math::Almost0), o2::constants::math::VeryBig, 0.5f * (ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::fwdtrack::tgl)) + 1.f / ntan(o2::constants::math::PIQuarter - 0.5f * natan(aod::fwdtrack::tgl))) / nabs(aod::fwdtrack::signed1Pt)));
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, //!
                           [](float pt, float phi) -> float {
                             return pt * std::cos(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float {
                             return pt * std::sin(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float tgl) -> float {
                             return pt * tgl;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(MIDBoardCh1, midBoardCh1, //!
                           [](uint32_t midBoards) -> int {
                             return static_cast<int>(midBoards & 0xFF);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(MIDBoardCh2, midBoardCh2, //!
                           [](uint32_t midBoards) -> int {
                             return static_cast<int>((midBoards >> 8) & 0xFF);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(MIDBoardCh3, midBoardCh3, //!
                           [](uint32_t midBoards) -> int {
                             return static_cast<int>((midBoards >> 16) & 0xFF);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(MIDBoardCh4, midBoardCh4, //!
                           [](uint32_t midBoards) -> int {
                             return static_cast<int>((midBoards >> 24) & 0xFF);
                           });

namespace v001
{
DECLARE_SOA_DYNAMIC_COLUMN(NClusters, nClusters, //! Number of MFT clusters
                           [](uint64_t mftClusterSizesAndTrackFlags) -> int8_t {
                             int8_t nClusters = 0;
                             for (int layer = 0; layer < 10; layer++) {
                               if ((mftClusterSizesAndTrackFlags >> (layer * 6)) & 0x3F) {
                                 nClusters++;
                               }
                             }
                             return nClusters;
                           });
} // namespace v001

// FwdTracksCov columns definitions
DECLARE_SOA_COLUMN(SigmaX, sigmaX, float);        //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaY, sigmaY, float);        //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaPhi, sigmaPhi, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(SigmaTgl, sigmaTgl, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(Sigma1Pt, sigma1Pt, float);    //! Covariance matrix
DECLARE_SOA_COLUMN(RhoXY, rhoXY, int8_t);         //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoPhiX, rhoPhiX, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoPhiY, rhoPhiY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglX, rhoTglX, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglY, rhoTglY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(RhoTglPhi, rhoTglPhi, int8_t); //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtX, rho1PtX, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtY, rho1PtY, int8_t);     //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtPhi, rho1PtPhi, int8_t); //! Covariance matrix in compressed form
DECLARE_SOA_COLUMN(Rho1PtTgl, rho1PtTgl, int8_t); //! Covariance matrix in compressed form

DECLARE_SOA_EXPRESSION_COLUMN(CXX, cXX, float, //!
                              aod::fwdtrack::sigmaX* aod::fwdtrack::sigmaX);
DECLARE_SOA_EXPRESSION_COLUMN(CXY, cXY, float, //!
                              (aod::fwdtrack::rhoXY / 128.f) * (aod::fwdtrack::sigmaX * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CYY, cYY, float, //!
                              aod::fwdtrack::sigmaY* aod::fwdtrack::sigmaY);
DECLARE_SOA_EXPRESSION_COLUMN(CPhiX, cPhiX, float, //!
                              (aod::fwdtrack::rhoPhiX / 128.f) * (aod::fwdtrack::sigmaPhi * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(CPhiY, cPhiY, float, //!
                              (aod::fwdtrack::rhoPhiY / 128.f) * (aod::fwdtrack::sigmaPhi * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CPhiPhi, cPhiPhi, float, //!
                              aod::fwdtrack::sigmaPhi* aod::fwdtrack::sigmaPhi);
DECLARE_SOA_EXPRESSION_COLUMN(CTglX, cTglX, float, //!
                              (aod::fwdtrack::rhoTglX / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(CTglY, cTglY, float, //!
                              (aod::fwdtrack::rhoTglY / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CTglPhi, cTglPhi, float, //!
                              (aod::fwdtrack::rhoTglPhi / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaPhi));
DECLARE_SOA_EXPRESSION_COLUMN(CTglTgl, cTglTgl, float, //!
                              aod::fwdtrack::sigmaTgl* aod::fwdtrack::sigmaTgl);
DECLARE_SOA_EXPRESSION_COLUMN(C1PtY, c1PtY, float, //!
                              (aod::fwdtrack::rho1PtY / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtX, c1PtX, float, //!
                              (aod::fwdtrack::rho1PtX / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtPhi, c1PtPhi, float, //!
                              (aod::fwdtrack::rho1PtPhi / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaPhi));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtTgl, c1PtTgl, float, //!
                              (aod::fwdtrack::rho1PtTgl / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaTgl));
DECLARE_SOA_EXPRESSION_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float, //!
                              aod::fwdtrack::sigma1Pt* aod::fwdtrack::sigma1Pt);
} // namespace fwdtrack

// MFTStandalone tracks
DECLARE_SOA_TABLE_FULL(StoredMFTTracks_000, "MFTTracks", "AOD", "MFTTRACK", //! On disk version of MFTTracks, version 0
                       o2::soa::Index<>, fwdtrack::CollisionId,
                       fwdtrack::X, fwdtrack::Y, fwdtrack::Z, fwdtrack::Phi, fwdtrack::Tgl,
                       fwdtrack::Signed1Pt, fwdtrack::NClusters,
                       fwdtrack::Px<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Py<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Pz<fwdtrack::Pt, fwdtrack::Tgl>,
                       fwdtrack::Sign<fwdtrack::Signed1Pt>, fwdtrack::Chi2,
                       fwdtrack::TrackTime, fwdtrack::TrackTimeRes);

DECLARE_SOA_TABLE_FULL_VERSIONED(StoredMFTTracks_001, "MFTTracks", "AOD", "MFTTRACK", 1, //! On disk version of MFTTracks, version 1
                                 o2::soa::Index<>, fwdtrack::CollisionId,
                                 fwdtrack::X, fwdtrack::Y, fwdtrack::Z, fwdtrack::Phi, fwdtrack::Tgl,
                                 fwdtrack::Signed1Pt, fwdtrack::v001::NClusters<fwdtrack::MFTClusterSizesAndTrackFlags>, fwdtrack::MFTClusterSizesAndTrackFlags, fwdtrack::IsCA<fwdtrack::MFTClusterSizesAndTrackFlags>,
                                 fwdtrack::Px<fwdtrack::Pt, fwdtrack::Phi>,
                                 fwdtrack::Py<fwdtrack::Pt, fwdtrack::Phi>,
                                 fwdtrack::Pz<fwdtrack::Pt, fwdtrack::Tgl>,
                                 fwdtrack::Sign<fwdtrack::Signed1Pt>, fwdtrack::Chi2,
                                 fwdtrack::TrackTime, fwdtrack::TrackTimeRes);

DECLARE_SOA_EXTENDED_TABLE(MFTTracks_000, StoredMFTTracks_000, "EXMFTTRACK", 0, //! Additional MFTTracks information (Pt, Eta, P), version 0
                           aod::fwdtrack::Pt,
                           aod::fwdtrack::Eta,
                           aod::fwdtrack::P);

DECLARE_SOA_EXTENDED_TABLE(MFTTracks_001, StoredMFTTracks_001, "EXMFTTRACK", 1, //! Additional MFTTracks information (Pt, Eta, P), version 1
                           aod::fwdtrack::Pt,
                           aod::fwdtrack::Eta,
                           aod::fwdtrack::P);

using MFTTracks = MFTTracks_001;
using StoredMFTTracks = StoredMFTTracks_001;

using MFTTrack = MFTTracks::iterator;

namespace fwdtrack // Index to MFTtrack column must be defined after table definition.
{
DECLARE_SOA_INDEX_COLUMN(MFTTrack, matchMFTTrack); //! ID of matching MFT track for GlobalMuonTracks and GlobalForwardTracks
}

// Tracks including MCH and/or MCH (plus optionally MFT)          //!
DECLARE_SOA_TABLE_FULL(StoredFwdTracks, "FwdTracks", "AOD", "FWDTRACK",
                       o2::soa::Index<>, fwdtrack::CollisionId, fwdtrack::TrackType,
                       fwdtrack::X, fwdtrack::Y, fwdtrack::Z, fwdtrack::Phi, fwdtrack::Tgl,
                       fwdtrack::Signed1Pt, fwdtrack::NClusters, fwdtrack::PDca, fwdtrack::RAtAbsorberEnd,
                       fwdtrack::Px<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Py<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Pz<fwdtrack::Pt, fwdtrack::Tgl>,
                       fwdtrack::Sign<fwdtrack::Signed1Pt>,
                       fwdtrack::Chi2, fwdtrack::Chi2MatchMCHMID, fwdtrack::Chi2MatchMCHMFT,
                       fwdtrack::MatchScoreMCHMFT, fwdtrack::MFTTrackId, fwdtrack::MCHTrackId,
                       fwdtrack::MCHBitMap, fwdtrack::MIDBitMap, fwdtrack::MIDBoards,
                       fwdtrack::TrackTime, fwdtrack::TrackTimeRes);

DECLARE_SOA_EXTENDED_TABLE(FwdTracks, StoredFwdTracks, "EXFWDTRACK", 0, //!
                           aod::fwdtrack::Pt,
                           aod::fwdtrack::Eta,
                           aod::fwdtrack::P);

DECLARE_SOA_TABLE_FULL(StoredFwdTracksCov, "FwdTracksCov", "AOD", "FWDTRACKCOV", //!
                       fwdtrack::SigmaX, fwdtrack::SigmaY, fwdtrack::SigmaPhi, fwdtrack::SigmaTgl, fwdtrack::Sigma1Pt,
                       fwdtrack::RhoXY, fwdtrack::RhoPhiY, fwdtrack::RhoPhiX, fwdtrack::RhoTglX, fwdtrack::RhoTglY,
                       fwdtrack::RhoTglPhi, fwdtrack::Rho1PtX, fwdtrack::Rho1PtY, fwdtrack::Rho1PtPhi, fwdtrack::Rho1PtTgl);

DECLARE_SOA_EXTENDED_TABLE(FwdTracksCov, StoredFwdTracksCov, "EXFWDTRACKCOV", 0, //!
                           aod::fwdtrack::CXX,
                           aod::fwdtrack::CXY,
                           aod::fwdtrack::CYY,
                           aod::fwdtrack::CPhiX,
                           aod::fwdtrack::CPhiY,
                           aod::fwdtrack::CPhiPhi,
                           aod::fwdtrack::CTglX,
                           aod::fwdtrack::CTglY,
                           aod::fwdtrack::CTglPhi,
                           aod::fwdtrack::CTglTgl,
                           aod::fwdtrack::C1PtX,
                           aod::fwdtrack::C1PtY,
                           aod::fwdtrack::C1PtPhi,
                           aod::fwdtrack::C1PtTgl,
                           aod::fwdtrack::C1Pt21Pt2);

using FwdTrack = FwdTracks::iterator;
using FwdTrackCovFwd = FwdTracksCov::iterator;

DECLARE_SOA_TABLE_FULL(StoredMFTTracksCov, "MFTTracksCov", "AOD", "MFTTRACKCOV", //!
                       o2::soa::Index<>, fwdtrack::MFTTrackId,
                       fwdtrack::SigmaX, fwdtrack::SigmaY, fwdtrack::SigmaPhi, fwdtrack::SigmaTgl, fwdtrack::Sigma1Pt,
                       fwdtrack::RhoXY, fwdtrack::RhoPhiX, fwdtrack::RhoPhiY, fwdtrack::RhoTglX, fwdtrack::RhoTglY,
                       fwdtrack::RhoTglPhi, fwdtrack::Rho1PtX, fwdtrack::Rho1PtY, fwdtrack::Rho1PtPhi, fwdtrack::Rho1PtTgl);

DECLARE_SOA_EXTENDED_TABLE(MFTTracksCov, StoredMFTTracksCov, "EXMFTTRACKCOV", 0, //!
                           aod::fwdtrack::CXX,
                           aod::fwdtrack::CXY,
                           aod::fwdtrack::CYY,
                           aod::fwdtrack::CPhiX,
                           aod::fwdtrack::CPhiY,
                           aod::fwdtrack::CPhiPhi,
                           aod::fwdtrack::CTglX,
                           aod::fwdtrack::CTglY,
                           aod::fwdtrack::CTglPhi,
                           aod::fwdtrack::CTglTgl,
                           aod::fwdtrack::C1PtX,
                           aod::fwdtrack::C1PtY,
                           aod::fwdtrack::C1PtPhi,
                           aod::fwdtrack::C1PtTgl,
                           aod::fwdtrack::C1Pt21Pt2);

using MFTTrack = MFTTracks::iterator;
using MFTTrackCovFwd = MFTTracksCov::iterator;

} // namespace aod
namespace soa
{
extern template struct JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::FwdTracks, aod::FwdTracksCov>;
}
namespace aod
{
using FullFwdTracks = soa::Join<FwdTracks, FwdTracksCov>;
using FullFwdTrack = FullFwdTracks::iterator;

// Some tracks cannot be uniquely identified with a collision. Some tracks cannot be assigned to a collision at all.
// Those tracks have -1 as collision index and have an entry in the AmbiguousTracks table.
// The estimated track time is used to assign BCs which are compatible with this track. Those are stored as a slice.
// All collisions compatible with these BCs may then have produced the ambiguous track.
// In the future possibly the DCA information can be exploited to reduce the possible collisions and then this table will be extended.
namespace ambiguous
{
DECLARE_SOA_INDEX_COLUMN(Track, track);       //! Track index
DECLARE_SOA_INDEX_COLUMN(MFTTrack, mfttrack); //! MFTTrack index
DECLARE_SOA_INDEX_COLUMN(FwdTrack, fwdtrack); //! FwdTrack index
DECLARE_SOA_SLICE_INDEX_COLUMN(BC, bc);       //! BC index (slice for 1 to N entries)
} // namespace ambiguous

DECLARE_SOA_TABLE(AmbiguousTracks, "AOD", "AMBIGUOUSTRACK", //! Table for tracks which are not uniquely associated with a collision
                  o2::soa::Index<>, ambiguous::TrackId, ambiguous::BCIdSlice);

using AmbiguousTrack = AmbiguousTracks::iterator;

DECLARE_SOA_TABLE(AmbiguousMFTTracks, "AOD", "AMBIGUOUSMFTTR", //! Table for MFT tracks which are not uniquely associated with a collision
                  o2::soa::Index<>, ambiguous::MFTTrackId, ambiguous::BCIdSlice);

using AmbiguousMFTTrack = AmbiguousMFTTracks::iterator;

DECLARE_SOA_TABLE(AmbiguousFwdTracks, "AOD", "AMBIGUOUSFWDTR", //! Table for Fwd tracks which are not uniquely associated with a collision
                  o2::soa::Index<>, ambiguous::FwdTrackId, ambiguous::BCIdSlice);

using AmbiguousFwdTrack = AmbiguousFwdTracks::iterator;

// Forward Tracks Cluster information
namespace fwdtrkcl
{
DECLARE_SOA_INDEX_COLUMN(FwdTrack, fwdtrack); //! Track index
DECLARE_SOA_COLUMN(X, x, float);              //! Cluster x coordinate
DECLARE_SOA_COLUMN(Y, y, float);              //! Cluster y coordinate
DECLARE_SOA_COLUMN(Z, z, float);              //! Cluster z coordinate
DECLARE_SOA_COLUMN(ClInfo, clInfo, uint16_t); //! Encoded detection element of cluster and cluster type along x and y
DECLARE_SOA_DYNAMIC_COLUMN(DEId, deId, [](uint16_t info) -> uint16_t { return (info & 0x7FF); });
DECLARE_SOA_DYNAMIC_COLUMN(IsGoodX, isGoodX, [](uint16_t info) -> bool { return ((info & 0x800) >> 11); });
DECLARE_SOA_DYNAMIC_COLUMN(IsGoodY, isGoodY, [](uint16_t info) -> bool { return ((info & 0x1000) >> 12); });
} // namespace fwdtrkcl

DECLARE_SOA_TABLE(FwdTrkCls, "AOD", "FWDTRKCL", //! Forward Track Cluster information
                  o2::soa::Index<>,
                  fwdtrkcl::FwdTrackId,
                  fwdtrkcl::X,
                  fwdtrkcl::Y,
                  fwdtrkcl::Z,
                  fwdtrkcl::ClInfo,
                  fwdtrkcl::DEId<fwdtrkcl::ClInfo>,
                  fwdtrkcl::IsGoodX<fwdtrkcl::ClInfo>,
                  fwdtrkcl::IsGoodY<fwdtrkcl::ClInfo>);

using FwdTrkCl = FwdTrkCls::iterator;

// HMPID information
namespace hmpid
{
DECLARE_SOA_INDEX_COLUMN(Track, track);                            //! Track index
DECLARE_SOA_COLUMN(HMPIDSignal, hmpidSignal, float);               //! Signal of the HMPID
DECLARE_SOA_COLUMN(HMPIDDistance, hmpidDistance, float);           //! Distance between the matched HMPID signal and the propagated track
DECLARE_SOA_COLUMN(HMPIDXTrack, hmpidXTrack, float);               //! Extrapolated track point x coordinate
DECLARE_SOA_COLUMN(HMPIDYTrack, hmpidYTrack, float);               //! Extrapolated track point y coordinate
DECLARE_SOA_COLUMN(HMPIDXMip, hmpidXMip, float);                   //! Matched MIP track point x coordinate
DECLARE_SOA_COLUMN(HMPIDYMip, hmpidYMip, float);                   //! Matched MIP track point y coordinate
DECLARE_SOA_COLUMN(HMPIDNPhotons, hmpidNPhotons, int);             //! Number of detected photons in HMPID
DECLARE_SOA_COLUMN(HMPIDQMip, hmpidQMip, float);                   //! Matched MIP cluster charge
DECLARE_SOA_COLUMN(HMPIDClusSize, hmpidClusSize, int);             //! Matched MIP cluster size
DECLARE_SOA_COLUMN(HMPIDMom, hmpidMom, float);                     //! Track momentum at the HMPID
DECLARE_SOA_COLUMN(HMPIDPhotsCharge, hmpidPhotsCharge, float[10]); //! Photon cluster charge
} // namespace hmpid

DECLARE_SOA_TABLE(HMPID_000, "AOD", "HMPID", //! HMPID information
                  o2::soa::Index<>,
                  hmpid::TrackId,
                  hmpid::HMPIDSignal,
                  hmpid::HMPIDDistance,
                  hmpid::HMPIDNPhotons,
                  hmpid::HMPIDQMip);

DECLARE_SOA_TABLE_VERSIONED(HMPID_001, "AOD", "HMPID", 1, //! HMPID information  version 1
                            o2::soa::Index<>,
                            hmpid::TrackId,
                            hmpid::HMPIDSignal,
                            hmpid::HMPIDXTrack,
                            hmpid::HMPIDYTrack,
                            hmpid::HMPIDXMip,
                            hmpid::HMPIDYMip,
                            hmpid::HMPIDNPhotons,
                            hmpid::HMPIDQMip,
                            hmpid::HMPIDClusSize,
                            hmpid::HMPIDMom,
                            hmpid::HMPIDPhotsCharge);

using HMPIDs = HMPID_001;
using HMPID = HMPIDs::iterator;

namespace calo
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                    //! BC index
DECLARE_SOA_COLUMN(CellNumber, cellNumber, int16_t); //!
DECLARE_SOA_COLUMN(Amplitude, amplitude, float);     //!
DECLARE_SOA_COLUMN(Time, time, float);               //!
DECLARE_SOA_COLUMN(CellType, cellType, int8_t);      //!
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);      //!
} // namespace calo

DECLARE_SOA_TABLE(Calos, "AOD", "CALO", //! Calorimeter cells
                  o2::soa::Index<>, calo::BCId, calo::CellNumber, calo::Amplitude,
                  calo::Time, calo::CellType, calo::CaloType);
using Calo = Calos::iterator;

namespace calotrigger
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                      //! BC index
DECLARE_SOA_COLUMN(FastOrAbsID, fastOrAbsID, int16_t); //! FastOR absolute ID
DECLARE_SOA_COLUMN(LnAmplitude, lnAmplitude, int16_t); //! L0 amplitude (ADC) := Peak Amplitude
DECLARE_SOA_COLUMN(TriggerBits, triggerBits, int32_t); //! Online trigger bits
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);        //! Calorimeter type (-1 is undefined, 0 is PHOS, 1 is EMCAL)
} // namespace calotrigger

DECLARE_SOA_TABLE(CaloTriggers, "AOD", "CALOTRIGGER", //! Trigger information from the calorimeter detectors
                  o2::soa::Index<>, calotrigger::BCId, calotrigger::FastOrAbsID,
                  calotrigger::LnAmplitude, calotrigger::TriggerBits, calotrigger::CaloType);
using CaloTrigger = CaloTriggers::iterator;

namespace cpvcluster
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                          //! BC index
DECLARE_SOA_COLUMN(PosX, posX, float);                     //! X position in cm
DECLARE_SOA_COLUMN(PosZ, posZ, float);                     //! Z position in cm
DECLARE_SOA_COLUMN(Amplitude, amplitude, float);           //! Signal amplitude
DECLARE_SOA_COLUMN(ClusterStatus, clusterStatus, uint8_t); //! 8 bits packed cluster status (bits 0-4 = pads mult, bits 5-6 = (module number - 2), bit 7 = isUnfolded)
DECLARE_SOA_DYNAMIC_COLUMN(PadMult, padMult, [](uint8_t status) -> uint8_t {
  return status & 0b00011111;
}); //! Multiplicity of pads in cluster
DECLARE_SOA_DYNAMIC_COLUMN(ModuleNumber, moduleNumber, [](uint8_t status) -> uint8_t {
  return 2 + ((status & 0b01100000) >> 5);
}); //! CPV module number (2, 3 or 4)
DECLARE_SOA_DYNAMIC_COLUMN(IsUnfolded, isUnfolded, [](uint8_t status) -> bool {
  return (status & 0b01100000) >> 7;
}); //! Number of local maxima in cluster
} // namespace cpvcluster

DECLARE_SOA_TABLE(CPVClusters, "AOD", "CPVCLUSTER", //! CPV clusters
                  o2::soa::Index<>, cpvcluster::BCId, cpvcluster::PosX, cpvcluster::PosZ, cpvcluster::Amplitude,
                  cpvcluster::ClusterStatus, cpvcluster::PadMult<cpvcluster::ClusterStatus>,
                  cpvcluster::ModuleNumber<cpvcluster::ClusterStatus>, cpvcluster::IsUnfolded<cpvcluster::ClusterStatus>);
using CPVCluster = CPVClusters::iterator;

namespace zdc
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                               //! BC index, to be used by both legacy and new table
DECLARE_SOA_COLUMN(EnergyZEM1, energyZEM1, float);              //!
DECLARE_SOA_COLUMN(EnergyZEM2, energyZEM2, float);              //!
DECLARE_SOA_COLUMN(EnergyCommonZNA, energyCommonZNA, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZNC, energyCommonZNC, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZPA, energyCommonZPA, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZPC, energyCommonZPC, float);    //!
DECLARE_SOA_COLUMN(EnergySectorZNA, energySectorZNA, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZNC, energySectorZNC, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZPA, energySectorZPA, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZPC, energySectorZPC, float[4]); //!
DECLARE_SOA_COLUMN(TimeZEM1, timeZEM1, float);                  //!
DECLARE_SOA_COLUMN(TimeZEM2, timeZEM2, float);                  //!
DECLARE_SOA_COLUMN(TimeZNA, timeZNA, float);                    //!
DECLARE_SOA_COLUMN(TimeZNC, timeZNC, float);                    //!
DECLARE_SOA_COLUMN(TimeZPA, timeZPA, float);                    //!
DECLARE_SOA_COLUMN(TimeZPC, timeZPC, float);                    //!
// New summarized table, minimal disk footprint, per channel like other detectors
DECLARE_SOA_COLUMN(Energy, energy, std::vector<float>);       //! Energy of non-zero channels. The channel IDs are given in ChannelE (at the same index)
DECLARE_SOA_COLUMN(ChannelE, channelE, std::vector<uint8_t>); //! Channel IDs which have reconstructed energy. There are at maximum 26 channels.
DECLARE_SOA_COLUMN(Amplitude, amplitude, std::vector<float>); //! Amplitudes of non-zero channels. The channel IDs are given in ChannelT (at the same index)
DECLARE_SOA_COLUMN(Time, time, std::vector<float>);           //! Times of non-zero channels. The channel IDs are given in ChannelT (at the same index)
DECLARE_SOA_COLUMN(ChannelT, channelT, std::vector<uint8_t>); //! Channel IDs which had non-zero amplitudes. There are at maximum 26 channels.
// Dynamic columns to take into account packed information; replace old getters
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyZEM1, energyZEM1, //! return ZEM1 energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto ne = channelE.size();
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZEM1) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyZEM2, energyZEM2, //! return ZEM2 energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZEM2) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyCommonZNA, energyCommonZNA, //! return common ZNA energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZNAC) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyCommonZNC, energyCommonZNC, //! return common ZNC energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZNCC) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyCommonZPA, energyCommonZPA, //! return common ZPA energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZPAC) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergyCommonZPC, energyCommonZPC, //! return common ZPC energy
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> float {
                             auto thisenergy = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZPCC) {
                                 thisenergy = energy[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergySectorZNA, energySectorZNA, //! return sector ZNA energy (array of 4 floats)
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> std::array<float, 4> {
                             std::array<float, 4> thisenergy = {
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZNA1)
                                 thisenergy[0] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNA2)
                                 thisenergy[1] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNA3)
                                 thisenergy[2] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNA4)
                                 thisenergy[3] = energy[ie];
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergySectorZNC, energySectorZNC, //! return sector ZNC energy (array of 4 floats)
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> std::array<float, 4> {
                             std::array<float, 4> thisenergy = {
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZNC1)
                                 thisenergy[0] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNC2)
                                 thisenergy[1] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNC3)
                                 thisenergy[2] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZNC4)
                                 thisenergy[3] = energy[ie];
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergySectorZPA, energySectorZPA, //! return sector ZPA energy (array of 4 floats)
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> std::array<float, 4> {
                             std::array<float, 4> thisenergy = {
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZPA1)
                                 thisenergy[0] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPA2)
                                 thisenergy[1] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPA3)
                                 thisenergy[2] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPA4)
                                 thisenergy[3] = energy[ie];
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyEnergySectorZPC, energySectorZPC, //! return sector ZPC energy (array of 4 floats)
                           [](gsl::span<const uint8_t> channelE, gsl::span<const float> energy) -> std::array<float, 4> {
                             std::array<float, 4> thisenergy = {
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};
                             for (uint64_t ie = 0; ie < channelE.size(); ie++) {
                               if (channelE[ie] == o2::zdc::IdZPC1)
                                 thisenergy[0] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPC2)
                                 thisenergy[1] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPC3)
                                 thisenergy[2] = energy[ie];
                               if (channelE[ie] == o2::zdc::IdZPC4)
                                 thisenergy[3] = energy[ie];
                             }
                             return thisenergy;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZEM1, timeZEM1, //! return ZEM1 time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZEM1) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZEM2, timeZEM2, //! return ZEM2 time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZEM2) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZNA, timeZNA, //! return ZNA time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZNAC) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZNC, timeZNC, //! return ZNC time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZNCC) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZPA, timeZPA, //! return ZPA time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZPAC) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyTimeZPC, timeZPC, //! return ZPC time information
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> time) -> float {
                             auto thistime = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZPCC) {
                                 thistime = time[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thistime;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZEM1, amplitudeZEM1, //! return ZEM1 amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZEM1) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZEM2, amplitudeZEM2, //! return ZEM2 amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZEM2) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZNA, amplitudeZNA, //! return ZNA amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZNAC) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZNC, amplitudeZNC, //! return ZNC amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZNCC) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZPA, amplitudeZPA, //! return ZPA amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZPAC) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(DyAmplitudeZPC, amplitudeZPC, //! return ZPC amplitude
                           [](gsl::span<const uint8_t> channelT, gsl::span<const float> amplitude) -> float {
                             auto thisamplitude = -std::numeric_limits<float>::infinity();
                             for (uint64_t ie = 0; ie < channelT.size(); ie++) {
                               if (channelT[ie] == o2::zdc::IdZPCC) {
                                 thisamplitude = amplitude[ie];
                                 break; // avoid unnecessary looping
                               }
                             }
                             return thisamplitude;
                           });
} // namespace zdc

DECLARE_SOA_TABLE(Zdcs_000, "AOD", "ZDC", //! ZDC information
                  o2::soa::Index<>, zdc::BCId, zdc::EnergyZEM1, zdc::EnergyZEM2,
                  zdc::EnergyCommonZNA, zdc::EnergyCommonZNC, zdc::EnergyCommonZPA, zdc::EnergyCommonZPC,
                  zdc::EnergySectorZNA, zdc::EnergySectorZNC, zdc::EnergySectorZPA, zdc::EnergySectorZPC,
                  zdc::TimeZEM1, zdc::TimeZEM2, zdc::TimeZNA, zdc::TimeZNC, zdc::TimeZPA, zdc::TimeZPC);

// new version of tables: different names for dynamic columns, but same getters
DECLARE_SOA_TABLE_VERSIONED(Zdcs_001, "AOD", "ZDC", 1, //! ZDC information, version 1, std::vector format
                            o2::soa::Index<>, zdc::BCId, zdc::Energy, zdc::ChannelE, zdc::Amplitude, zdc::Time, zdc::ChannelT,
                            zdc::DyEnergyZEM1<zdc::ChannelE, zdc::Energy>, zdc::DyEnergyZEM2<zdc::ChannelE, zdc::Energy>,
                            zdc::DyEnergyCommonZNA<zdc::ChannelE, zdc::Energy>, zdc::DyEnergyCommonZNC<zdc::ChannelE, zdc::Energy>,
                            zdc::DyEnergyCommonZPA<zdc::ChannelE, zdc::Energy>, zdc::DyEnergyCommonZPC<zdc::ChannelE, zdc::Energy>,
                            zdc::DyEnergySectorZNA<zdc::ChannelE, zdc::Energy>, zdc::DyEnergySectorZNC<zdc::ChannelE, zdc::Energy>,
                            zdc::DyEnergySectorZPA<zdc::ChannelE, zdc::Energy>, zdc::DyEnergySectorZPC<zdc::ChannelE, zdc::Energy>,
                            zdc::DyTimeZEM1<zdc::ChannelT, zdc::Time>, zdc::DyTimeZEM2<zdc::ChannelT, zdc::Time>,
                            zdc::DyTimeZNA<zdc::ChannelT, zdc::Time>, zdc::DyTimeZNC<zdc::ChannelT, zdc::Time>,
                            zdc::DyTimeZPA<zdc::ChannelT, zdc::Time>, zdc::DyTimeZPC<zdc::ChannelT, zdc::Time>,
                            zdc::DyAmplitudeZEM1<zdc::ChannelT, zdc::Amplitude>, zdc::DyAmplitudeZEM2<zdc::ChannelT, zdc::Amplitude>,
                            zdc::DyAmplitudeZNA<zdc::ChannelT, zdc::Amplitude>, zdc::DyAmplitudeZNC<zdc::ChannelT, zdc::Amplitude>,
                            zdc::DyAmplitudeZPA<zdc::ChannelT, zdc::Amplitude>, zdc::DyAmplitudeZPC<zdc::ChannelT, zdc::Amplitude>); //
using Zdcs = Zdcs_001;                                                                                                               //! new version
using Zdc = Zdcs::iterator;

namespace fv0a
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                             //! BC index
DECLARE_SOA_COLUMN(Amplitude, amplitude, std::vector<float>); //! Amplitudes of non-zero channels. The channel IDs are given in Channel (at the same index)
DECLARE_SOA_COLUMN(TimeFV0A, timeFV0A, std::vector<float>);   //! Time of non-zero channels. The channel IDs are given in Channel (at the same index). Only for the FITExtra table
DECLARE_SOA_COLUMN(Channel, channel, std::vector<uint8_t>);   //! Channel IDs which had non-zero amplitudes. There are at maximum 48 channels.
DECLARE_SOA_COLUMN(Time, time, float);                        //! Time in ns
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t);        //!
} // namespace fv0a

DECLARE_SOA_TABLE(FV0As, "AOD", "FV0A", //!
                  o2::soa::Index<>, fv0a::BCId, fv0a::Amplitude, fv0a::Channel, fv0a::Time, fv0a::TriggerMask);
using FV0A = FV0As::iterator;

DECLARE_SOA_TABLE(FV0AsExtra, "AOD", "FV0AEXTRA", //! FV0AsExtra table
                  o2::soa::Index<>, fv0a::BCId, fv0a::TimeFV0A);
using FV0AExtra = FV0AsExtra::iterator;

// V0C table for Run2 only
namespace fv0c
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                             //! BC index
DECLARE_SOA_COLUMN(Amplitude, amplitude, std::vector<float>); //! Amplitudes of non-zero channels. The channel IDs are given in Channel (at the same index)
DECLARE_SOA_COLUMN(Channel, channel, std::vector<uint8_t>);   //! Channel IDs which had non-zero amplitudes. There are at maximum 32 channels.
DECLARE_SOA_COLUMN(Time, time, float);                        //! Time in ns
} // namespace fv0c

DECLARE_SOA_TABLE(FV0Cs, "AOD", "FV0C", //! Only for RUN 2 converted data: V0C table
                  o2::soa::Index<>, fv0c::BCId, fv0c::Amplitude, fv0a::Channel, fv0c::Time);
using FV0C = FV0Cs::iterator;

namespace ft0
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                               //! BC index
DECLARE_SOA_COLUMN(AmplitudeA, amplitudeA, std::vector<float>); //! Amplitudes of non-zero channels on the A-side. The channel IDs are given in ChannelA (at the same index)
DECLARE_SOA_COLUMN(TimeFT0A, timeFT0A, std::vector<float>);     //! Time of non-zero channels on the A-side. The channel IDs are given in ChannelA (at the same index). Only for the FITExtra table
DECLARE_SOA_COLUMN(ChannelA, channelA, std::vector<uint8_t>);   //! Channel IDs on the A side which had non-zero amplitudes. There are at maximum 96 channels.
DECLARE_SOA_COLUMN(AmplitudeC, amplitudeC, std::vector<float>); //! Amplitudes of non-zero channels on the C-side. The channel IDs are given in ChannelC (at the same index)
DECLARE_SOA_COLUMN(TimeFT0C, timeFT0C, std::vector<float>);     //! Time of non-zero channels on the C-side. The channel IDs are given in ChannelC (at the same index). Only for the FITExtra table
DECLARE_SOA_COLUMN(ChannelC, channelC, std::vector<uint8_t>);   //! Channel IDs on the C side which had non-zero amplitudes. There are at maximum 112 channels.
DECLARE_SOA_COLUMN(TimeA, timeA, float);                        //! Average A-side time
DECLARE_SOA_COLUMN(TimeC, timeC, float);                        //! Average C-side time
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t);          //!
DECLARE_SOA_DYNAMIC_COLUMN(PosZ, posZ,                          //! Z position calculated from timeA and timeC in cm
                           [](float timeA, float timeC) -> float {
                             return o2::constants::physics::LightSpeedCm2NS * (timeC - timeA) / 2;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(CollTime, collTime, //! Collision time, one need also check validation (code below) for timeA and timeC
                           [](float timeA, float timeC) -> float {
                             return (timeA + timeC) / 2;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(IsValidTimeA, isValidTimeA, //! Checks if time from A side was calculated, and if is not dummy
                           [](float timeA) -> bool {
                             return timeA < 30.f; // Due to HW limitations time can be only within range (-25,25) ns, dummy time is around 32 ns
                           });
DECLARE_SOA_DYNAMIC_COLUMN(IsValidTimeC, isValidTimeC, //! Checks if time from C side was calculated
                           [](float timeC) -> bool {
                             return timeC < 30.f; // Due to HW limitations time can be only within range (-25,25) ns, dummy time is around 32 ns
                           });
DECLARE_SOA_DYNAMIC_COLUMN(IsValidTime, isValidTime, //! Checks if times from A and C side were calculated simultaneously
                           [](float timeA, float timeC) -> bool {
                             return (timeA < 30.f) && (timeC < 30.f); // Due to HW limitations time can be only within range (-25,25) ns, dummy time is around 32 ns
                           });
DECLARE_SOA_DYNAMIC_COLUMN(SumAmpA, sumAmpA, //! Calculates sum of positive amplitudes from side A
                           [](gsl::span<const float> vecAmpA) -> float {
                             return std::accumulate(vecAmpA.begin(), vecAmpA.end(), 0.f, [](auto&& sum, auto&& curr) { return sum + (curr > 0 ? curr : 0); });
                           });
DECLARE_SOA_DYNAMIC_COLUMN(SumAmpC, sumAmpC, //! Calculates sum of positive amplitudes from side C
                           [](gsl::span<const float> vecAmpC) -> float {
                             return std::accumulate(vecAmpC.begin(), vecAmpC.end(), 0.f, [](auto&& sum, auto&& curr) { return sum + (curr > 0 ? curr : 0); });
                           });

} // namespace ft0

DECLARE_SOA_TABLE(FT0s, "AOD", "FT0", //!
                  o2::soa::Index<>, ft0::BCId,
                  ft0::AmplitudeA, ft0::ChannelA, ft0::AmplitudeC, ft0::ChannelC, ft0::TimeA, ft0::TimeC,
                  ft0::TriggerMask, ft0::PosZ<ft0::TimeA, ft0::TimeC>, ft0::CollTime<ft0::TimeA, ft0::TimeC>,
                  ft0::IsValidTimeA<ft0::TimeA>, ft0::IsValidTimeC<ft0::TimeC>, ft0::IsValidTime<ft0::TimeA, ft0::TimeC>,
                  ft0::SumAmpA<ft0::AmplitudeA>, ft0::SumAmpC<ft0::AmplitudeC>);
using FT0 = FT0s::iterator;

DECLARE_SOA_TABLE(FT0sExtra, "AOD", "FT0EXTRA", //! FT0sExtra table
                  o2::soa::Index<>, ft0::BCId,
                  ft0::TimeFT0A, ft0::TimeFT0C);
using FT0Extra = FT0sExtra::iterator;

namespace fdd
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                     //! BC index
DECLARE_SOA_COLUMN(AmplitudeA, amplitudeA, float[4]); //! Amplitude in adjacent pairs A-side
DECLARE_SOA_COLUMN(AmplitudeC, amplitudeC, float[4]); //! Amplitude in adjacent pairs C-side

DECLARE_SOA_COLUMN(ChargeA, chargeA, int16_t[8]); //! Amplitude per channel A-side
DECLARE_SOA_COLUMN(ChargeC, chargeC, int16_t[8]); //! Amplitude per channel C-side

DECLARE_SOA_COLUMN(TimeFDDA, timeFDDA, float[8]); //! Time per channel A-side, only for the FITExtra table
DECLARE_SOA_COLUMN(TimeFDDC, timeFDDC, float[8]); //! Time per channel C-side, only for the FITExtra table

DECLARE_SOA_COLUMN(TimeA, timeA, float);               //!
DECLARE_SOA_COLUMN(TimeC, timeC, float);               //!
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t); //!
} // namespace fdd

DECLARE_SOA_TABLE(FDDs_000, "AOD", "FDD", //! FDD table, version 000
                  o2::soa::Index<>, fdd::BCId,
                  fdd::AmplitudeA, fdd::AmplitudeC,
                  fdd::TimeA, fdd::TimeC,
                  fdd::TriggerMask);

DECLARE_SOA_TABLE_VERSIONED(FDDs_001, "AOD", "FDD", 1, //! FDD table, version 001
                            o2::soa::Index<>,
                            fdd::BCId,
                            fdd::ChargeA, fdd::ChargeC,
                            fdd::TimeA, fdd::TimeC,
                            fdd::TriggerMask);

using FDDs = FDDs_001; //! this defines the current default version
using FDD = FDDs::iterator;

DECLARE_SOA_TABLE(FDDsExtra, "AOD", "FDDEXTRA", //! FDDsExtra table
                  o2::soa::Index<>, fdd::BCId,
                  fdd::TimeFDDA, fdd::TimeFDDC);
using FDDExtra = FDDsExtra::iterator;

namespace v0
{
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "_Pos"); //! Positive track
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "_Neg"); //! Negative track
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                         //! Collision index
DECLARE_SOA_COLUMN(V0Type, v0Type, uint8_t);                            //! custom bitmap for various selections (see below)

DECLARE_SOA_DYNAMIC_COLUMN(IsStandardV0, isStandardV0, //! is standard V0
                           [](uint8_t V0Type) -> bool { return V0Type == 1; });
DECLARE_SOA_DYNAMIC_COLUMN(IsPhotonV0, isPhotonV0, //! is TPC-only V0 for which the photon-mass-hypothesis was good
                           [](uint8_t V0Type) -> bool { return V0Type & (1 << 1); });
DECLARE_SOA_DYNAMIC_COLUMN(IsCollinearV0, isCollinearV0, //! is V0 for which the photon-mass-hypothesis was good and was fitted collinearly
                           [](uint8_t V0Type) -> bool { return V0Type & (1 << 2); });

} // namespace v0

DECLARE_SOA_TABLE(V0s_000, "AOD", "V0", //! Run 2 V0 table (version 000)
                  o2::soa::Index<>,
                  v0::PosTrackId, v0::NegTrackId);
DECLARE_SOA_TABLE_VERSIONED(V0s_001, "AOD", "V0", 1, //! Run 3 V0 table (version 001)
                            o2::soa::Index<>, v0::CollisionId,
                            v0::PosTrackId, v0::NegTrackId);
DECLARE_SOA_TABLE_VERSIONED(V0s_002, "AOD", "V0", 2, //! Run 3 V0 table (version 002)
                            o2::soa::Index<>, v0::CollisionId,
                            v0::PosTrackId, v0::NegTrackId,
                            v0::V0Type,
                            v0::IsStandardV0<v0::V0Type>,
                            v0::IsPhotonV0<v0::V0Type>,
                            v0::IsCollinearV0<v0::V0Type>);

using V0s = V0s_002; //! this defines the current default version
using V0 = V0s::iterator;

namespace cascade
{
DECLARE_SOA_INDEX_COLUMN(V0, v0);                                   //! V0 index
DECLARE_SOA_INDEX_COLUMN_FULL(Bachelor, bachelor, int, Tracks, ""); //! Bachelor track index
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                     //! Collision index
} // namespace cascade

DECLARE_SOA_TABLE(Cascades_000, "AOD", "CASCADE", //! Run 2 cascade table
                  o2::soa::Index<>, cascade::V0Id, cascade::BachelorId);
DECLARE_SOA_TABLE_VERSIONED(Cascades_001, "AOD", "CASCADE", 1, //! Run 3 cascade table
                            o2::soa::Index<>, cascade::CollisionId, cascade::V0Id, cascade::BachelorId);

using Cascades = Cascades_001; //! this defines the current default version
using Cascade = Cascades::iterator;

namespace decay3body
{
DECLARE_SOA_INDEX_COLUMN_FULL(Track0, track0, int, Tracks, "_0"); //! Track 0 index
DECLARE_SOA_INDEX_COLUMN_FULL(Track1, track1, int, Tracks, "_1"); //! Track 1 index
DECLARE_SOA_INDEX_COLUMN_FULL(Track2, track2, int, Tracks, "_2"); //! Track 2 index
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                   //! Collision index
} // namespace decay3body

DECLARE_SOA_TABLE(Decay3Bodys, "AOD", "DECAY3BODY", //! 3-body decay table
                  o2::soa::Index<>, decay3body::CollisionId, decay3body::Track0Id, decay3body::Track1Id, decay3body::Track2Id);

using Decay3Bodys = Decay3Bodys; //! this defines the current default version
using Decay3Body = Decay3Bodys::iterator;

namespace strangenesstracking
{
DECLARE_SOA_INDEX_COLUMN(Track, track);                                 //! Strange track index
DECLARE_SOA_INDEX_COLUMN_FULL(ITSTrack, itsTrack, int, Tracks, "_ITS"); //! ITS index
DECLARE_SOA_INDEX_COLUMN(Cascade, cascade);                             //! Cascade index
DECLARE_SOA_INDEX_COLUMN(V0, v0);                                       //! V0 index
DECLARE_SOA_INDEX_COLUMN(Decay3Body, decay3Body);                       //! Decay 3 body index

DECLARE_SOA_COLUMN(DecayX, decayX, float);             //! X coordinate of decay vertex
DECLARE_SOA_COLUMN(DecayY, decayY, float);             //! Y coordinate of decay vertex
DECLARE_SOA_COLUMN(DecayZ, decayZ, float);             //! Z coordinate of decay vertex
DECLARE_SOA_COLUMN(XiMass, xiMass, float);             //! Xi mass
DECLARE_SOA_COLUMN(OmegaMass, omegaMass, float);       //! Omega mass
DECLARE_SOA_COLUMN(H3Lmass, h3Lmass, float);           //! H3L mass
DECLARE_SOA_COLUMN(H4Lmass, h4Lmass, float);           //! H4L mass
DECLARE_SOA_COLUMN(He4Lmass, he4Lmass, float);         //! He4L mass
DECLARE_SOA_COLUMN(MatchingChi2, matchingChi2, float); //! Matching Chi2
DECLARE_SOA_COLUMN(TopologyChi2, topologyChi2, float); //! Topology Chi2
DECLARE_SOA_COLUMN(ITSclsSize, itsClsSize, float);     //! Average ITS cluster size
} // namespace strangenesstracking

/// Strangeness tracks
DECLARE_SOA_TABLE_FULL(TrackedCascades, "TrackedCascade", "AOD", "TRACKEDCASCADE", //! Strangeness tracking cascade table
                       o2::soa::Index<>,
                       strangenesstracking::TrackId,
                       strangenesstracking::ITSTrackId,
                       strangenesstracking::CascadeId,
                       strangenesstracking::DecayX,
                       strangenesstracking::DecayY,
                       strangenesstracking::DecayZ,
                       strangenesstracking::XiMass,
                       strangenesstracking::OmegaMass,
                       strangenesstracking::MatchingChi2,
                       strangenesstracking::TopologyChi2,
                       strangenesstracking::ITSclsSize);

DECLARE_SOA_TABLE_FULL(TrackedV0s, "TrackedV0", "AOD", "TRACKEDV0", //! Strangeness tracking V0 table
                       o2::soa::Index<>,
                       strangenesstracking::TrackId,
                       strangenesstracking::ITSTrackId,
                       strangenesstracking::V0Id,
                       strangenesstracking::DecayX,
                       strangenesstracking::DecayY,
                       strangenesstracking::DecayZ,
                       strangenesstracking::H3Lmass,
                       strangenesstracking::H4Lmass,
                       strangenesstracking::MatchingChi2,
                       strangenesstracking::TopologyChi2,
                       strangenesstracking::ITSclsSize);

DECLARE_SOA_TABLE_FULL(Tracked3Bodys, "Tracked3Body", "AOD", "TRACKED3BODY", //! Strangeness tracking 3-body decay table
                       o2::soa::Index<>,
                       strangenesstracking::TrackId,
                       strangenesstracking::ITSTrackId,
                       strangenesstracking::Decay3BodyId,
                       strangenesstracking::DecayX,
                       strangenesstracking::DecayY,
                       strangenesstracking::DecayZ,
                       strangenesstracking::H3Lmass,
                       strangenesstracking::He4Lmass,
                       strangenesstracking::MatchingChi2,
                       strangenesstracking::TopologyChi2,
                       strangenesstracking::ITSclsSize);

using TrackedCascades = TrackedCascades;
using TrackedCascade = TrackedCascades::iterator;
using TrackedV0s = TrackedV0s;
using TrackedV0 = TrackedV0s::iterator;
using Tracked3Bodys = Tracked3Bodys;
using Tracked3body = Tracked3Bodys::iterator;

namespace origins
{
DECLARE_SOA_COLUMN(DataframeID, dataframeID, uint64_t); //! Data frame ID (what is usually found in directory name in the AO2D.root, i.e. DF_XXX)
} // namespace origins

DECLARE_SOA_TABLE(Origins, "AOD", "ORIGIN", //! Table which contains the IDs of all dataframes merged into this dataframe
                  o2::soa::Index<>, origins::DataframeID);

using Origin = Origins::iterator;

// ---- Run 2 tables ----
namespace run2
{
DECLARE_SOA_COLUMN(EventCuts, eventCuts, uint32_t);                   //! Event selection flags. Check enum Run2EventSelectionCut
DECLARE_SOA_COLUMN(TriggerMaskNext50, triggerMaskNext50, uint64_t);   //! 50 further trigger classes after bc.triggerMask()
DECLARE_SOA_COLUMN(L0TriggerInputMask, l0TriggerInputMask, uint32_t); //! CTP L0 trigger input mask
DECLARE_SOA_COLUMN(SPDClustersL0, spdClustersL0, uint16_t);           //! Number of clusters in the first layer of the SPD
DECLARE_SOA_COLUMN(SPDClustersL1, spdClustersL1, uint16_t);           //! Number of clusters in the second layer of the SPD
DECLARE_SOA_COLUMN(SPDFiredChipsL0, spdFiredChipsL0, uint16_t);       //! Fired chips in the first layer of the SPD (offline)
DECLARE_SOA_COLUMN(SPDFiredChipsL1, spdFiredChipsL1, uint16_t);       //! Fired chips in the second layer of the SPD (offline)
DECLARE_SOA_COLUMN(SPDFiredFastOrL0, spdFiredFastOrL0, uint16_t);     //! Fired FASTOR signals in the first layer of the SPD (online)
DECLARE_SOA_COLUMN(SPDFiredFastOrL1, spdFiredFastOrL1, uint16_t);     //! Fired FASTOR signals in the first layer of the SPD (online)
DECLARE_SOA_COLUMN(V0TriggerChargeA, v0TriggerChargeA, uint16_t);     //! V0A trigger charge
DECLARE_SOA_COLUMN(V0TriggerChargeC, v0TriggerChargeC, uint16_t);     //! V0C trigger charge
DECLARE_SOA_COLUMN(NTPCClusters, nTPCClusters, uint32_t);             //! total number of TPC clusters (for ev sel)
DECLARE_SOA_COLUMN(NSDDSSDClusters, nSDDSSDClusters, uint32_t);       //! total number of SSD + SDD clusters (for ev sel)
namespace oftv0
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                         //! Collision index
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "_Pos"); //! Positive track
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "_Neg"); //! Negative track
DECLARE_SOA_COLUMN(Px, px, float);                                      //! momentum in x
DECLARE_SOA_COLUMN(Py, py, float);                                      //! momentum in y
DECLARE_SOA_COLUMN(Pz, pz, float);                                      //! momentum in z
DECLARE_SOA_COLUMN(E, e, float);                                        //! energy
DECLARE_SOA_COLUMN(Qt, qt, float);                                      //! Qt of AP
DECLARE_SOA_COLUMN(Alpha, alpha, float);                                //! Alpha of AP
DECLARE_SOA_COLUMN(X, x, float);                                        //! conversion point x coordiante
DECLARE_SOA_COLUMN(Y, y, float);                                        //! conversion point y coordiante
DECLARE_SOA_COLUMN(Z, z, float);                                        //! conversion point z coordiante
DECLARE_SOA_COLUMN(Chi2NDF, chi2NDF, float);                            //! chi^2 over NDF from KFParticle
DECLARE_SOA_COLUMN(PsiPair, psiPair, float);                            //! Psi pair
DECLARE_SOA_COLUMN(DCAr, dcaR, float);                                  //! DCA in radial direction
DECLARE_SOA_COLUMN(DCAz, dcaZ, float);                                  //! DCA in z direction
DECLARE_SOA_COLUMN(Mass, mass, float);                                  //! mass of the conversion. Do NOT use for cut!
} // namespace oftv0
} // namespace run2

DECLARE_SOA_TABLE(Run2BCInfos_000, "AOD", "RUN2BCINFO", run2::EventCuts, //! Legacy information for Run 2 event selection
                  run2::TriggerMaskNext50, run2::L0TriggerInputMask,
                  run2::SPDClustersL0, run2::SPDClustersL1,
                  run2::SPDFiredChipsL0, run2::SPDFiredChipsL1,
                  run2::SPDFiredFastOrL0, run2::SPDFiredFastOrL1,
                  run2::V0TriggerChargeA, run2::V0TriggerChargeC);

DECLARE_SOA_TABLE_VERSIONED(Run2BCInfos_001, "AOD", "RUN2BCINFO", 1,
                            run2::EventCuts, //! Legacy information for Run 2 event selection
                            run2::TriggerMaskNext50, run2::L0TriggerInputMask,
                            run2::SPDClustersL0, run2::SPDClustersL1,
                            run2::SPDFiredChipsL0, run2::SPDFiredChipsL1,
                            run2::SPDFiredFastOrL0, run2::SPDFiredFastOrL1,
                            run2::V0TriggerChargeA, run2::V0TriggerChargeC,
                            run2::NTPCClusters, run2::NSDDSSDClusters);

using Run2BCInfos = Run2BCInfos_001;
using Run2BCInfo = Run2BCInfos::iterator;

DECLARE_SOA_TABLE(Run2OTFV0s, "AOD", "Run2OTFV0", //! Run 2 V0 on the fly table
                  o2::soa::Index<>,
                  run2::oftv0::CollisionId, run2::oftv0::PosTrackId, run2::oftv0::NegTrackId,
                  run2::oftv0::Px, run2::oftv0::Py, run2::oftv0::Pz, run2::oftv0::E,
                  run2::oftv0::Qt, run2::oftv0::Alpha,
                  run2::oftv0::X, run2::oftv0::Y, run2::oftv0::Z,
                  run2::oftv0::Chi2NDF, run2::oftv0::PsiPair,
                  run2::oftv0::DCAr, run2::oftv0::DCAz,
                  run2::oftv0::Mass);

using Run2OTFV0 = Run2OTFV0s::iterator;

// ---- MC tables ----
namespace mccollision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                            //! BC index
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short);       //! disentangled generator IDs should be accessed using getGeneratorId, getSubGeneratorId and getSourceId
DECLARE_SOA_COLUMN(PosX, posX, float);                       //! X vertex position in cm
DECLARE_SOA_COLUMN(PosY, posY, float);                       //! Y vertex position in cm
DECLARE_SOA_COLUMN(PosZ, posZ, float);                       //! Z vertex position in cm
DECLARE_SOA_COLUMN(T, t, float);                             //! Collision time relative to given bc in ns
DECLARE_SOA_COLUMN(Weight, weight, float);                   //! MC weight
DECLARE_SOA_COLUMN(ImpactParameter, impactParameter, float); //! Impact parameter for A-A
DECLARE_SOA_COLUMN(EventPlaneAngle, eventPlaneAngle, float); //! Event plane angle for A-A
DECLARE_SOA_DYNAMIC_COLUMN(GetGeneratorId, getGeneratorId,   //! The global generator ID which might have been assigned by the user
                           [](short generatorsID) -> int { return o2::mcgenid::getGeneratorId(generatorsID); });
DECLARE_SOA_DYNAMIC_COLUMN(GetSubGeneratorId, getSubGeneratorId, //! A specific sub-generator ID in case the generator has some sub-generator logic
                           [](short generatorsID) -> int { return o2::mcgenid::getSubGeneratorId(generatorsID); });
DECLARE_SOA_DYNAMIC_COLUMN(GetSourceId, getSourceId, //! The source ID to differentiate between signals and background in an embedding simulation
                           [](short generatorsID) -> int { return o2::mcgenid::getSourceId(generatorsID); });

} // namespace mccollision

DECLARE_SOA_TABLE(McCollisions_000, "AOD", "MCCOLLISION", //! MC collision table
                  o2::soa::Index<>, mccollision::BCId,
                  mccollision::GeneratorsID,
                  mccollision::PosX, mccollision::PosY, mccollision::PosZ,
                  mccollision::T, mccollision::Weight,
                  mccollision::ImpactParameter,
                  mccollision::GetGeneratorId<mccollision::GeneratorsID>,
                  mccollision::GetSubGeneratorId<mccollision::GeneratorsID>,
                  mccollision::GetSourceId<mccollision::GeneratorsID>);
DECLARE_SOA_TABLE_VERSIONED(McCollisions_001, "AOD", "MCCOLLISION", 1, //! MC collision table with event plane
                            o2::soa::Index<>, mccollision::BCId,
                            mccollision::GeneratorsID,
                            mccollision::PosX, mccollision::PosY, mccollision::PosZ,
                            mccollision::T, mccollision::Weight,
                            mccollision::ImpactParameter,
                            mccollision::EventPlaneAngle,
                            mccollision::GetGeneratorId<mccollision::GeneratorsID>,
                            mccollision::GetSubGeneratorId<mccollision::GeneratorsID>,
                            mccollision::GetSourceId<mccollision::GeneratorsID>);

using McCollisions = McCollisions_001;
using McCollision = McCollisions::iterator;

namespace mcparticle
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);                                     //! MC collision of this particle
DECLARE_SOA_COLUMN(PdgCode, pdgCode, int);                                              //! PDG code
DECLARE_SOA_COLUMN(StatusCode, statusCode, int);                                        //! Generators status code or physics process. Do not use directly. Use dynamic columns getGenStatusCode() or getProcess()
DECLARE_SOA_COLUMN(Flags, flags, uint8_t);                                              //! ALICE specific flags, see MCParticleFlags. Do not use directly. Use the dynamic columns, e.g. producedByGenerator()
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Mother0, mother0, int, "McParticles_Mother0");       //! Track index of the first mother
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Mother1, mother1, int, "McParticles_Mother1");       //! Track index of the last mother
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Daughter0, daughter0, int, "McParticles_Daughter0"); //! Track index of the first daugther
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Daughter1, daughter1, int, "McParticles_Daughter1"); //! Track index of the last daugther
DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN(Mothers, mothers);                                  //! Mother tracks (possible empty) array. Iterate over mcParticle.mothers_as<aod::McParticles>())
DECLARE_SOA_SELF_SLICE_INDEX_COLUMN(Daughters, daughters);                              //! Daughter tracks (possibly empty) slice. Check for non-zero with mcParticle.has_daughters(). Iterate over mcParticle.daughters_as<aod::McParticles>())
DECLARE_SOA_COLUMN(Weight, weight, float);                                              //! MC weight
DECLARE_SOA_COLUMN(Px, px, float);                                                      //! Momentum in x in GeV/c
DECLARE_SOA_COLUMN(Py, py, float);                                                      //! Momentum in y in GeV/c
DECLARE_SOA_COLUMN(Pz, pz, float);                                                      //! Momentum in z in GeV/c
DECLARE_SOA_COLUMN(E, e, float);                                                        //! Energy
DECLARE_SOA_COLUMN(Vx, vx, float);                                                      //! X production vertex in cm
DECLARE_SOA_COLUMN(Vy, vy, float);                                                      //! Y production vertex in cm
DECLARE_SOA_COLUMN(Vz, vz, float);                                                      //! Z production vertex in cm
DECLARE_SOA_COLUMN(Vt, vt, float);                                                      //! Production time
DECLARE_SOA_DYNAMIC_COLUMN(ProducedByGenerator, producedByGenerator,                    //! True if particle produced by the generator (==TMCProcess::kPrimary); False if by the transport code
                           [](uint8_t flags) -> bool { return (flags & o2::aod::mcparticle::enums::ProducedByTransport) == 0x0; });
DECLARE_SOA_DYNAMIC_COLUMN(FromBackgroundEvent, fromBackgroundEvent, //! Particle from background event
                           [](uint8_t flags) -> bool { return (flags & o2::aod::mcparticle::enums::FromBackgroundEvent) == o2::aod::mcparticle::enums::FromBackgroundEvent; });
DECLARE_SOA_DYNAMIC_COLUMN(GetProcess, getProcess, //! The VMC physics code (as int) that generated this particle (see header TMCProcess.h in ROOT)
                           [](uint8_t flags, int statusCode) -> int { if ((flags & o2::aod::mcparticle::enums::ProducedByTransport) == 0x0) { return 0 /*TMCProcess::kPrimary*/; } else { return statusCode; } });
DECLARE_SOA_DYNAMIC_COLUMN(GetGenStatusCode, getGenStatusCode, //! The native status code put by the generator, or -1 if a particle produced during transport
                           [](uint8_t flags, int statusCode) -> int { if ((flags & o2::aod::mcparticle::enums::ProducedByTransport) == 0x0) { return o2::mcgenstatus::getGenStatusCode(statusCode); } else { return -1; } });
DECLARE_SOA_DYNAMIC_COLUMN(GetHepMCStatusCode, getHepMCStatusCode, //! The HepMC status code put by the generator, or -1 if a particle produced during transport
                           [](uint8_t flags, int statusCode) -> int { if ((flags & o2::aod::mcparticle::enums::ProducedByTransport) == 0x0) { return o2::mcgenstatus::getHepMCStatusCode(statusCode); } else { return -1; } });
DECLARE_SOA_DYNAMIC_COLUMN(IsPhysicalPrimary, isPhysicalPrimary, //! True if particle is considered a physical primary according to the ALICE definition
                           [](uint8_t flags) -> bool { return (flags & o2::aod::mcparticle::enums::PhysicalPrimary) == o2::aod::mcparticle::enums::PhysicalPrimary; });
DECLARE_SOA_DYNAMIC_COLUMN(PVector, pVector, //! Momentum vector in x,y,z-directions in GeV/c
                           [](float px, float py, float pz) -> std::array<float, 3> { return std::array<float, 3>{px, py, pz}; });

DECLARE_SOA_EXPRESSION_COLUMN(Phi, phi, float, //! Phi in the range [0, 2pi)
                              o2::constants::math::PI + natan2(-1.0f * aod::mcparticle::py, -1.0f * aod::mcparticle::px));
DECLARE_SOA_EXPRESSION_COLUMN(Eta, eta, float, //! Pseudorapidity, conditionally defined to avoid FPEs
                              ifnode((nsqrt(aod::mcparticle::px * aod::mcparticle::px +
                                            aod::mcparticle::py * aod::mcparticle::py +
                                            aod::mcparticle::pz * aod::mcparticle::pz) -
                                      aod::mcparticle::pz) < static_cast<float>(1e-7),
                                     ifnode(aod::mcparticle::pz < 0.f, -100.f, 100.f),
                                     0.5f * nlog((nsqrt(aod::mcparticle::px * aod::mcparticle::px +
                                                        aod::mcparticle::py * aod::mcparticle::py +
                                                        aod::mcparticle::pz * aod::mcparticle::pz) +
                                                  aod::mcparticle::pz) /
                                                 (nsqrt(aod::mcparticle::px * aod::mcparticle::px +
                                                        aod::mcparticle::py * aod::mcparticle::py +
                                                        aod::mcparticle::pz * aod::mcparticle::pz) -
                                                  aod::mcparticle::pz))));
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //! Transverse momentum in GeV/c
                              nsqrt(aod::mcparticle::px* aod::mcparticle::px +
                                    aod::mcparticle::py * aod::mcparticle::py));
DECLARE_SOA_EXPRESSION_COLUMN(P, p, float, //! Total momentum in GeV/c
                              nsqrt(aod::mcparticle::px* aod::mcparticle::px +
                                    aod::mcparticle::py * aod::mcparticle::py +
                                    aod::mcparticle::pz * aod::mcparticle::pz));
DECLARE_SOA_EXPRESSION_COLUMN(Y, y, float, //! Particle rapidity, conditionally defined to avoid FPEs
                              ifnode((aod::mcparticle::e - aod::mcparticle::pz) < static_cast<float>(1e-7),
                                     ifnode(aod::mcparticle::pz < 0.f, -100.f, 100.f),
                                     0.5f * nlog((aod::mcparticle::e + aod::mcparticle::pz) /
                                                 (aod::mcparticle::e - aod::mcparticle::pz))));
} // namespace mcparticle

DECLARE_SOA_TABLE_FULL(StoredMcParticles_000, "McParticles", "AOD", "MCPARTICLE", //! MC particle table, version 000
                       o2::soa::Index<>, mcparticle::McCollisionId,
                       mcparticle::PdgCode, mcparticle::StatusCode, mcparticle::Flags,
                       mcparticle::Mother0Id, mcparticle::Mother1Id,
                       mcparticle::Daughter0Id, mcparticle::Daughter1Id, mcparticle::Weight,
                       mcparticle::Px, mcparticle::Py, mcparticle::Pz, mcparticle::E,
                       mcparticle::Vx, mcparticle::Vy, mcparticle::Vz, mcparticle::Vt,
                       mcparticle::PVector<mcparticle::Px, mcparticle::Py, mcparticle::Pz>,
                       mcparticle::ProducedByGenerator<mcparticle::Flags>,
                       mcparticle::FromBackgroundEvent<mcparticle::Flags>,
                       mcparticle::GetGenStatusCode<mcparticle::Flags, mcparticle::StatusCode>,
                       mcparticle::GetHepMCStatusCode<mcparticle::Flags, mcparticle::StatusCode>,
                       mcparticle::GetProcess<mcparticle::Flags, mcparticle::StatusCode>,
                       mcparticle::IsPhysicalPrimary<mcparticle::Flags>);

DECLARE_SOA_TABLE_FULL_VERSIONED(StoredMcParticles_001, "McParticles", "AOD", "MCPARTICLE", 1, //! MC particle table, version 001
                                 o2::soa::Index<>, mcparticle::McCollisionId,
                                 mcparticle::PdgCode, mcparticle::StatusCode, mcparticle::Flags,
                                 mcparticle::MothersIds, mcparticle::DaughtersIdSlice, mcparticle::Weight,
                                 mcparticle::Px, mcparticle::Py, mcparticle::Pz, mcparticle::E,
                                 mcparticle::Vx, mcparticle::Vy, mcparticle::Vz, mcparticle::Vt,
                                 mcparticle::PVector<mcparticle::Px, mcparticle::Py, mcparticle::Pz>,
                                 mcparticle::ProducedByGenerator<mcparticle::Flags>,
                                 mcparticle::FromBackgroundEvent<mcparticle::Flags>,
                                 mcparticle::GetGenStatusCode<mcparticle::Flags, mcparticle::StatusCode>,
                                 mcparticle::GetHepMCStatusCode<mcparticle::Flags, mcparticle::StatusCode>,
                                 mcparticle::GetProcess<mcparticle::Flags, mcparticle::StatusCode>,
                                 mcparticle::IsPhysicalPrimary<mcparticle::Flags>);

DECLARE_SOA_EXTENDED_TABLE(McParticles_000, StoredMcParticles_000, "EXMCPARTICLE", 0, //! Basic MC particle properties
                           mcparticle::Phi,
                           mcparticle::Eta,
                           mcparticle::Pt,
                           mcparticle::P,
                           mcparticle::Y);

DECLARE_SOA_EXTENDED_TABLE(McParticles_001, StoredMcParticles_001, "EXMCPARTICLE", 1, //! Basic MC particle properties
                           mcparticle::Phi,
                           mcparticle::Eta,
                           mcparticle::Pt,
                           mcparticle::P,
                           mcparticle::Y);

using StoredMcParticles = StoredMcParticles_001;
using McParticles = McParticles_001;
using McParticle = McParticles::iterator;
} // namespace aod
namespace soa
{
DECLARE_EQUIVALENT_FOR_INDEX_NG("COLLISION/0", "COLLISION/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("MCPARTICLE/0", "MCPARTICLE/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK/0", "TRACK_IU/0");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK/0", "TRACKEXTRA/0");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK_IU/0", "TRACKEXTRA/0");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK/0", "TRACKEXTRA/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK_IU/0", "TRACKEXTRA/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK/0", "TRACKEXTRA/2");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACK_IU/0", "TRACKEXTRA/2");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACKEXTRA/0", "TRACKEXTRA/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACKEXTRA/0", "TRACKEXTRA/2");
DECLARE_EQUIVALENT_FOR_INDEX_NG("TRACKEXTRA/1", "TRACKEXTRA/2");
DECLARE_EQUIVALENT_FOR_INDEX_NG("HMPID/0", "HMPID/1");
DECLARE_EQUIVALENT_FOR_INDEX_NG("MFTTracks/0", "MFTTracks/1");
} // namespace soa

namespace aod
{
namespace mctracklabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle); //! MC particle
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);     //! Bit mask to indicate detector mismatches (bit ON means mismatch). Bit 0-6: mismatch at ITS layer. Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64. Bit 10: TRD, bit 11: TOF, bit 15: indicates negative label
} // namespace mctracklabel

DECLARE_SOA_TABLE(McTrackLabels, "AOD", "MCTRACKLABEL", //! Table joined to the track table containing the MC index
                  mctracklabel::McParticleId, mctracklabel::McMask);
using McTrackLabel = McTrackLabels::iterator;

namespace mcmfttracklabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle); //! MC particle
DECLARE_SOA_COLUMN(McMask, mcMask, uint8_t);
} // namespace mcmfttracklabel

DECLARE_SOA_TABLE(McMFTTrackLabels, "AOD", "MCMFTTRACKLABEL", //! Table joined to the mft track table containing the MC index
                  mcmfttracklabel::McParticleId, mcmfttracklabel::McMask);
using McMFTTrackLabel = McMFTTrackLabels::iterator;

namespace mcfwdtracklabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle); //! MC particle
DECLARE_SOA_COLUMN(McMask, mcMask, uint8_t);
} // namespace mcfwdtracklabel

DECLARE_SOA_TABLE(McFwdTrackLabels, "AOD", "MCFWDTRACKLABEL", //! Table joined to the mft track table containing the MC index
                  mcfwdtracklabel::McParticleId, mcfwdtracklabel::McMask);
using McFwdTrackLabel = McFwdTrackLabels::iterator;

namespace mccalolabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle);               //! MC particle
DECLARE_SOA_ARRAY_INDEX_COLUMN(McParticle, mcParticle);         //! Array of MC particles that deposited energy in this calo cell
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);                   //! Bit mask to indicate detector mismatches (bit ON means mismatch). Bit 15: indicates negative label
DECLARE_SOA_COLUMN(AmplitudeA, amplitudeA, std::vector<float>); //! Energy fraction deposited by a particle inside this calo cell.
} // namespace mccalolabel

DECLARE_SOA_TABLE(McCaloLabels_000, "AOD", "MCCALOLABEL", //! Table joined to the calo table containing the MC index (version 000, Run 2 format)
                  mccalolabel::McParticleId, mccalolabel::McMask);
DECLARE_SOA_TABLE_VERSIONED(McCaloLabels_001, "AOD", "MCCALOLABEL", 1, //! Table joined to the calo table containing multiple MC indices and the amplitude fraction (version 001)
                            mccalolabel::McParticleIds, mccalolabel::AmplitudeA);
using McCaloLabels = McCaloLabels_000;
using McCaloLabel = McCaloLabels::iterator;

namespace mccollisionlabel
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision); //! MC collision
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);       //! Bit mask to indicate collision mismatches (bit ON means mismatch). Bit 15: indicates negative label
} // namespace mccollisionlabel

DECLARE_SOA_TABLE(McCollisionLabels, "AOD", "MCCOLLISLABEL", //! Table joined to the collision table containing the MC index
                  mccollisionlabel::McCollisionId, mccollisionlabel::McMask);
using McCollisionLabel = McCollisionLabels::iterator;

// --- HepMC ---
namespace hepmcxsection
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);    //! MC collision index
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short); //!
DECLARE_SOA_COLUMN(Accepted, accepted, uint64_t);      //! The number of events generated so far
DECLARE_SOA_COLUMN(Attempted, attempted, uint64_t);    //! The number of events attempted so far
DECLARE_SOA_COLUMN(XsectGen, xsectGen, float);         //! Cross section in pb
DECLARE_SOA_COLUMN(XsectErr, xsectErr, float);         //! Error associated with this cross section
DECLARE_SOA_COLUMN(PtHard, ptHard, float);             //! PT-hard (event scale, for pp collisions)
DECLARE_SOA_COLUMN(NMPI, nMPI, int);                   //! number of MPIs (for pp collisions)
DECLARE_SOA_COLUMN(ProcessId, processId, int);         //! process id from MC generator
} // namespace hepmcxsection

DECLARE_SOA_TABLE(HepMCXSections, "AOD", "HEPMCXSECTION", //! HepMC table for cross sections
                  o2::soa::Index<>, hepmcxsection::McCollisionId, hepmcxsection::GeneratorsID,
                  hepmcxsection::Accepted, hepmcxsection::Attempted, hepmcxsection::XsectGen,
                  hepmcxsection::XsectErr, hepmcxsection::PtHard, hepmcxsection::NMPI, hepmcxsection::ProcessId);
using HepMCXSection = HepMCXSections::iterator;

namespace hepmcpdfinfo
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);    //! MC collision index
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short); //!
DECLARE_SOA_COLUMN(Id1, id1, int);                     //! flavour code of first parton
DECLARE_SOA_COLUMN(Id2, id2, int);                     //! flavour code of second parton
DECLARE_SOA_COLUMN(PdfId1, pdfId1, int);               //! LHAPDF set id of first parton
DECLARE_SOA_COLUMN(PdfId2, pdfId2, int);               //! LHAPDF set id of second parton
DECLARE_SOA_COLUMN(X1, x1, float);                     //! fraction of beam momentum carried by first parton ("beam side")
DECLARE_SOA_COLUMN(X2, x2, float);                     //! fraction of beam momentum carried by second parton ("target side")
DECLARE_SOA_COLUMN(ScalePdf, scalePdf, float);         //! Q-scale used in evaluation of PDF's   (in GeV)
DECLARE_SOA_COLUMN(Pdf1, pdf1, float);                 //! PDF (id1, x1, Q) = x*f(x)
DECLARE_SOA_COLUMN(Pdf2, pdf2, float);                 //! PDF (id2, x2, Q) = x*f(x)
} // namespace hepmcpdfinfo

DECLARE_SOA_TABLE(HepMCPdfInfos, "AOD", "HEPMCPDFINFO", //! HepMC table for PDF infos
                  o2::soa::Index<>, hepmcpdfinfo::McCollisionId, hepmcpdfinfo::GeneratorsID,
                  hepmcpdfinfo::Id1, hepmcpdfinfo::Id2,
                  hepmcpdfinfo::PdfId1, hepmcpdfinfo::PdfId2,
                  hepmcpdfinfo::X1, hepmcpdfinfo::X2,
                  hepmcpdfinfo::ScalePdf, hepmcpdfinfo::Pdf1, hepmcpdfinfo::Pdf2);
using HepMCPdfInfo = HepMCPdfInfos::iterator;

namespace hepmcheavyion
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);                              //! MC collision index
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short);                           //!
DECLARE_SOA_COLUMN(NcollHard, ncollHard, int);                                   //! Number of hard scatterings
DECLARE_SOA_COLUMN(NpartProj, npartProj, int);                                   //! Number of projectile participants
DECLARE_SOA_COLUMN(NpartTarg, npartTarg, int);                                   //! Number of target participants
DECLARE_SOA_COLUMN(Ncoll, ncoll, int);                                           //! Number of NN (nucleon-nucleon) collisions
DECLARE_SOA_COLUMN(NNwoundedCollisions, nNwoundedCollisions, int);               //! Number of N-Nwounded collisions
DECLARE_SOA_COLUMN(NwoundedNCollisions, nwoundedNCollisions, int);               //! Number of Nwounded-N collisions
DECLARE_SOA_COLUMN(NwoundedNwoundedCollisions, nwoundedNwoundedCollisions, int); //! Number of Nwounded-Nwounded collisions
DECLARE_SOA_COLUMN(SpectatorNeutrons, spectatorNeutrons, int);                   //! Number of spectator neutrons
DECLARE_SOA_COLUMN(SpectatorProtons, spectatorProtons, int);                     //! Number of spectator protons
DECLARE_SOA_COLUMN(ImpactParameter, impactParameter, float);                     //! Impact Parameter(fm) of collision
DECLARE_SOA_COLUMN(EventPlaneAngle, eventPlaneAngle, float);                     //! Azimuthal angle of event plane
DECLARE_SOA_COLUMN(Eccentricity, eccentricity, float);                           //! eccentricity of participating nucleons in the transverse plane (as in phobos nucl-ex/0510031)
DECLARE_SOA_COLUMN(SigmaInelNN, sigmaInelNN, float);                             //! nucleon-nucleon inelastic (including diffractive) cross-section
DECLARE_SOA_COLUMN(Centrality, centrality, float);                               //! centrality (prcentile of geometric cross section)
} // namespace hepmcheavyion

DECLARE_SOA_TABLE(HepMCHeavyIons, "AOD", "HEPMCHEAVYION", //! HepMC table for cross sections
                  o2::soa::Index<>, hepmcheavyion::McCollisionId, hepmcheavyion::GeneratorsID,
                  hepmcheavyion::NcollHard, hepmcheavyion::NpartProj, hepmcheavyion::NpartTarg,
                  hepmcheavyion::Ncoll, hepmcheavyion::NNwoundedCollisions, hepmcheavyion::NwoundedNCollisions,
                  hepmcheavyion::NwoundedNwoundedCollisions, hepmcheavyion::SpectatorNeutrons,
                  hepmcheavyion::SpectatorProtons, hepmcheavyion::ImpactParameter, hepmcheavyion::EventPlaneAngle,
                  hepmcheavyion::Eccentricity, hepmcheavyion::SigmaInelNN, hepmcheavyion::Centrality);
using HepMCHeavyIon = HepMCHeavyIons::iterator;

// --- Matching between collisions and other tables through BC ---

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);        //!
DECLARE_SOA_ARRAY_INDEX_COLUMN(Collision, collisions); //!
DECLARE_SOA_INDEX_COLUMN(BC, bc);                      //!
DECLARE_SOA_INDEX_COLUMN(Zdc, zdc);                    //!
DECLARE_SOA_INDEX_COLUMN(FV0A, fv0a);                  //!
DECLARE_SOA_INDEX_COLUMN(FV0C, fv0c);                  //!
DECLARE_SOA_INDEX_COLUMN(FT0, ft0);                    //!
DECLARE_SOA_INDEX_COLUMN(FDD, fdd);                    //!
} // namespace indices

// First entry: Collision
#define INDEX_LIST_RUN2 indices::CollisionId, indices::ZdcId, indices::BCId, indices::FT0Id, indices::FV0AId, indices::FV0CId, indices::FDDId
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run2MatchedExclusive, BCs, "MA_RN2_EX", INDEX_LIST_RUN2); //!
DECLARE_SOA_INDEX_TABLE(Run2MatchedSparse, BCs, "MA_RN2_SP", INDEX_LIST_RUN2);              //!

#define INDEX_LIST_RUN3 indices::CollisionId, indices::ZdcId, indices::BCId, indices::FT0Id, indices::FV0AId, indices::FDDId
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run3MatchedExclusive, BCs, "MA_RN3_EX", INDEX_LIST_RUN3); //!
DECLARE_SOA_INDEX_TABLE(Run3MatchedSparse, BCs, "MA_RN3_SP", INDEX_LIST_RUN3);              //!

// First entry: BC
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(MatchedBCCollisionsExclusive, BCs, "MA_BCCOL_EX", //!
                                  indices::BCId, indices::CollisionId);
DECLARE_SOA_INDEX_TABLE(MatchedBCCollisionsSparse, BCs, "MA_BCCOL_SP", //!
                        indices::BCId, indices::CollisionId);

DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(MatchedBCCollisionsExclusiveMulti, BCs, "MA_BCCOLS_EX", //!
                                  indices::BCId, indices::CollisionIds);
DECLARE_SOA_INDEX_TABLE(MatchedBCCollisionsSparseMulti, BCs, "MA_BCCOLS_SP", //!
                        indices::BCId, indices::CollisionIds);

DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run3MatchedToBCExclusive, BCs, "MA_RN3_BC_EX", //!
                                  indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FDDId);
DECLARE_SOA_INDEX_TABLE(Run3MatchedToBCSparse, BCs, "MA_RN3_BC_SP", //!
                        indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FDDId);

DECLARE_SOA_INDEX_TABLE(Run2MatchedToBCSparse, BCs, "MA_RN2_BC_SP", //!
                        indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FV0CId, indices::FDDId);

} // namespace aod
namespace soa
{
// equivalences
DECLARE_EQUIVALENT_FOR_INDEX(aod::StoredTracks, aod::McTrackLabels);
DECLARE_EQUIVALENT_FOR_INDEX(aod::StoredTracksIU, aod::McTrackLabels);
DECLARE_EQUIVALENT_FOR_INDEX(aod::Collisions, aod::McCollisionLabels);
// Joins with collisions (only for sparse ones)
// NOTE: index table needs to be always last argument
extern template struct JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::Collisions, aod::Run2MatchedSparse>;
extern template struct JoinFull<o2::aod::Hash<"JOIN/0"_h>, aod::Collisions, aod::Run3MatchedSparse>;
} // namespace soa
namespace aod
{
using CollisionMatchedRun2Sparse = soa::Join<Collisions, Run2MatchedSparse>::iterator;
using CollisionMatchedRun3Sparse = soa::Join<Collisions, Run3MatchedSparse>::iterator;

} // namespace aod

} // namespace o2
#endif // O2_FRAMEWORK_ANALYSISDATAMODEL_H_
