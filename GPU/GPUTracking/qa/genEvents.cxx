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

/// \file genEvents.cxx
/// \author Sergey Gorbunov

#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

#include "Rtypes.h"
#include "TRandom.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPad.h"

#include <iostream>
#include <iomanip>
#include <limits>

#include "genEvents.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCMCInfo.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUParam.inc"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMMerger.h"
#include "GPUChainTracking.h"

#include "../utils/qconfig.h"

using namespace o2::gpu;
using namespace std;
namespace o2::gpu
{
extern GPUSettingsStandalone configStandalone;
}

int32_t genEvents::GetSector(double GlobalPhi)
{
  double phi = GlobalPhi;
  //  std::cout<<" GetSector: phi = "<<phi<<std::endl;

  if (phi >= mTwoPi) {
    phi -= mTwoPi;
  }
  if (phi < 0) {
    phi += mTwoPi;
  }
  return (int32_t)(phi / mSectorDAngle);
}

int32_t genEvents::GetDSector(double LocalPhi) { return GetSector(LocalPhi + mSectorAngleOffset); }

double genEvents::GetSectorAngle(int32_t iSector) { return mSectorAngleOffset + iSector * mSectorDAngle; }

int32_t genEvents::RecalculateSector(GPUTPCGMPhysicalTrackModel& t, int32_t& iSector)
{
  double phi = atan2(t.GetY(), t.GetX());
  //  std::cout<<" recalculate: phi = "<<phi<<std::endl;
  int32_t dSector = GetDSector(phi);

  if (dSector == 0) {
    return 0; // nothing to do
  }
  //  std::cout<<" dSector = "<<dSector<<std::endl;
  double dAlpha = dSector * mSectorDAngle;
  // rotate track on angle dAlpha

  t.Rotate(dAlpha);

  iSector += dSector;
  if (iSector >= 18) {
    iSector -= 18;
  }
  return 1;
}

double genEvents::GetGaus(double sigma)
{
  double x = 0;
  do {
    x = gRandom->Gaus(0., sigma);
    if (fabs(x) <= 3.5 * sigma) {
      break;
    }
  } while (1);
  return x;
}

void genEvents::InitEventGenerator()
{
  const char* rows[3] = {"0-63", "128-159", "64-127"};
  for (int32_t i = 0; i < 3; i++) {
    for (int32_t j = 0; j < 2; j++) {
      char name[1024], title[1024];

      snprintf(name, 1024, "clError%s%d", (j == 0 ? "Y" : "Z"), i);

      snprintf(title, 1024, "Cluster %s Error for rows %s", (j == 0 ? "Y" : "Z"), rows[i]);

      mClusterError[i][j] = new TH1F(name, title, 1000, 0., .7);
      mClusterError[i][j]->GetXaxis()->SetTitle("Cluster Error [cm]");
    }
  }
}

void genEvents::FinishEventGenerator()
{
  TFile* tout = new TFile("generator.root", "RECREATE");
  TCanvas* c = new TCanvas("ClusterErrors", "Cluste rErrors", 0, 0, 700, 700. * 2. / 3.);
  c->Divide(3, 2);
  int32_t ipad = 1;
  for (int32_t j = 0; j < 2; j++) {
    for (int32_t i = 0; i < 3; i++) {
      c->cd(ipad++);
      int32_t k = i;
      if (i == 1) {
        k = 2;
      }
      if (i == 2) {
        k = 1;
      }
      if (tout) {
        mClusterError[k][j]->Write();
      }
      gPad->SetLogy();
      mClusterError[k][j]->Draw();
      // delete mClusterError[i][j];
      // mClusterError[i][j]=0;
    }
  }
  c->Print("plots/clusterErrors.pdf");
  delete c;
  if (tout) {
    tout->Close();
    delete tout;
  }
}

int32_t genEvents::GenerateEvent(const GPUParam& param, char* filename)
{
  mRec->ClearIOPointers();
  static int32_t iEvent = -1;
  iEvent++;
  if (iEvent == 0) {
    gRandom->SetSeed(configStandalone.seed);
  }

  int32_t nTracks = configStandalone.EG.numberOfTracks; // Number of MC tracks, must be at least as large as the largest fMCID assigned above
  cout << "NTracks " << nTracks << endl;
  std::vector<GPUTPCMCInfo> mcInfo(nTracks);
  memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));

  // double Bz = param.ConstBz();
  // std::cout<<"Bz[kG] = "<<param.BzkG()<<std::endl;

  GPUTPCGMPropagator prop;
  {
    prop.SetToyMCEventsFlag(kTRUE);
    const GPUTPCGMMerger& merger = mRec->GetTPCMerger();
    prop.SetPolynomialField(&merger.Param().polynomialField);
  }

  // Bz*=o2::gpu::gpu_common_constants::kCLight;

  std::vector<GenCluster> vClusters;
  int32_t clusterId = 0; // Here we count up the cluster ids we fill (must be unique).
  // gRandom->SetSeed(0);
  // uint32_t seed = gRandom->GetSeed();

  for (int32_t itr = 0; itr < nTracks; itr++) {
    // std::cout<<"Track "<<itr<<":"<<std::endl;
    // gRandom->SetSeed(seed);

    mcInfo[itr].pid = -100; //-100: Unknown / other, 0: Electron, 1, Muon, 2: Pion, 3: Kaon, 4: Proton
    mcInfo[itr].charge = 1;
    mcInfo[itr].prim = 1;          // Primary particle
    mcInfo[itr].primDaughters = 0; // Primary particle with daughters in the TPC
    mcInfo[itr].x = 0;             // Position of MC track at entry of TPC / first hit in the TPC
    mcInfo[itr].y = 0;
    mcInfo[itr].z = 0;
    mcInfo[itr].pX = 0; // Momentum of MC track at that position
    mcInfo[itr].pY = 0;
    mcInfo[itr].pZ = 0;

    GPUTPCGMPhysicalTrackModel t;
    double dphi = mTwoPi / nTracks;
    double phi = mSectorAngleOffset + dphi * itr;
    double eta = gRandom->Uniform(-1.5, 1.5);

    double theta = 2 * std::atan(1. / exp(eta));
    double lambda = theta - M_PI / 2;
    // double theta = gRandom->Uniform(-60,60)*M_PI/180.;
    double pt = .08 * std::pow(10, gRandom->Uniform(0, 2.2));

    double q = 1.;
    int32_t iSector = GetSector(phi);
    phi = phi - GetSectorAngle(iSector);

    // std::cout<<"phi = "<<phi<<std::endl;
    double x0 = cosf(phi);
    double y0 = sinf(phi);
    double z0 = tanf(lambda);
    t.Set(x0, y0, z0, pt * x0, pt * y0, pt * z0, q);

    if (RecalculateSector(t, iSector) != 0) {
      std::cout << "Initial sector wrong!!!" << std::endl;
      // exit(0);
    }

    for (int32_t iRow = 0; iRow < GPUCA_ROW_COUNT; iRow++) {
      // if( iRow>=50 ) break; //SG!!!
      float xRow = GPUTPCGeometry::Row2X(iRow);
      // transport to row
      int32_t err = 0;
      for (int32_t itry = 0; itry < 1; itry++) {
        float B[3];
        prop.GetBxByBz(GetSectorAngle(iSector), t.GetX(), t.GetY(), t.GetZ(), B);
        float dLp = 0;
        err = t.PropagateToXBxByBz(xRow, B[0], B[1], B[2], dLp);
        if (err) {
          std::cout << "Can not propagate to x = " << xRow << std::endl;
          t.Print();
          break;
        }
        if (fabsf(t.GetZ()) >= GPUTPCGeometry::TPCLength()) {
          std::cout << "Can not propagate to x = " << xRow << ": Z outside the volume" << std::endl;
          t.Print();
          err = -1;
          break;
        }
        // rotate track coordinate system to current sector
        int32_t isNewSector = RecalculateSector(t, iSector);
        if (!isNewSector) {
          break;
        } else {
          std::cout << "track " << itr << ": new sector " << iSector << " at row " << iRow << std::endl;
        }
      }
      if (err) {
        break;
      }
      // std::cout<<" track "<<itr<<": Sector "<<iSector<<" row "<<iRow<<" params :"<<std::endl;
      // t.Print();
      // track at row iRow, sector iSector
      if (iRow == 0) { // store MC track at first row
        // std::cout<<std::setprecision( 20 );
        // std::cout<<"track "<<itr<<": x "<<t.X()<<" y "<<t.Y()<<" z "<<t.Z()<<std::endl;
        GPUTPCGMPhysicalTrackModel tg(t); // global coordinates
        tg.Rotate(-GetSectorAngle(iSector));

        mcInfo[itr].pid = 2; // pion
        mcInfo[itr].charge = 3 * q;
        mcInfo[itr].x = tg.GetX(); // Position of MC track at entry of TPC / first hit in the TPC
        mcInfo[itr].y = tg.GetY();
        mcInfo[itr].z = tg.GetZ();
        mcInfo[itr].pX = tg.GetPx(); // Momentum of MC track at that position
        mcInfo[itr].pY = tg.GetPy();
        mcInfo[itr].pZ = tg.GetPz();
        // std::cout<<" mc Z = "<<tg.GetZ()<<std::endl;
      }

      // create cluster
      GenCluster c;
      float sigmaY = 0.3;
      float sigmaZ = 0.5;
      const int32_t rowType = iRow < 64 ? 0 : iRow < 128 ? 2 : 1;
      t.UpdateValues();
      param.GetClusterErrors2(iSector, rowType, t.GetZ(), t.GetSinPhi(), t.GetDzDs(), -1.f, 0.f, 0.f, sigmaY, sigmaZ);
      sigmaY = std::sqrt(sigmaY);
      sigmaZ = std::sqrt(sigmaZ);
      mClusterError[rowType][0]->Fill(sigmaY);
      mClusterError[rowType][1]->Fill(sigmaZ);
      // std::cout<<sigmaY<<" "<<sigmaY<<std::endl;
      // if( sigmaY > 0.5 ) sigmaY = 0.5;
      // if( sigmaZ > 0.5 ) sigmaZ = 0.5;
      c.sector = (t.GetZ() >= 0.) ? iSector : iSector + 18;
      c.row = iRow;
      c.mcID = itr;
      c.x = t.GetX();
      c.y = t.GetY() + GetGaus(sigmaY);
      c.z = t.GetZ() + GetGaus(sigmaZ);
      c.id = clusterId++;
      vClusters.push_back(c);
    } // iRow
  } // itr

  std::vector<AliHLTTPCClusterMCLabel> labels;

  std::unique_ptr<GPUTPCClusterData> clSectors[GPUChainTracking::NSECTORS];

  for (int32_t iSector = 0; iSector < (int32_t)GPUChainTracking::NSECTORS; iSector++) // HLT Sector numbering, sectors go from 0 to 35, all spanning all rows from 0 to 158.
  {
    int32_t nNumberOfHits = 0;
    for (uint32_t i = 0; i < vClusters.size(); i++) {
      if (vClusters[i].sector == iSector) {
        nNumberOfHits++;
      }
    }
    // For every sector we first have to fill the number of hits in this sector to the file
    mRec->mIOPtrs.nClusterData[iSector] = nNumberOfHits;

    GPUTPCClusterData* clusters = new GPUTPCClusterData[nNumberOfHits];
    clSectors[iSector].reset(clusters);
    int32_t icl = 0;
    for (uint32_t i = 0; i < vClusters.size(); i++) {
      GenCluster& c = vClusters[i];
      if (c.sector == iSector) {
        clusters[icl].id = c.id;
        clusters[icl].row = c.row; // We fill one hit per TPC row
        clusters[icl].x = c.x;
        clusters[icl].y = c.y;
        clusters[icl].z = c.z;
        clusters[icl].amp = 100; // Arbitrary amplitude
        icl++;
        AliHLTTPCClusterMCLabel clusterLabel;
        for (int32_t j = 0; j < 3; j++) {
          clusterLabel.fClusterID[j].fMCID = -1;
          clusterLabel.fClusterID[j].fWeight = 0;
        }
        clusterLabel.fClusterID[0].fMCID = c.mcID;
        clusterLabel.fClusterID[0].fWeight = 1;
        labels.push_back(clusterLabel);
      }
    }
    mRec->mIOPtrs.clusterData[iSector] = clusters;
  }

  // Create vector with cluster MC labels, clusters are counter from 0 to clusterId in the order they have been written above. No separation in sectors.

  mRec->mIOPtrs.nMCLabelsTPC = labels.size();
  mRec->mIOPtrs.mcLabelsTPC = labels.data();

  mRec->mIOPtrs.nMCInfosTPC = mcInfo.size();
  mRec->mIOPtrs.mcInfosTPC = mcInfo.data();
  static const GPUTPCMCInfoCol mcColInfo = {0, (uint32_t)mcInfo.size()};
  mRec->mIOPtrs.mcInfosTPCCol = &mcColInfo;
  mRec->mIOPtrs.nMCInfosTPCCol = 1;

  mRec->DumpData(filename);
  labels.clear();
  mcInfo.clear();
  return (0);
}

void genEvents::RunEventGenerator(GPUChainTracking* rec)
{
  std::unique_ptr<genEvents> gen(new genEvents(rec));
  char dirname[256];
  snprintf(dirname, 256, "events/%s/", configStandalone.eventsDir);
  mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  rec->DumpSettings(dirname);

  gen->InitEventGenerator();

  for (int32_t i = 0; i < (configStandalone.nEvents == -1 ? 10 : configStandalone.nEvents); i++) {
    GPUInfo("Generating event %d/%d", i, configStandalone.nEvents == -1 ? 10 : configStandalone.nEvents);
    snprintf(dirname, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.eventsDir, i);
    gen->GenerateEvent(rec->GetParam(), dirname);
  }
  gen->FinishEventGenerator();
}
