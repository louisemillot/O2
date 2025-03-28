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

/// \file GPUTPCClusterStatistics.cxx
/// \author David Rohr

#include "GPUTPCClusterStatistics.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <queue>

using namespace o2::gpu;

// Small helper to compute Huffman probabilities
namespace o2::gpu
{
namespace // anonymous
{
typedef std::vector<bool> HuffCode;
typedef std::map<uint32_t, HuffCode> HuffCodeMap;

class INode
{
 public:
  const double f;

  virtual ~INode() = default;

 protected:
  INode(double v) : f(v) {}
};

class InternalNode : public INode
{
 public:
  INode* const left;
  INode* const right;

  InternalNode(INode* c0, INode* c1) : INode(c0->f + c1->f), left(c0), right(c1) {}
  ~InternalNode() override
  {
    delete left;
    delete right;
  }
};

class LeafNode : public INode
{
 public:
  const uint32_t c;

  LeafNode(double v, uint32_t w) : INode(v), c(w) {}
};

struct NodeCmp {
  bool operator()(const INode* lhs, const INode* rhs) const { return lhs->f > rhs->f; }
};

INode* BuildTree(const double* frequencies, uint32_t UniqueSymbols)
{
  std::priority_queue<INode*, std::vector<INode*>, NodeCmp> trees;

  for (uint32_t i = 0; i < UniqueSymbols; i++) {
    if (frequencies[i] != 0) {
      trees.push(new LeafNode(frequencies[i], i));
    }
  }
  while (trees.size() > 1) {
    INode* childR = trees.top();
    trees.pop();

    INode* childL = trees.top();
    trees.pop();

    INode* parent = new InternalNode(childR, childL);
    trees.push(parent);
  }
  return trees.top();
}

void GenerateCodes(const INode* node, const HuffCode& prefix, HuffCodeMap& outCodes)
{
  if (const LeafNode* lf = dynamic_cast<const LeafNode*>(node)) {
    outCodes[lf->c] = prefix;
  } else if (const InternalNode* in = dynamic_cast<const InternalNode*>(node)) {
    HuffCode leftPrefix = prefix;
    leftPrefix.push_back(false);
    GenerateCodes(in->left, leftPrefix, outCodes);

    HuffCode rightPrefix = prefix;
    rightPrefix.push_back(true);
    GenerateCodes(in->right, rightPrefix, outCodes);
  }
}
} // anonymous namespace
} // namespace o2::gpu

void GPUTPCClusterStatistics::RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param)
{
  uint32_t decodingErrors = 0;
  o2::tpc::ClusterNativeAccess clustersNativeDecoded;
  std::vector<o2::tpc::ClusterNative> clusterBuffer;
  GPUInfo("Compression statistics, decoding: %d attached (%d tracks), %d unattached", clustersCompressed->nAttachedClusters, clustersCompressed->nTracks, clustersCompressed->nUnattachedClusters);
  auto allocator = [&clusterBuffer](size_t size) {clusterBuffer.resize(size); return clusterBuffer.data(); };
  mDecoder.decompress(clustersCompressed, clustersNativeDecoded, allocator, param, true);
  std::vector<o2::tpc::ClusterNative> tmpClusters;
  if (param.rec.tpc.rejectionStrategy == GPUSettings::RejectionNone) { // verification does not make sense if we reject clusters during compression
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
        if (clustersNative->nClusters[i][j] != clustersNativeDecoded.nClusters[i][j]) {
          GPUError("Number of clusters mismatch sector %u row %u: expected %d v.s. decoded %d", i, j, clustersNative->nClusters[i][j], clustersNativeDecoded.nClusters[i][j]);
          decodingErrors++;
          continue;
        }
        tmpClusters.resize(clustersNative->nClusters[i][j]);
        for (uint32_t k = 0; k < clustersNative->nClusters[i][j]; k++) {
          tmpClusters[k] = clustersNative->clusters[i][j][k];
          if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionTruncate) {
            GPUTPCCompression::truncateSignificantBitsChargeMax(tmpClusters[k].qMax, param);
            GPUTPCCompression::truncateSignificantBitsCharge(tmpClusters[k].qTot, param);
            GPUTPCCompression::truncateSignificantBitsWidth(tmpClusters[k].sigmaPadPacked, param);
            GPUTPCCompression::truncateSignificantBitsWidth(tmpClusters[k].sigmaTimePacked, param);
          }
        }
        std::sort(tmpClusters.begin(), tmpClusters.end());
        for (uint32_t k = 0; k < clustersNative->nClusters[i][j]; k++) {
          const o2::tpc::ClusterNative& c1 = tmpClusters[k];
          const o2::tpc::ClusterNative& c2 = clustersNativeDecoded.clusters[i][j][k];
          if (c1.timeFlagsPacked != c2.timeFlagsPacked || c1.padPacked != c2.padPacked || c1.sigmaTimePacked != c2.sigmaTimePacked || c1.sigmaPadPacked != c2.sigmaPadPacked || c1.qMax != c2.qMax || c1.qTot != c2.qTot) {
            if (decodingErrors++ < 100) {
              GPUWarning("Cluster mismatch: sector %2u row %3u hit %5u: %6d %3d %4d %3d %3d %4d %4d", i, j, k, (int32_t)c1.getTimePacked(), (int32_t)c1.getFlags(), (int32_t)c1.padPacked, (int32_t)c1.sigmaTimePacked, (int32_t)c1.sigmaPadPacked, (int32_t)c1.qMax, (int32_t)c1.qTot);
              GPUWarning("%45s %6d %3d %4d %3d %3d %4d %4d", "", (int32_t)c2.getTimePacked(), (int32_t)c2.getFlags(), (int32_t)c2.padPacked, (int32_t)c2.sigmaTimePacked, (int32_t)c2.sigmaPadPacked, (int32_t)c2.qMax, (int32_t)c2.qTot);
            }
          }
        }
      }
    }
    if (decodingErrors) {
      mDecodingError = true;
      GPUWarning("Errors during cluster decoding %u\n", decodingErrors);
    } else {
      GPUInfo("Cluster decoding verification: PASSED");
    }
  }

  FillStatistic(mPqTotA, clustersCompressed->qTotA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPqMaxA, clustersCompressed->qMaxA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPflagsA, clustersCompressed->flagsA, clustersCompressed->nAttachedClusters);
  FillStatistic(mProwDiffA, clustersCompressed->rowDiffA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPsectorLegDiffA, clustersCompressed->sliceLegDiffA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPpadResA, clustersCompressed->padResA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPtimeResA, clustersCompressed->timeResA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPsigmaPadA, clustersCompressed->sigmaPadA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPsigmaTimeA, clustersCompressed->sigmaTimeA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPqPtA, clustersCompressed->qPtA, clustersCompressed->nTracks);
  FillStatistic(mProwA, clustersCompressed->rowA, clustersCompressed->nTracks);
  FillStatistic(mPsectorA, clustersCompressed->sliceA, clustersCompressed->nTracks);
  FillStatistic(mPtimeA, clustersCompressed->timeA, clustersCompressed->nTracks);
  FillStatistic(mPpadA, clustersCompressed->padA, clustersCompressed->nTracks);
  FillStatistic(mPqTotU, clustersCompressed->qTotU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPqMaxU, clustersCompressed->qMaxU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPflagsU, clustersCompressed->flagsU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPpadDiffU, clustersCompressed->padDiffU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPtimeDiffU, clustersCompressed->timeDiffU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPsigmaPadU, clustersCompressed->sigmaPadU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPsigmaTimeU, clustersCompressed->sigmaTimeU, clustersCompressed->nUnattachedClusters);
  FillStatistic<uint16_t, 1>(mPnTrackClusters, clustersCompressed->nTrackClusters, clustersCompressed->nTracks);
  FillStatistic<uint32_t, 1>(mPnSectorRowClusters, clustersCompressed->nSliceRowClusters, clustersCompressed->nSliceRows);
  FillStatisticCombined(mPsigmaA, clustersCompressed->sigmaPadA, clustersCompressed->sigmaTimeA, clustersCompressed->nAttachedClusters, P_MAX_SIGMA);
  FillStatisticCombined(mPsigmaU, clustersCompressed->sigmaPadU, clustersCompressed->sigmaTimeU, clustersCompressed->nUnattachedClusters, P_MAX_SIGMA);
  FillStatisticCombined(mPQA, clustersCompressed->qMaxA, clustersCompressed->qTotA, clustersCompressed->nAttachedClusters, P_MAX_QMAX);
  FillStatisticCombined(mPQU, clustersCompressed->qMaxU, clustersCompressed->qTotU, clustersCompressed->nUnattachedClusters, P_MAX_QMAX);
  FillStatisticCombined(mProwSectorA, clustersCompressed->rowDiffA, clustersCompressed->sliceLegDiffA, clustersCompressed->nAttachedClustersReduced, GPUCA_ROW_COUNT);
  mNTotalClusters += clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters;
}

void GPUTPCClusterStatistics::Finish()
{
  if (mDecodingError) {
    GPUError("-----------------------------------------\nERROR - INCORRECT CLUSTER DECODING!\n-----------------------------------------");
  }
  if (mNTotalClusters == 0) {
    return;
  }

  GPUInfo("\nRunning cluster compression entropy statistics");
  double eQ = Analyze(mPqTotA, "qTot Attached", false);
  eQ += Analyze(mPqMaxA, "qMax Attached", false);
  Analyze(mPflagsA, "flags Attached");
  double eRowSector = Analyze(mProwDiffA, "rowDiff Attached", false);
  eRowSector += Analyze(mPsectorLegDiffA, "sectorDiff Attached", false);
  Analyze(mPpadResA, "padRes Attached");
  Analyze(mPtimeResA, "timeRes Attached");
  double eSigma = Analyze(mPsigmaPadA, "sigmaPad Attached", false);
  eSigma += Analyze(mPsigmaTimeA, "sigmaTime Attached", false);
  Analyze(mPqPtA, "qPt Attached");
  Analyze(mProwA, "row Attached");
  Analyze(mPsectorA, "sector Attached");
  Analyze(mPtimeA, "time Attached");
  Analyze(mPpadA, "pad Attached");
  eQ += Analyze(mPqTotU, "qTot Unattached", false);
  eQ += Analyze(mPqMaxU, "qMax Unattached", false);
  Analyze(mPflagsU, "flags Unattached");
  Analyze(mPpadDiffU, "padDiff Unattached");
  Analyze(mPtimeDiffU, "timeDiff Unattached");
  eSigma += Analyze(mPsigmaPadU, "sigmaPad Unattached", false);
  eSigma += Analyze(mPsigmaTimeU, "sigmaTime Unattached", false);
  Analyze(mPnTrackClusters, "nClusters in Track");
  Analyze(mPnSectorRowClusters, "nClusters in Row");
  double eSigmaCombined = Analyze(mPsigmaA, "combined sigma Attached");
  eSigmaCombined += Analyze(mPsigmaU, "combined sigma Unattached");
  double eQCombined = Analyze(mPQA, "combined Q Attached");
  eQCombined += Analyze(mPQU, "combined Q Unattached");
  double eRowSectorCombined = Analyze(mProwSectorA, "combined row/sector Attached");

  GPUInfo("Combined Row/Sector: %6.4f --> %6.4f (%6.4f%%)", eRowSector, eRowSectorCombined, eRowSector > 1e-1 ? (100. * (eRowSector - eRowSectorCombined) / eRowSector) : 0.f);
  GPUInfo("Combined Sigma: %6.4f --> %6.4f (%6.4f%%)", eSigma, eSigmaCombined, eSigma > 1e-3 ? (100. * (eSigma - eSigmaCombined) / eSigma) : 0.f);
  GPUInfo("Combined Q: %6.4f --> %6.4f (%6.4f%%)", eQ, eQCombined, eQ > 1e-3 ? (100. * (eQ - eQCombined) / eQ) : 0.f);

  printf("\nCombined Entropy: %7.4f   (Size %'13.0f, %'zu clusters)\nCombined Huffman: %7.4f   (Size %'13.0f, %f%%)\n\n", mEntropy / mNTotalClusters, mEntropy, mNTotalClusters, mHuffman / mNTotalClusters, mHuffman, 100. * (mHuffman - mEntropy) / mHuffman);
}

float GPUTPCClusterStatistics::Analyze(std::vector<int32_t>& p, const char* name, bool count)
{
  double entropy = 0.;
  double huffmanSize = 0;

  std::vector<double> prob(p.size());
  double log2 = log(2.);
  size_t total = 0;
  for (uint32_t i = 0; i < p.size(); i++) {
    total += p[i];
  }
  if (total) {
    for (uint32_t i = 0; i < prob.size(); i++) {
      if (p[i]) {
        prob[i] = (double)p[i] / total;
        double I = -log(prob[i]) / log2;
        double H = I * prob[i];

        entropy += H;
      }
    }

    INode* root = BuildTree(prob.data(), prob.size());

    HuffCodeMap codes;
    GenerateCodes(root, HuffCode(), codes);
    delete root;

    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); it++) {
      huffmanSize += it->second.size() * prob[it->first];
    }

    if (count) {
      mEntropy += entropy * total;
      mHuffman += huffmanSize * total;
    }
  }
  GPUInfo("Size: %30s: Entropy %7.4f Huffman %7.4f (Count) %9ld", name, entropy, huffmanSize, (int64_t)total);
  return entropy;
}

template <class T, int32_t I>
void GPUTPCClusterStatistics::FillStatistic(std::vector<int32_t>& p, const T* ptr, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    uint32_t val = ptr[i];
    if (val >= p.size()) {
      if (I) {
        p.resize(val + 1);
      } else {
        GPUError("Invalid Value: %d >= %d", val, (int32_t)p.size());
        continue;
      }
    }
    p[val]++;
  }
}

template <class T, class S, int32_t I>
void GPUTPCClusterStatistics::FillStatisticCombined(std::vector<int32_t>& p, const T* ptr1, const S* ptr2, size_t n, int32_t max1)
{
  for (size_t i = 0; i < n; i++) {
    uint32_t val = ptr1[i] + ptr2[i] * max1;
    if (val >= p.size()) {
      if (I) {
        p.resize(val + 1);
      } else {
        GPUError("Invalid Value: %d >= %d", val, (int32_t)p.size());
        continue;
      }
    }
    p[val]++;
  }
}
