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

/// \file standalone-cluster-dump-entropy-analysed.cxx
/// \author David Rohr

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <queue>
#include <map>
#include <iterator>
#include <algorithm>
#include <iostream>

const int32_t sort_method = 1; // 0 No sorting, 1 sort after pad, 2 sort after time, 3/4 mixed methods favoring pad / time
const int32_t sector_diff = 1;
const int32_t row_diff = 1;
const int32_t pad_diff = 1;
const int32_t time_diff = 1;
const int32_t res_diff = 0;
const int32_t approximate_qtot = 0;
const int32_t combine_maxtot = 1;
const int32_t combine_sigmapadtime = 1;
const int32_t track_based = 1;
const int32_t track_avgtot = track_based && 0;
const int32_t track_avgmax = track_based && 0;
const int32_t track_diffqtot = track_based && 0;
const int32_t track_diffqmax = track_based && 0;
const int32_t track_separate_q = track_based && 1;
const int32_t track_diffsigma = track_based && 0;
const int32_t track_separate_sigma = track_based && 1;
const int32_t truncate_bits = 1;
const int32_t separate_sectors = 0;
const int32_t separate_patches = 0;
const int32_t separate_sides = 0;
const int32_t full_row_numbers = 1;
const int32_t distinguish_rows = 0;
const int32_t optimized_negative_values = 1;

const int32_t print_clusters = 0;

const char* file = "clusters-pbpb.dump";
const int32_t max_clusters = 2000000;

const int32_t truncate_sigma = 3;
const int32_t truncate_charge = 4;

const int32_t sort_pad_mixed_bins = 100;
const int32_t sort_time_mixed_bins = 400;

#define EVENT 0
#define SECTOR 1
#define PATCH 2
#define ROW 3
#define PAD 4
#define TIME 5
#define SIGMA_PAD 6
#define SIGMA_TIME 7
#define QMAX 8
#define QTOT 9
#define FLAG_PADTIME 10
#define CLUSTER_ID 11
#define RES_PAD 12
#define RES_TIME 13
#define AVG_TOT 14
#define AVG_MAX 15
#define QMAX_QTOT 16
#define SIGMA_PAD_TIME 17
#define DIFF_SIGMA_PAD 18
#define DIFF_SIGMA_TIME 19
#define DIFF_SIGMA_PAD_TIME 20
#define AVG_TOT_MAX 21
#define ROW_TRACK_FIRST 22
#define ROW_TRACK 23

#define PAD_80 24
#define PAD_92 25
#define PAD_104 26
#define PAD_116 27
#define PAD_128 28
#define PAD_140 29

const int32_t rr = optimized_negative_values && 0 ? 13 : 14; // We can make them all 14 for convenience, the encoding will handle it

const uint32_t field_bits[] = {0, 6, 0, 8, 14, 15, 8, 8, 10, 16, 2, 0, 14, 15, 16, 10, 26, 16, 8, 8, 16, 26, 8, 8, rr, rr, rr, rr, rr, 14};
const uint32_t significant_bits[] = {0, 6, 0, 8, 14, 15, truncate_sigma, truncate_sigma, truncate_charge, truncate_charge, 2, 0, 14, 15, truncate_charge, truncate_charge, 26, 16, truncate_sigma, truncate_sigma, 16, 26, 8, 8, rr, rr, rr, rr, rr, 14};
const int32_t nFields = sizeof(field_bits) / sizeof(field_bits[0]);
const char* field_names[] = {"event", "sector", "patch", "row", "pad", "time", "sigmaPad", "sigmaTime", "qmax", "qtot", "flagPadTime", "trackID", "resTrackPad",
                             "resTrackTime", "trackQTot", "trackQMax", "qmaxtot", "sigmapadtime", "diffsigmapad", "diffsigmatime", "diffsigmapadtime", "tracktotmax", "trackfirstrow", "trackrow", "pad_80", "pad_92",
                             "pad_104", "pad_116", "pad_128", "pad_140"};

union cluster_struct {
  struct
  {
    uint32_t event, sector, patch, row, pad, time, sigmaPad, sigmaTime, qmax, qtot, splitPadTime;
    int32_t trackID;
    uint32_t resPad, resTime, avgtot, avgmax;
  };
  uint32_t vals[16];
};

int32_t fgRows[6][2] = {{0, 30}, {30, 62}, {63, 90}, {90, 116}, {117, 139}, {139, 158}};
int32_t fgNRows[6] = {31, 33, 28, 27, 23, 20};

int32_t fgNPads[159] = {68, 68, 68, 68, 70, 70, 70, 72, 72, 72, 74, 74, 74, 76, 76, 76, 78, 78, 78, 80, 80, 80, 82, 82, 82, 84, 84, 84, 86, 86, 86, 88, 88, 88, 90, 90, 90, 92, 92, 92, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 100, 102,
                        102, 102, 104, 104, 104, 106, 106, 106, 108, 108, 74, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 80, 82, 82, 82, 84, 84, 84, 86, 86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 96, 96, 96, 96, 98, 98, 98, 100,
                        100, 100, 100, 102, 102, 102, 104, 104, 104, 106, 106, 106, 106, 108, 108, 108, 110, 110, 110, 110, 112, 112, 114, 114, 114, 116, 116, 118, 118, 120, 120, 122, 122, 122, 124, 124, 126, 126, 128, 128, 130, 130, 130, 132, 132, 134, 134, 136, 136, 138, 138, 138, 140};

int32_t fgNPadsMod[159] = {80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104,
                           104, 104, 104, 104, 104, 116, 116, 116, 116, 116, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104,
                           104, 104, 104, 104, 104, 104, 104, 104, 104, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 128, 128, 128, 128, 128, 128, 128, 128, 128, 126, 126, 128, 128, 140, 140, 140, 140, 140, 134, 134, 140, 140, 140, 140, 140, 140};

// ---------------------------------- HUFFMAN TREE

typedef std::vector<bool> HuffCode;
typedef std::map<uint32_t, HuffCode> HuffCodeMap;

class INode
{
 public:
  const double f;

  virtual ~INode() {}

 protected:
  INode(double f) : f(f) {}
};

class InternalNode : public INode
{
 public:
  INode* const left;
  INode* const right;

  InternalNode(INode* c0, INode* c1) : INode(c0->f + c1->f), left(c0), right(c1) {}
  ~InternalNode()
  {
    delete left;
    delete right;
  }
};

class LeafNode : public INode
{
 public:
  const uint32_t c;

  LeafNode(double f, uint32_t c) : INode(f), c(c) {}
};

struct NodeCmp {
  bool operator()(const INode* lhs, const INode* rhs) const { return lhs->f > rhs->f; }
};

INode* BuildTree(const double* frequencies, uint32_t UniqueSymbols)
{
  std::priority_queue<INode*, std::vector<INode*>, NodeCmp> trees;

  for (int32_t i = 0; i < UniqueSymbols; i++) {
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

//--------------------------------------------- END HUFFMAN

bool clustercompare_padtime(cluster_struct a, cluster_struct b) { return (a.pad < b.pad || (a.pad == b.pad && a.time < b.time)); }

bool clustercompare_timepad(cluster_struct a, cluster_struct b) { return (a.time < b.time || (a.time == b.time && a.pad < b.pad)); }

bool clustercompare_padtime_mixed(cluster_struct a, cluster_struct b) { return (a.pad / sort_pad_mixed_bins < b.pad / sort_pad_mixed_bins || (a.pad / sort_pad_mixed_bins == b.pad / sort_pad_mixed_bins && a.time < b.time)); }

bool clustercompare_timepad_mixed(cluster_struct a, cluster_struct b) { return (a.time / sort_time_mixed_bins < b.time / sort_time_mixed_bins || (a.time / sort_time_mixed_bins == b.time / sort_time_mixed_bins && a.pad < b.pad)); }

bool clustercompare_inevent(cluster_struct a, cluster_struct b) { return (a.sector < b.sector || (a.sector == b.sector && a.patch < b.patch) || (a.sector == b.sector && a.patch == b.patch && a.row < b.row)); }

void do_diff(uint32_t& val, int32_t& last, uint32_t bits, uint32_t maxval = 0)
{
  int32_t tmp = val;
  val -= last;
  if (maxval && optimized_negative_values) {
    while ((signed)val < 0) {
      val += maxval;
    }
  } else {
    val &= (1 << bits) - 1;
  }
  last = tmp;
}

uint32_t truncate(int32_t j, uint32_t val)
{
  if (truncate_bits && field_bits[j] != significant_bits[j] && val) {
    int32_t ldz = sizeof(uint32_t) * 8 - __builtin_clz(val);
    if (ldz > significant_bits[j]) {
      val &= ((1 << ldz) - 1) ^ ((1 << (ldz - significant_bits[j])) - 1);
    }
  }
  return (val);
}

int32_t main(int argc, char** argv)
{
  FILE* fp;

  if (truncate_bits && (track_avgmax || track_diffqmax || track_diffqtot)) {
    printf("Cannot use truncate bits with differential qmax / qtot");
    return (1);
  }
  if (truncate_bits && (track_diffsigma)) {
    printf("Cannot use truncate bits with differential sigma");
    return (1);
  }

  if (!(fp = fopen(file, "rb"))) {
    printf("Error opening file\n");
    return (1);
  }

  fseek(fp, 0, SEEK_END);
  size_t nFileSize = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  size_t nClusters = nFileSize / sizeof(cluster_struct);
  if (max_clusters && nClusters > max_clusters) {
    nClusters = max_clusters;
  }

  cluster_struct* clusters = new cluster_struct[nClusters];
  if (clusters == NULL) {
    printf("Memory allocation error\n");
    return (1);
  }

  fprintf(stderr, "Reading %d clusters...", (int32_t)nClusters);
  fread(clusters, sizeof(cluster_struct), nClusters, fp);

  fprintf(stderr, "Done\nSorting clusters...");

  if (sort_method) {
    int32_t starti = 0;
    if (!track_based) {
      fprintf(stderr, " (removing track ordering)...");
      int32_t last_event = 0;
      for (int32_t i = 0; i <= nClusters; i++) {
        int32_t event = (i == nClusters ? -1 : clusters[i].event);
        if (last_event != event) {
          if (i - 1 > starti) {
            std::sort(clusters + starti, clusters + i - 1, clustercompare_inevent);
          }
          starti = i;
        }
        last_event = event;
      }
    }

    starti = 0;
    int32_t startrow = -1;
    for (int32_t i = 0; i <= nClusters; i++) {
      int32_t currow;
      if (i == nClusters) {
        currow = -1;
      } else if (track_based && clusters[i].trackID != -1) {
        currow = -2;
      } else {
        currow = clusters[i].row;
      }
      if (currow != startrow && startrow != -2) {
        if (i - 1 > starti) {
          if (sort_method == 1) {
            std::sort(clusters + starti, clusters + i - 1, clustercompare_padtime);
          } else if (sort_method == 2) {
            std::sort(clusters + starti, clusters + i - 1, clustercompare_timepad);
          } else if (sort_method == 3) {
            std::sort(clusters + starti, clusters + i - 1, clustercompare_padtime_mixed);
          } else if (sort_method == 4) {
            std::sort(clusters + starti, clusters + i - 1, clustercompare_timepad_mixed);
          }
        }
        starti = i;
        startrow = currow;
      }
    }
  }
  fprintf(stderr, "Done\n");

  fclose(fp);

  int64_t* histograms[nFields];
  double* probabilities[nFields];
  int64_t counts[nFields];
  int32_t used[nFields];
  for (int32_t i = SECTOR; i < nFields; i++) {
    if (i == CLUSTER_ID) {
      continue;
    }
    histograms[i] = new int64_t[1 << field_bits[i]];
    probabilities[i] = new double[1 << field_bits[i]];
  }

  double rawtotalbytes = 0;
  double entrototalbytes = 0;
  for (int32_t isector = 0; isector < 36; isector++) {
    for (int32_t ipatch = 0; ipatch < 6; ipatch++) {
      if (separate_sectors) {
        printf("SECTOR %d ", isector);
      }
      if (separate_patches) {
        printf("PATCH %d", ipatch);
      }
      if (separate_sectors || separate_patches) {
        printf("\n");
      }
      for (int32_t i = SECTOR; i < nFields; i++) {
        if (i == CLUSTER_ID || i == PATCH) {
          continue;
        }
        memset(histograms[i], 0, sizeof(int64_t) * (1 << field_bits[i]));
        counts[i] = 0;
        used[i] = 0;
      }

      size_t nClustersUsed = 0;

      int32_t lastRow = 0, lastPad = 0, lastTime = 0, lastSector = 0, lastResPad = 0, lastResTime = 0, lastQTot = 0, lastQMax = 0, lastSigmaPad = 0, lastSigmaTime = 0, lastTrack = -1, lastEvent = 0;

      for (size_t i = 0; i < nClusters; i++) {
        const cluster_struct& cluster_org = clusters[i];
        cluster_struct cluster = clusters[i];
        if (cluster.pad >= 32768) {
          printf("%d\n", cluster.pad);
        }

        if ((separate_sectors && cluster.sector != isector) || (separate_patches && cluster.patch != ipatch)) {
          continue;
        }
        if (separate_sides && !(cluster.sector < 18 ^ isector < 18)) {
          continue;
        }

        bool newTrack = lastTrack != cluster.trackID;
        uint32_t dSigmaPad, dSigmaTime;

        if (cluster.event != lastEvent) {
          lastRow = lastPad = lastTime = lastSector = 0;
          lastTrack = -1;
        }

        if (full_row_numbers) {
          cluster.row += fgRows[cluster.patch][0];
        }

        if ((sector_diff || res_diff || track_diffqtot || track_diffqmax) && cluster.trackID != -1 && track_based) {
          if (lastTrack != cluster.trackID) {
            lastSector = lastResPad = lastResTime = lastQTot = lastQMax = lastSigmaPad = lastSigmaTime = 0;
          }

          if (sector_diff) {
            do_diff(cluster.sector, lastSector, field_bits[SECTOR]);
          }

          if (res_diff) {
            do_diff(cluster.resPad, lastResPad, field_bits[RES_PAD]);
            do_diff(cluster.resTime, lastResTime, field_bits[RES_TIME]);
          }

          if (track_diffqtot) {
            cluster.avgtot = cluster.qtot;
            do_diff(cluster.avgtot, lastQTot, field_bits[QTOT]);
          }
          if (track_diffqmax) {
            cluster.avgmax = cluster.qmax;
            do_diff(cluster.avgmax, lastQMax, field_bits[QMAX]);
          }
          if (track_diffsigma) {
            dSigmaPad = cluster.sigmaPad;
            dSigmaTime = cluster.sigmaTime;
            do_diff(dSigmaPad, lastSigmaPad, field_bits[SIGMA_PAD]);
            do_diff(dSigmaTime, lastSigmaTime, field_bits[SIGMA_TIME]);
          } else if (track_separate_sigma) {
            dSigmaPad = truncate(SIGMA_PAD, cluster.sigmaPad);
            dSigmaTime = truncate(SIGMA_TIME, cluster.sigmaTime);
          }
        }

        if (cluster.row != lastRow) {
          lastPad = lastTime = 0;
        }
        if (row_diff) {
          do_diff(cluster.row, lastRow, field_bits[ROW]);
        } else {
          lastRow = cluster.row;
        }

        if (pad_diff && (cluster.trackID == -1 || !track_based)) {
          do_diff(cluster.pad, lastPad, field_bits[PAD], (distinguish_rows ? fgNPadsMod[cluster_org.row + fgRows[cluster.patch][0]] : 140) * 60);
        }
        if (time_diff && (cluster.trackID == -1 || !track_based)) {
          do_diff(cluster.time, lastTime, field_bits[TIME], 1024 * 25);
        }

        if (approximate_qtot && (!track_based || cluster.trackID == -1 || (track_avgtot == 0 && track_diffqtot == 0))) {
          cluster.qtot -= cluster.sigmaPad * cluster.qmax / 3;
          if (cluster.qtot < 0) {
            cluster.qtot = -truncate(QTOT, -cluster.qtot);
          } else {
            cluster.qtot = truncate(QTOT, cluster.qtot);
          }
          cluster.qtot &= (1 << field_bits[QTOT]) - 1;
        }

        if (track_avgtot && cluster.trackID != -1) {
          int32_t tmp = truncate(QTOT, cluster.qtot) - truncate(QTOT, cluster.avgtot);
          if (newTrack) {
            cluster.qtot = truncate(QTOT, cluster.avgtot);
          }
          cluster.avgtot = tmp & ((1 << field_bits[QTOT]) - 1);
        }
        if (track_avgmax && cluster.trackID != -1) {
          int32_t tmp = cluster.qmax - cluster.avgmax;
          if (newTrack) {
            cluster.qmax = cluster.avgmax;
          }
          cluster.avgmax = tmp & ((1 << field_bits[QMAX]) - 1);
        }

        // Copy qmax / qtot to combined track avg... slot, to use for combine_maxtot
        if ((((combine_maxtot && (track_avgtot || track_diffqtot)) || track_separate_q) && track_avgmax == 0 && track_diffqmax == 0) && cluster.trackID != -1) {
          cluster.avgmax = cluster.qmax;
        }
        if ((((combine_maxtot && (track_avgmax || track_diffqmax)) || track_separate_q) && track_avgtot == 0 && track_diffqtot == 0) && cluster.trackID != -1) {
          cluster.avgtot = cluster.qtot;
        }

        for (int32_t j = 0; j < sizeof(cluster_struct) / sizeof(uint32_t); j++) {
          if (approximate_qtot && (j == QTOT || j == AVG_TOT)) {
            continue;
          }
          if (track_avgtot && (j == QTOT || j == AVG_TOT)) {
            continue;
          }
          cluster.vals[j] = truncate(j, cluster.vals[j]);
        }

        lastEvent = cluster.event;
        lastTrack = cluster.trackID;

        if (print_clusters > 0 || (print_clusters < 0 && i < -print_clusters)) {
          printf("Event %u Track %d Sector %u Patch %u Row %u Pad %u Time %u sigmaPad %u sigmaTime %u qTot %u qMax %u Flag %u resPad %u resTime %u avgTot %u avgMax %u\n", cluster.event, cluster.trackID, cluster.sector, cluster.patch, cluster.row, cluster.pad, cluster.time, cluster.sigmaPad,
                 cluster.sigmaTime, cluster.qtot, cluster.qmax, cluster.splitPadTime, cluster.resPad, cluster.resTime, cluster.avgtot, cluster.avgmax);
        }

        for (int32_t j = SECTOR; j < nFields; j++) {
          bool forceStore = false;
          if (j == CLUSTER_ID || j == PATCH) {
            continue;
          }

          if (j == SECTOR && (track_based == 0 || cluster.trackID == -1)) {
            continue;
          }

          if (track_based && cluster.trackID != -1 && !newTrack) {
            if (j == PAD || j == TIME || (j >= PAD_80 && j <= PAD_140)) {
              continue;
            }
            if (j == RES_PAD || j == RES_TIME) {
              cluster.vals[j] &= (1 << field_bits[j]) - 1;
              forceStore = true;
            }
          }

          if ((track_avgtot || track_diffqtot || track_separate_q) && cluster.trackID != -1) {
            if (j == QTOT && (!newTrack || (track_avgtot == 0 && track_diffqtot == 0))) {
              continue;
            }
            if (j == AVG_TOT && (track_diffqtot == 0 || !newTrack)) {
              forceStore = true;
            }
          }
          if ((track_avgmax || track_diffqmax || track_separate_q) && cluster.trackID != -1) {
            if (j == QMAX && (!newTrack || (track_avgmax == 0 && track_diffqmax == 0))) {
              continue;
            }
            if (j == AVG_MAX && (track_diffqmax == 0 || !newTrack)) {
              forceStore = true;
            }
          }

          if ((track_diffsigma || track_separate_sigma) && cluster.trackID != -1) {
            if (j == SIGMA_PAD || j == SIGMA_TIME) {
              continue;
            }
            if (j == DIFF_SIGMA_PAD) {
              histograms[j][dSigmaPad]++;
              counts[j]++;
            }
            if (j == DIFF_SIGMA_TIME) {
              histograms[j][dSigmaTime]++;
              counts[j]++;
            }
          }

          if (track_based && row_diff && cluster.trackID != -1) {
            if (j == ROW) {
              continue;
            }
            int32_t myj = newTrack ? ROW_TRACK_FIRST : ROW_TRACK;
            if (j == myj) {
              histograms[myj][cluster.vals[ROW]]++;
              counts[myj]++;
            }
          }

          if (j <= FLAG_PADTIME || forceStore) {
            if (cluster.vals[j] >= (1 << field_bits[j])) {
              printf("Cluster value %d/%s out of bit range %d > %d\n", j, field_names[j], cluster.vals[j], (1 << field_bits[j]));
            } else {
              histograms[j][cluster.vals[j]]++;
              counts[j]++;
            }
          } else if (j == QMAX_QTOT && (!track_based || cluster.trackID == -1 || (((track_avgmax == 0 && track_avgtot == 0 && track_diffqmax == 0 && track_diffqtot == 0) || newTrack) && track_separate_q == 0))) {
            int32_t val = (cluster.qtot << field_bits[QMAX]) | cluster.qmax;
            histograms[j][val]++;
            counts[j]++;
          } else if (((track_avgmax || track_avgtot || track_diffqmax || track_diffqtot) && !newTrack || track_separate_q) && cluster.trackID != -1 && j == AVG_TOT_MAX) {
            int32_t val = (cluster.avgtot << field_bits[QMAX]) | cluster.avgmax;
            histograms[j][val]++;
            counts[j]++;
          } else if (j == SIGMA_PAD_TIME && (!track_based || cluster.trackID == -1 || (track_diffsigma == 0 && track_separate_sigma == 0))) {
            int32_t val = (cluster.sigmaTime << field_bits[SIGMA_PAD]) | cluster.sigmaPad;
            histograms[j][val]++;
            counts[j]++;
          } else if ((track_diffsigma || track_separate_sigma) && cluster.trackID != -1 && j == DIFF_SIGMA_PAD_TIME) {
            int32_t val = (dSigmaPad << field_bits[SIGMA_PAD]) | dSigmaTime;
            histograms[j][val]++;
            counts[j]++;
          } else if (distinguish_rows && j >= PAD_80 && j <= PAD_140) {
            int32_t myj = fgNPads[cluster_org.row + fgRows[cluster.patch][0]];
            myj = (myj - (80 - 11)) / 12;
            myj += PAD_80;
            if (myj == j) {
              if (cluster.pad >= (1 << field_bits[j])) {
                printf("Cluster value %d/%s out of bit range %d > %d\n", j, field_names[j], cluster.vals[j], (1 << field_bits[j]));
              } else {
                histograms[j][cluster.pad]++;
                counts[j]++;
              }
            }
          }
        }
        nClustersUsed++;
      }

      printf("Clusters in block: %ld / %ld\n", nClustersUsed, nClusters);

      double log2 = log(2.);
      double entropies[nFields];
      double huffmanSizes[nFields];
      for (int32_t i = SECTOR; i < nFields; i++) {
        if (i == CLUSTER_ID || i == PATCH) {
          continue;
        }
        double entropy = 0.;
        double huffmanSize = 0;

        if (counts[i]) {
          for (int32_t j = 0; j < (1 << field_bits[i]); j++) {
            // printf("Field %d/%s Value %d Entries %ld\n", i, field_names[i], j, histograms[i][j]);

            probabilities[i][j] = (double)histograms[i][j] / (double)counts[i];
            if (probabilities[i][j]) {
              double I = -log(probabilities[i][j]) / log2;
              double H = I * probabilities[i][j];
              // printf("Field %d/%s Value %d I prob %f I %f H %f\n", i, field_names[i], probabilities[i][j], I, H);

              entropy += H;
            }
          }

          INode* root = BuildTree(probabilities[i], 1 << field_bits[i]);

          HuffCodeMap codes;
          GenerateCodes(root, HuffCode(), codes);
          delete root;

          for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); it++) {
            huffmanSize += it->second.size() * probabilities[i][it->first];
          }
        }
        entropies[i] = entropy;
        huffmanSizes[i] = huffmanSize;
      }

      int32_t rawBits = 0;
      double entroTotal = 0., huffmanTotal = 0.;
      for (int32_t i = SECTOR; i < nFields; i++) {
        if (i == CLUSTER_ID || i == PATCH) {
          continue;
        }

        if (i <= FLAG_PADTIME) {
          rawBits += field_bits[i];
        }

        if (combine_maxtot && (i == QMAX || i == QTOT)) {
          continue;
        }
        if (combine_sigmapadtime && (i == SIGMA_PAD || i == SIGMA_TIME)) {
          continue;
        }
        if ((track_diffsigma || track_separate_sigma) && combine_sigmapadtime && (i == DIFF_SIGMA_PAD || i == DIFF_SIGMA_TIME)) {
          continue;
        }

        if (distinguish_rows && i == PAD) {
          continue;
        }

        if (i <= FLAG_PADTIME || (combine_maxtot && i == QMAX_QTOT) || (combine_maxtot && (track_avgmax || track_avgtot || track_diffqmax || track_diffqtot || track_separate_q) && combine_maxtot && i == AVG_TOT_MAX) || (combine_sigmapadtime && i == SIGMA_PAD_TIME) ||
            (combine_sigmapadtime && (track_diffsigma || track_separate_sigma) && i == DIFF_SIGMA_PAD_TIME) || (track_based && (i == RES_PAD || i == RES_TIME)) || ((track_avgtot || track_diffqtot || track_separate_q) && !combine_maxtot && i == AVG_TOT) ||
            ((track_avgmax || track_diffqmax || track_separate_q) && !combine_maxtot && i == AVG_MAX) || ((track_diffsigma || track_separate_sigma) && (i == DIFF_SIGMA_PAD || i == DIFF_SIGMA_TIME)) || (track_based && row_diff && (i == ROW_TRACK || i == ROW_TRACK_FIRST)) ||
            (distinguish_rows && i >= PAD_80 && i <= PAD_140)) {
          entroTotal += entropies[i] * counts[i];
          huffmanTotal += huffmanSizes[i] * counts[i];
          used[i] = 1;
        }
      }
      for (int32_t i = SECTOR; i < nFields; i++) {
        if (field_bits[i] == 0) {
          continue;
        }
        if (counts[i] == 0) {
          continue;
        }
        printf("Field %2d/%16s (count %10ld / used %1d) rawBits %2d huffman %9.6f entropy %9.6f\n", i, field_names[i], counts[i], used[i], field_bits[i], huffmanSizes[i], entropies[i]);
      }
      rawBits = 79; // Override incorrect calculation: Row is only 6 bit in raw format, and sector is not needed!
      printf("Raw Bits: %d - Total Size %f MB Clusters %d\n", rawBits, (double)rawBits * (double)nClustersUsed / 8. / 1.e6, nClustersUsed);
      printf("Huffman Bits: %f - Total Size %f MB\n", huffmanTotal / (double)nClustersUsed, huffmanTotal / 8. / 1.e6);
      printf("Entropy Bits: %f - Total Size %f MB\n", entroTotal / (double)nClustersUsed, entroTotal / 8. / 1.e6);
      printf("Maximum Compression Ratio: %f (Huffman %f)\n", (double)rawBits * (double)nClustersUsed / entroTotal, (double)rawBits * (double)nClustersUsed / huffmanTotal);
      entrototalbytes += entroTotal;
      rawtotalbytes += (double)rawBits * (double)nClustersUsed;

      if (separate_sides && !separate_sectors && isector == 0) {
        isector = 17;
      } else if (!separate_sectors) {
        isector = 9999999;
      }

      if (!separate_patches) {
        ipatch = 9999999;
      }
    }
  }

  if (separate_sectors || separate_patches || separate_sides) {
    printf("Total Compression: %f\n", rawtotalbytes / entrototalbytes);
  }

  printf("Exiting\n");
  for (int32_t i = SECTOR; i < nFields; i++) {
    if (i == CLUSTER_ID || i == PATCH) {
      continue;
    }
    delete[] histograms[i];
    delete[] probabilities[i];
  }
  delete[] clusters;
  return (0);
}
