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

/// @file   ThresholdCalibratorSpec.cxx

#include "ITSWorkflow/ThresholdCalibratorSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Define error function for ROOT fitting
double erf(double* xx, double* par)
{
  return (nInjScaled / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (nInjScaled / 2);
}

// ITHR erf is reversed
double erf_ithr(double* xx, double* par)
{
  return (nInjScaled / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])));
}

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSThresholdCalibrator::ITSThresholdCalibrator(const ITSCalibInpConf& inpConf)
  : mChipModSel(inpConf.chipModSel), mChipModBase(inpConf.chipModBase)
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSThresholdCalibrator");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSThresholdCalibrator::~ITSThresholdCalibrator()
{
  // Clear dynamic memory

  delete[] this->mX;
  this->mX = nullptr;

  if (this->mFitType == FIT) {
    delete this->mFitHist;
    this->mFitHist = nullptr;
    delete this->mFitFunction;
    this->mFitFunction = nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::init(InitContext& ic)
{
  LOGF(info, "ITSThresholdCalibrator init...", mSelfName);

  mPercentageCut = ic.options().get<short int>("percentage-cut");

  mColStep = ic.options().get<short int>("s-curve-col-step");
  if (mColStep >= N_COL) {
    LOG(warning) << "mColStep = " << mColStep << ": saving s-curves of only 1 pixel (pix 0) per row";
  }

  std::string fittype = ic.options().get<std::string>("fittype");
  if (fittype == "derivative") {
    this->mFitType = DERIVATIVE;

  } else if (fittype == "fit") {
    this->mFitType = FIT;

  } else if (fittype == "hitcounting") {
    this->mFitType = HITCOUNTING;

  } else {
    LOG(error) << "fittype " << fittype
               << " not recognized, please use 'derivative', 'fit', or 'hitcounting'";
    throw fittype;
  }

  // Get metafile directory from input
  try {
    this->mMetafileDir = ic.options().get<std::string>("meta-output-dir");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter meta-output-dir not found"
                 << "\n*** Setting metafile output directory to /dev/null";
  }
  if (this->mMetafileDir != "/dev/null") {
    this->mMetafileDir = o2::utils::Str::rectifyDirectory(this->mMetafileDir);
  }

  // Get ROOT output directory from input
  try {
    this->mOutputDir = ic.options().get<std::string>("output-dir");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter output-dir not found"
                 << "\n*** Setting ROOT output directory to ./";
  }
  this->mOutputDir = o2::utils::Str::rectifyDirectory(this->mOutputDir);

  // Get metadata data type from input
  try {
    this->mMetaType = ic.options().get<std::string>("meta-type");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter meta-type not found"
                 << "\n*** Disabling 'type' in metadata output files";
  }

  this->mVerboseOutput = ic.options().get<bool>("verbose");

  // Get number of threads
  this->mNThreads = ic.options().get<int>("nthreads");

  // Check fit type vs nthreads (fit option is not thread safe!)
  if (mFitType == FIT && mNThreads > 1) {
    throw std::runtime_error("Multiple threads are requested with fit method which is not thread safe");
  }

  // Machine hostname
  this->mHostname = boost::asio::ip::host_name();

  // check flag to tag single noisy pix in digital and analog scans
  this->mTagSinglePix = ic.options().get<bool>("enable-single-pix-tag");

  // get min and max ithr and vcasn (default if not specified)
  inMinVcasn = ic.options().get<short int>("min-vcasn");
  inMaxVcasn = ic.options().get<short int>("max-vcasn");
  inMinIthr = ic.options().get<short int>("min-ithr");
  inMaxIthr = ic.options().get<short int>("max-ithr");
  if (inMinVcasn > inMaxVcasn || inMinIthr > inMaxIthr) {
    throw std::runtime_error("Min VCASN/ITHR is larger than Max VCASN/ITHR: check the settings, analysis not possible");
  }

  // Get flag to enable most-probable value calculation
  isMpv = ic.options().get<bool>("enable-mpv");

  // Parameters to operate in manual mode (when run type is not recognized automatically)
  isManualMode = ic.options().get<bool>("manual-mode");
  if (isManualMode) {
    try {
      manualMin = ic.options().get<short int>("manual-min");
    } catch (std::exception const& e) {
      throw std::runtime_error("Min value of the scan parameter not found, mandatory in manual mode");
    }

    try {
      manualMax = ic.options().get<short int>("manual-max");
    } catch (std::exception const& e) {
      throw std::runtime_error("Max value of the scan parameter not found, mandatory in manual mode");
    }

    try {
      manualScanType = ic.options().get<std::string>("manual-scantype");
    } catch (std::exception const& e) {
      throw std::runtime_error("Scan type not found, mandatory in manual mode");
    }

    try {
      saveTree = ic.options().get<bool>("save-tree");
    } catch (std::exception const& e) {
      throw std::runtime_error("Please specify if you want to save the ROOT trees, mandatory in manual mode");
    }

    // this is not mandatory since it's 1 by default
    manualStep = ic.options().get<short int>("manual-step");

    // this is not mandatory since it's 0 by default
    manualMin2 = ic.options().get<short int>("manual-min2");

    // this is not mandatory since it's 0 by default
    manualMax2 = ic.options().get<short int>("manual-max2");

    // this is not mandatory since it's 1 by default
    manualStep2 = ic.options().get<short int>("manual-step2");

    // this is not mandatory since it's 5 by default
    manualStrobeWindow = ic.options().get<short int>("manual-strobewindow");

    // Flag to scale the number of injections by 3 in case --meb-select is used
    scaleNinj = ic.options().get<bool>("scale-ninj");
  }

  // Flag to enable the analysis of CRU_ITS data
  isCRUITS = ic.options().get<bool>("enable-cru-its");

  // Number of injections
  nInj = ic.options().get<int>("ninj");
  nInjScaled = nInj;

  // flag to set the url ccdb mgr
  this->mCcdbMgrUrl = ic.options().get<std::string>("ccdb-mgr-url");
  // FIXME: Temporary solution to retrieve ConfDBmap
  long int ts = o2::ccdb::getCurrentTimestamp();
  LOG(info) << "Getting confDB map from ccdb - timestamp: " << ts;
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCcdbMgrUrl);
  mgr.setTimestamp(ts);
  mConfDBmap = mgr.get<std::vector<int>>("ITS/Calib/Confdbmap");

  // Parameters to dump s-curves on disk
  isDumpS = ic.options().get<bool>("dump-scurves");
  maxDumpS = ic.options().get<int>("max-dump");
  chipDumpS = ic.options().get<std::string>("chip-dump"); // comma-separated list of chips
  chipDumpList = getIntegerVect(chipDumpS);
  if (isDumpS && mFitType != FIT) {
    LOG(error) << "S-curve dump enabled but `fittype` is not fit. Please check";
  }
  if (isDumpS) {
    fileDumpS = TFile::Open(Form("s-curves_%d.root", mChipModSel), "RECREATE"); // in case of multiple processes, every process will have it's own file
    if (maxDumpS < 0) {
      LOG(info) << "`max-dump` " << maxDumpS << ". Dumping all s-curves";
    } else {
      LOG(info) << "`max-dump` " << maxDumpS << ". Dumping " << maxDumpS << " s-curves";
    }
    if (!chipDumpList.size()) {
      LOG(info) << "Dumping s-curves for all chips";
    } else {
      LOG(info) << "Dumping s-curves for chips: " << chipDumpS;
    }
  }

  // flag to enable the calculation of the slope in 2d pulse shape scans
  doSlopeCalculation = ic.options().get<bool>("calculate-slope");
  if (doSlopeCalculation) {
    try {
      chargeA = ic.options().get<int>("charge-a");
    } catch (std::exception const& e) {
      throw std::runtime_error("You want to do the slop calculation but you did not specify charge-a");
    }

    try {
      chargeB = ic.options().get<int>("charge-b");
    } catch (std::exception const& e) {
      throw std::runtime_error("You want to do the slop calculation but you did not specify charge-b");
    }
  }

  // Variable to select from which multi-event buffer select the hits
  mMeb = ic.options().get<int>("meb-select");
  if (mMeb > 2) {
    LOG(error) << "MEB cannot be greater than 2. Please check your command line.";
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Get number of active links for a given RU
short int ITSThresholdCalibrator::getNumberOfActiveLinks(bool* links)
{
  int nL = 0;
  for (int i = 0; i < 3; i++) {
    if (links[i]) {
      nL++;
    }
  }
  return nL;
}

//////////////////////////////////////////////////////////////////////////////
// Get link ID: 0,1,2 for IB RUs / 0,1 for OB RUs
short int ITSThresholdCalibrator::getLinkID(short int chipID, short int ruID)
{
  if (chipID < 432) {
    return (chipID - ruID * 9) / 3;
  } else if (chipID >= 432 && chipID < 6480) {
    return (chipID - 48 * 9 - (ruID - 48) * 112) / 56;
  } else {
    return (chipID - 48 * 9 - 54 * 112 - (ruID - 102) * 196) / 98;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Get list of chipID (from 0 to 24119) attached to a RU based on the links which are active
std::vector<short int> ITSThresholdCalibrator::getChipListFromRu(short int ruID, bool* links)
{
  std::vector<short int> cList;
  int a, b;
  if (ruID < 48) {
    a = ruID * 9;
    b = a + 9 - 1;
  } else if (ruID >= 48 && ruID < 102) {
    a = 48 * 9 + (ruID - 48) * 112;
    b = a + 112 - 1;
  } else {
    a = 48 * 9 + 54 * 112 + (ruID - 102) * 196;
    b = a + 196 - 1;
  }

  for (int c = a; c <= b; c++) {
    short int lid = getLinkID(c, ruID);
    if (links[lid]) {
      cList.push_back(c);
    }
  }

  return cList;
}

//////////////////////////////////////////////////////////////////////////////
// Get RU ID (from 0 to 191) from a given O2ChipID (from 0 to 24119)
short int ITSThresholdCalibrator::getRUID(short int chipID)
{
  // below there are the inverse of the formulas in getChipListFromRu(...)
  if (chipID < 432) { // IB
    return chipID / 9;
  } else if (chipID >= 432 && chipID < 6480) { // ML
    return (chipID - 48 * 9 + 112 * 48) / 112;
  } else { // OL
    return (chipID - 48 * 9 - 54 * 112 + 102 * 196) / 196;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Convert comma-separated list of integers to a vector of int
std::vector<short int> ITSThresholdCalibrator::getIntegerVect(std::string& s)
{
  std::stringstream ss(s);
  std::vector<short int> result;
  char ch;
  short int tmp;
  while (ss >> tmp) {
    result.push_back(tmp);
    ss >> ch;
  }
  return result;
}

//////////////////////////////////////////////////////////////////////////////
// Open a new ROOT file and threshold TTree for that file
void ITSThresholdCalibrator::initThresholdTree(bool recreate /*=true*/)
{

  // Create output directory to store output
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
  o2::utils::createDirectoriesIfAbsent(dir);
  LOG(info) << "Created " << dir << " directory for ROOT trees output";

  std::string filename = dir + mDataTakingContext.runNumber + '_' +
                         std::to_string(this->mFileNumber) + '_' + this->mHostname + "_modSel" + std::to_string(mChipModSel) + ".root.part";

  // Check if file already exists
  struct stat buffer;
  if (recreate && stat(filename.c_str(), &buffer) == 0) {
    LOG(warning) << "File " << filename << " already exists, recreating";
  }

  // Initialize ROOT output file
  // to prevent premature external usage, use temporary name
  const char* option = recreate ? "RECREATE" : "UPDATE";
  mRootOutfile = new TFile(filename.c_str(), option);

  // Tree containing the s-curves points
  mScTree = new TTree("s-curve-points", "s-curve-points");
  mScTree->Branch("chipid", &vChipid, "vChipID[1024]/S");
  mScTree->Branch("row", &vRow, "vRow[1024]/S");

  // Initialize output TTree branches
  mThresholdTree = new TTree("ITS_calib_tree", "ITS_calib_tree");
  mThresholdTree->Branch("chipid", &vChipid, "vChipID[1024]/S");
  mThresholdTree->Branch("row", &vRow, "vRow[1024]/S");
  if (mScanType == 'T' || mScanType == 'V' || mScanType == 'I') {
    std::string bName = mScanType == 'T' ? "thr" : mScanType == 'V' ? "vcasn"
                                                                    : "ithr";
    mThresholdTree->Branch(bName.c_str(), &vThreshold, "vThreshold[1024]/S");
    mThresholdTree->Branch("noise", &vNoise, "vNoise[1024]/F");
    mThresholdTree->Branch("spoints", &vPoints, "vPoints[1024]/b");
    mThresholdTree->Branch("success", &vSuccess, "vSuccess[1024]/O");

    mScTree->Branch("chg", &vCharge, "vCharge[1024]/b");
    mScTree->Branch("hits", &vHits, "vHits[1024]/b");
  } else if (mScanType == 'D' || mScanType == 'A') { // this->mScanType == 'D' and this->mScanType == 'A'
    mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
  } else if (mScanType == 'P') {
    mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
    mThresholdTree->Branch("strobedel", &vMixData, "vMixData[1024]/S");
  } else if (mScanType == 'p' || mScanType == 't') {
    mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
    mThresholdTree->Branch("strobedel", &vMixData, "vMixData[1024]/S");
    mThresholdTree->Branch("charge", &vCharge, "vCharge[1024]/b");
    if (doSlopeCalculation) {
      mSlopeTree = new TTree("line_tree", "line_tree");
      mSlopeTree->Branch("chipid", &vChipid, "vChipID[1024]/S");
      mSlopeTree->Branch("row", &vRow, "vRow[1024]/S");
      mSlopeTree->Branch("slope", &vSlope, "vSlope[1024]/F");
      mSlopeTree->Branch("intercept", &vIntercept, "vIntercept[1024]/F");
    }
  } else if (mScanType == 'R') {
    mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
    mThresholdTree->Branch("vresetd", &vMixData, "vMixData[1024]/S");
  } else if (mScanType == 'r') {
    mThresholdTree->Branch("thr", &vThreshold, "vThreshold[1024]/S");
    mThresholdTree->Branch("noise", &vNoise, "vNoise[1024]/F");
    mThresholdTree->Branch("success", &vSuccess, "vSuccess[1024]/O");
    mThresholdTree->Branch("vresetd", &vMixData, "vMixData[1024]/S");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findUpperLower(
  std::vector<std::vector<unsigned short int>> data, const short int& NPoints,
  short int& lower, short int& upper, bool flip, int iloop2)
{
  // Initialize (or re-initialize) upper and lower
  upper = -1;
  lower = -1;

  if (flip) { // ITHR case. lower is at large mX[i], upper is at small mX[i]

    for (int i = 0; i < NPoints; i++) {
      int comp = mScanType != 'r' ? data[iloop2][i] : data[i][iloop2];
      if (comp == 0) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      int comp = mScanType != 'r' ? data[iloop2][i] : data[i][iloop2];
      if (comp >= nInjScaled) {
        lower = i;
        break;
      }
    }

  } else { // not flipped

    for (int i = 0; i < NPoints; i++) {
      int comp = mScanType != 'r' ? data[iloop2][i] : data[i][iloop2];
      if (comp >= nInjScaled) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      int comp = mScanType != 'r' ? data[iloop2][i] : data[i][iloop2];
      if (comp == 0) {
        lower = i;
        break;
      }
    }
  }

  // If search was successful, return central x value
  if ((lower == -1) || (upper < lower)) {
    return false;
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Main findThreshold function which calls one of the three methods
bool ITSThresholdCalibrator::findThreshold(
  const short int& chipID, std::vector<std::vector<unsigned short int>> data, const float* x, short int& NPoints,
  float& thresh, float& noise, int& spoints, int iloop2)
{
  bool success = false;

  switch (this->mFitType) {
    case DERIVATIVE: // Derivative method
      success = this->findThresholdDerivative(data, x, NPoints, thresh, noise, spoints, iloop2);
      break;

    case FIT: // Fit method
      success = this->findThresholdFit(chipID, data, x, NPoints, thresh, noise, spoints, iloop2);
      break;

    case HITCOUNTING: // Hit-counting method
      success = this->findThresholdHitcounting(data, x, NPoints, thresh, iloop2);
      // noise = 0;
      break;
  }

  return success;
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via S-curve fit
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// thresh, noise, chi2 pointers are updated with results from the fit
// spoints: number of points in the S of the S-curve (with n_hits between 0 and 50, excluding first and last point)
// iloop2 is 0 for thr scan but is equal to vresetd index in 2D vresetd scan
bool ITSThresholdCalibrator::findThresholdFit(
  const short int& chipID, std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints,
  float& thresh, float& noise, int& spoints, int iloop2)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  auto fndVal = std::find(chipDumpList.begin(), chipDumpList.end(), chipID);

  if (!this->findUpperLower(data, NPoints, lower, upper, flip, iloop2) || lower == upper) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: (lower, upper) = ("
                   << lower << ", " << upper << ")";
    }

    if (isDumpS && (dumpCounterS[chipID] < maxDumpS || maxDumpS < 0) && (fndVal != chipDumpList.end() || !chipDumpList.size())) { // save bad s-curves
      for (int i = 0; i < NPoints; i++) {
        this->mFitHist->SetBinContent(i + 1, mScanType != 'r' ? data[iloop2][i] : data[i][iloop2]);
      }
      fileDumpS->cd();
      mFitHist->Write();
    }
    if (isDumpS) {
      dumpCounterS[chipID]++;
    }

    return false;
  }
  float start = (this->mX[upper] + this->mX[lower]) / 2;

  if (start < 0) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: Start = " << start;
    }
    return false;
  }

  for (int i = 0; i < NPoints; i++) {
    this->mFitHist->SetBinContent(i + 1, mScanType != 'r' ? data[iloop2][i] : data[i][iloop2]);
  }

  // Initialize starting parameters
  this->mFitFunction->SetParameter(0, start);
  this->mFitFunction->SetParameter(1, 8);

  this->mFitHist->Fit("mFitFunction", "RQL");
  if (isDumpS && (dumpCounterS[chipID] < maxDumpS || maxDumpS < 0) && (fndVal != chipDumpList.end() || !chipDumpList.size())) { // save good s-curves
    fileDumpS->cd();
    mFitHist->Write();
  }
  if (isDumpS) {
    dumpCounterS[chipID]++;
  }

  noise = this->mFitFunction->GetParameter(1);
  thresh = this->mFitFunction->GetParameter(0);
  float chi2 = this->mFitFunction->GetChisquare() / this->mFitFunction->GetNDF();
  spoints = upper - lower - 1;

  // Clean up histogram for next time it is used
  this->mFitHist->Reset();

  return (chi2 < 5);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// spoints: number of points in the S of the S-curve (with n_hits between 0 and 50, excluding first and last point)
// iloop2 is 0 for thr scan but is equal to vresetd index in 2D vresetd scan
bool ITSThresholdCalibrator::findThresholdDerivative(std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints,
                                                     float& thresh, float& noise, int& spoints, int iloop2)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, NPoints, lower, upper, flip, iloop2) || lower == upper) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: (lower, upper) = (" << lower << ", " << upper << ")";
    }
    return false;
  }

  int deriv_size = upper - lower;
  float deriv[deriv_size];
  float xfx = 0, fx = 0;

  // Fill array with derivatives
  for (int i = lower; i < upper; i++) {
    deriv[i - lower] = std::abs(mScanType != 'r' ? (data[iloop2][i + 1] - data[iloop2][i]) : (data[i + 1][iloop2] - data[i][iloop2])) / (this->mX[i + 1] - mX[i]);
    xfx += this->mX[i] * deriv[i - lower];
    fx += deriv[i - lower];
  }

  if (fx > 0.) {
    thresh = xfx / fx;
  }
  float stddev = 0;
  for (int i = lower; i < upper; i++) {
    stddev += std::pow(this->mX[i] - thresh, 2) * deriv[i - lower];
  }

  stddev /= fx;
  noise = std::sqrt(stddev);
  spoints = upper - lower - 1;

  return fx > 0.;
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// iloop2 is 0 for thr scan but is equal to vresetd index in 2D vresetd scan
bool ITSThresholdCalibrator::findThresholdHitcounting(
  std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints, float& thresh, int iloop2)
{
  unsigned short int numberOfHits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    numberOfHits += (mScanType != 'r') ? data[iloop2][i] : data[i][iloop2];
    int comp = (mScanType != 'r') ? data[iloop2][i] : data[i][iloop2];
    if (!is50 && comp == nInjScaled) {
      is50 = true;
    }
  }

  // If not enough counts return a failure
  if (!is50) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Calculation unsuccessful: too few hits. Skipping this pixel";
    }
    return false;
  }

  if (this->mScanType == 'T') {
    thresh = this->mX[N_RANGE - 1] - numberOfHits / float(nInjScaled);
  } else if (this->mScanType == 'V') {
    thresh = (this->mX[N_RANGE - 1] * nInjScaled - numberOfHits) / float(nInjScaled);
  } else if (this->mScanType == 'I') {
    thresh = (numberOfHits + nInjScaled * this->mX[0]) / float(nInjScaled);
  } else {
    LOG(error) << "Unexpected runtype encountered in findThresholdHitcounting()";
    return false;
  }

  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Run threshold extraction on completed row and update memory
void ITSThresholdCalibrator::extractThresholdRow(const short int& chipID, const short int& row)
{
  if (this->mScanType == 'D' || this->mScanType == 'A') {
    // Loop over all columns (pixels) in the row
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {
      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][0][0];
      if (vThreshold[col_i] > nInj) {
        this->mNoisyPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] > 0 && vThreshold[col_i] < nInj) {
        this->mIneffPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] == 0) {
        this->mDeadPixID[chipID].push_back(col_i * 1000 + row);
      }
    }
  } else if (this->mScanType == 'P' || this->mScanType == 'p' || mScanType == 'R' || mScanType == 't') {
    // Loop over all columns (pixels) in the row
    for (short int var1_i = 0; var1_i < this->N_RANGE; var1_i++) {
      for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {
        for (short int col_i = 0; col_i < this->N_COL; col_i++) {
          vChipid[col_i] = chipID;
          vRow[col_i] = row;
          vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][chg_i][var1_i];
          vMixData[col_i] = (var1_i * this->mStep) + mMin;
          if (mScanType != 'R') {
            vMixData[col_i]++; // +1 because a delay of n correspond to a real delay of n+1 (from ALPIDE manual)
          }
          vCharge[col_i] = (unsigned char)(chg_i * this->mStep2 + mMin2);
        }
        this->saveThreshold();
      }
    }

    if (doSlopeCalculation) {
      int delA = -1, delB = -1;
      for (short int col_i = 0; col_i < this->N_COL; col_i++) {
        for (short int chg_i = 0; chg_i < 2; chg_i++) {
          bool isFound = false;
          int checkchg = !chg_i ? chargeA / mStep2 : chargeB / mStep2;
          for (short int sdel_i = N_RANGE - 1; sdel_i >= 0; sdel_i--) {
            if (mPixelHits[chipID][row][col_i][checkchg - 1][sdel_i] == nInj) {
              if (!chg_i) {
                delA = mMin + sdel_i * mStep + mStep / 2;
                isFound = true;
              } else {
                delB = mMin + sdel_i * mStep + mStep / 2;
                isFound = true;
              }
              break;
            }
          } // end loop on strobe delays

          if (!isFound) { // if not found, take the first point with hits starting from the left (i.e. in principle the closest point to the one with MAX n_hits)
            for (short int sdel_i = 0; sdel_i < N_RANGE; sdel_i++) {
              if (mPixelHits[chipID][row][col_i][checkchg - 1][sdel_i] > 0) {
                if (!chg_i) {
                  delA = mMin + sdel_i * mStep + mStep / 2;
                } else {
                  delB = mMin + sdel_i * mStep + mStep / 2;
                }
                break;
              }
            } // end loop on strobe delays
          }   // end if on isFound
        }     // end loop on the two charges

        if (delA > 0 && delB > 0 && delA != delB) {
          vSlope[col_i] = ((float)(chargeA - chargeB) / (float)(delA - delB));
          vIntercept[col_i] = (float)chargeA - (float)(vSlope[col_i] * delA);
          if (vSlope[col_i] < 0) { // protection for non expected slope
            vSlope[col_i] = 0.;
            vIntercept[col_i] = 0.;
          }
        } else {
          vSlope[col_i] = 0.;
          vIntercept[col_i] = 0.;
        }
      } // end loop on pix

      mSlopeTree->Fill();
    }

  } else { // threshold, vcasn, ithr, vresetd_2d

    short int iRU = getRUID(chipID);
#ifdef WITH_OPENMP
    omp_set_num_threads(mNThreads);
#pragma omp parallel for schedule(dynamic)
#endif
    // Loop over all columns (pixels) in the row
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {
      // Do the threshold fit
      float thresh = 0., noise = 0.;
      bool success = false;
      int spoints = 0;
      int scan_i = mScanType == 'r' ? (mLoopVal[iRU][row] - mMin) / mStep : 0;
      if (isDumpS) { // already protected for multi-thread in the init
        mFitHist->SetName(Form("scurve_chip%d_row%d_col%d_scani%d", chipID, row, col_i, scan_i));
      }

      success = this->findThreshold(chipID, mPixelHits[chipID][row][col_i],
                                    this->mX, mScanType == 'r' ? N_RANGE2 : N_RANGE, thresh, noise, spoints, scan_i);

      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = (mScanType == 'T' || mScanType == 'r') ? (short int)(thresh * 10.) : (short int)(thresh);
      vNoise[col_i] = (float)(noise * 10.); // always factor 10 also for ITHR/VCASN to not have all zeros
      vSuccess[col_i] = success;
      vPoints[col_i] = spoints > 0 ? (unsigned char)(spoints) : 0;

      if (mScanType == 'r') {
        vMixData[col_i] = mLoopVal[iRU][row];
      }
    }
    if (mScanType == 'r') {
      this->saveThreshold(); // save before moving to the next vresetd
    }

    // Fill the ScTree tree
    if (mScanType == 'T' || mScanType == 'V' || mScanType == 'I') { // TODO: store also for other scans?
      for (int ichg = mMin; ichg <= mMax; ichg += mStep) {
        for (short int col_i = 0; col_i < this->N_COL; col_i += mColStep) {
          vCharge[col_i] = ichg;
          vHits[col_i] = mPixelHits[chipID][row][col_i][0][(ichg - mMin) / mStep];
        }
        mScTree->Fill();
      }
    }
  } // end of the else

  // Saves threshold information to internal memory
  if (mScanType != 'P' && mScanType != 'p' && mScanType != 't' && mScanType != 'R' && mScanType != 'r') {
    this->saveThreshold();
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::saveThreshold()
{
  // write to TTree
  this->mThresholdTree->Fill();

  if (this->mScanType == 'V' || this->mScanType == 'I' || this->mScanType == 'T') {
    // Save info in a map for later averaging
    int sumT = 0, sumSqT = 0, sumN = 0, sumSqN = 0;
    int countSuccess = 0, countUnsuccess = 0;
    for (int i = 0; i < this->N_COL; i++) {
      if (vSuccess[i]) {
        sumT += vThreshold[i];
        sumN += (int)vNoise[i];
        sumSqT += (vThreshold[i]) * (vThreshold[i]);
        sumSqN += ((int)vNoise[i]) * ((int)vNoise[i]);
        countSuccess++;
        if (vThreshold[i] >= mMin && vThreshold[i] <= mMax && (mScanType == 'I' || mScanType == 'V')) {
          mpvCounter[vChipid[0]][vThreshold[i] - mMin]++;
        }
      } else {
        countUnsuccess++;
      }
    }
    short int chipID = vChipid[0];
    std::array<long int, 6> dataSum{{sumT, sumSqT, sumN, sumSqN, countSuccess, countUnsuccess}};
    if (!(this->mThresholds.count(chipID))) {
      this->mThresholds[chipID] = dataSum;
    } else {
      std::array<long int, 6> dataAll{{this->mThresholds[chipID][0] + dataSum[0], this->mThresholds[chipID][1] + dataSum[1], this->mThresholds[chipID][2] + dataSum[2], this->mThresholds[chipID][3] + dataSum[3], this->mThresholds[chipID][4] + dataSum[4], this->mThresholds[chipID][5] + dataSum[5]}};
      this->mThresholds[chipID] = dataAll;
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Perform final operations on output objects. In the case of a full threshold
// scan, rename ROOT file and create metadata file for writing to EOS
void ITSThresholdCalibrator::finalizeOutput()
{
  // Check that objects actually exist in memory
  if (!(mScTree) || !(this->mRootOutfile) || !(this->mThresholdTree) || (doSlopeCalculation && !(this->mSlopeTree))) {
    return;
  }

  // Ensure that everything has been written to the ROOT file
  this->mRootOutfile->cd();
  this->mThresholdTree->Write(nullptr, TObject::kOverwrite);
  this->mScTree->Write(nullptr, TObject::kOverwrite);

  if (doSlopeCalculation) {
    this->mSlopeTree->Write(nullptr, TObject::kOverwrite);
  }

  // Clean up the mThresholdTree, mScTree and ROOT output file
  delete this->mThresholdTree;
  this->mThresholdTree = nullptr;
  delete mScTree;
  mScTree = nullptr;
  if (doSlopeCalculation) {
    delete this->mSlopeTree;
    this->mSlopeTree = nullptr;
  }

  this->mRootOutfile->Close();
  delete this->mRootOutfile;
  this->mRootOutfile = nullptr;

  // Check that expected output directory exists
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
  if (!std::filesystem::exists(dir)) {
    LOG(error) << "Cannot find expected output directory " << dir;
    return;
  }

  // Expected ROOT output filename
  std::string filename = mDataTakingContext.runNumber + '_' +
                         std::to_string(this->mFileNumber) + '_' + this->mHostname + "_modSel" + std::to_string(mChipModSel);
  std::string filenameFull = dir + filename;
  try {
    std::rename((filenameFull + ".root.part").c_str(),
                (filenameFull + ".root").c_str());
  } catch (std::exception const& e) {
    LOG(error) << "Failed to rename ROOT file " << filenameFull
               << ".root.part, reason: " << e.what();
  }

  // Create metadata file
  o2::dataformats::FileMetaData* mdFile = new o2::dataformats::FileMetaData();
  mdFile->fillFileData(filenameFull + ".root");
  mdFile->setDataTakingContext(mDataTakingContext);
  if (!(this->mMetaType.empty())) {
    mdFile->type = this->mMetaType;
  }
  mdFile->priority = "high";
  mdFile->lurl = filenameFull + ".root";
  auto metaFileNameTmp = fmt::format("{}{}.tmp", this->mMetafileDir, filename);
  auto metaFileName = fmt::format("{}{}.done", this->mMetafileDir, filename);
  try {
    std::ofstream metaFileOut(metaFileNameTmp);
    metaFileOut << mdFile->asString() << '\n';
    metaFileOut.close();
    std::filesystem::rename(metaFileNameTmp, metaFileName);
  } catch (std::exception const& e) {
    LOG(error) << "Failed to create threshold metadata file "
               << metaFileName << ", reason: " << e.what();
  }
  delete mdFile;

  // Next time a file is created, use a larger number
  this->mFileNumber++;

  return;

} // finalizeOutput

//////////////////////////////////////////////////////////////////////////////
// Set the run_type for this run
// Initialize the memory needed for this specific type of run
void ITSThresholdCalibrator::setRunType(const short int& runtype)
{

  // Save run type info for future evaluation
  this->mRunType = runtype;

  if (runtype == THR_SCAN) {
    // full_threshold-scan -- just extract thresholds for each pixel and write to TTree
    // 512 rows per chip
    this->mScanType = 'T';
    this->initThresholdTree();
    this->mMin = 0;
    this->mMax = 50;
    this->N_RANGE = 51;
    this->mCheckExactRow = true;

  } else if (runtype == THR_SCAN_SHORT || runtype == THR_SCAN_SHORT_100HZ ||
             runtype == THR_SCAN_SHORT_200HZ || runtype == THR_SCAN_SHORT_33 || runtype == THR_SCAN_SHORT_2_10HZ || runtype == THR_SCAN_SHORT_150INJ) {
    // threshold_scan_short -- just extract thresholds for each pixel and write to TTree
    // 10 rows per chip
    this->mScanType = 'T';
    this->initThresholdTree();
    this->mMin = 0;
    this->mMax = 50;
    this->N_RANGE = 51;
    this->mCheckExactRow = true;
    if (runtype == THR_SCAN_SHORT_150INJ) {
      nInj = 150;
      if (mMeb >= 0) {
        nInjScaled = nInj / 3;
      }
    }
  } else if (runtype == VCASN150 || runtype == VCASN100 || runtype == VCASN100_100HZ || runtype == VCASN130 || runtype == VCASNBB) {
    // VCASN tuning for different target thresholds
    // Store average VCASN for each chip into CCDB
    // ATTENTION: with back bias (VCASNBB) put max vcasn to 130 (default is 80)
    // 4 rows per chip
    this->mScanType = 'V';
    this->initThresholdTree();
    this->mMin = inMinVcasn; // 30 is the default
    this->mMax = inMaxVcasn; // 80 is the default
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = true;

  } else if (runtype == ITHR150 || runtype == ITHR100 || runtype == ITHR100_100HZ || runtype == ITHR130) {
    // ITHR tuning  -- average ITHR per chip
    // S-curve is backwards from VCASN case, otherwise same
    // 4 rows per chip
    this->mScanType = 'I';
    this->initThresholdTree();
    this->mMin = inMinIthr; // 25 is the default
    this->mMax = inMaxIthr; // 100 is the default
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = true;

  } else if (runtype == DIGITAL_SCAN || runtype == DIGITAL_SCAN_100HZ || runtype == DIGITAL_SCAN_NOMASK) {
    // Digital scan -- only storing one value per chip, no fit needed
    this->mScanType = 'D';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 0;
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = false;

  } else if (runtype == ANALOGUE_SCAN) {
    // Analogue scan -- only storing one value per chip, no fit needed
    this->mScanType = 'A';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 0;
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = false;

  } else if (runtype == PULSELENGTH_SCAN) {
    // Pulse length scan
    this->mScanType = 'P';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 400; // strobe delay goes from 0 to 400 (included) in steps of 4
    this->mStep = 1;
    this->mStrobeWindow = 1; // it's 0 but it corresponds to 0+1 (as from alpide manual)
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mCheckExactRow = true;
  } else if (runtype == TOT_CALIBRATION_1_ROW) {
    // Pulse length scan 2D (charge vs strobe delay)
    this->mScanType = 'p'; // small p, just to distinguish from capital P
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 2000; // strobe delay goes from 0 to 2000 in steps of 10
    this->mStep = 10;
    this->mStrobeWindow = 10; // it's 9 but it corresponds to 9+1 (as from alpide manual)
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mMin2 = 0;   // charge min
    this->mMax2 = 170; // charge max
    this->mStep2 = 1;  // step for the charge
    this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
    this->mCheckExactRow = true;
  } else if (runtype == TOT_CALIBRATION) {
    // TOT calibration (like pulse shape 2D but with a reduced range in both strobe delay and charge)
    this->mScanType = 't';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 300;
    this->mMax = 1100; // strobe delay goes from 300 to 1100 (included) in steps of 10
    this->mStep = 10;
    this->mStrobeWindow = 10; // it's 9 but it corresponds to 9+1 (as from alpide manual)
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mMin2 = 30;                 // charge min
    this->mMax2 = 60;                 // charge max
    this->mStep2 = 30;                // step for the charge
    this->mCalculate2DParams = false; // do not calculate time over threshold, pulse length, etc..
    this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
    this->mCheckExactRow = true;

  } else if (runtype == VRESETD_150 || runtype == VRESETD_300 || runtype == VRESETD_2D) {
    this->mScanType = 'R'; // capital R is for 1D scan
    if (runtype == VRESETD_150 || runtype == VRESETD_300) {
      this->mFitType = NO_FIT;
    }
    this->mMin = 100;
    this->mMax = 240; // vresetd goes from 100 to 240 in steps of 5
    this->mStep = 5;
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    if (runtype == VRESETD_2D) {
      this->mScanType = 'r'; // small r, just to distinguish from capital R
      this->mMin2 = 0;       // charge min
      this->mMax2 = 50;      // charge max
      this->mStep2 = 1;      // step for the charge
      this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
    }
    this->mCheckExactRow = true;
    this->initThresholdTree();
  } else {
    // No other run type recognized by this workflow
    LOG(error) << "Runtype " << runtype << " not recognized by calibration workflow (ignore if you are in manual mode)";
    if (isManualMode) {
      LOG(info) << "Entering manual mode: be sure to have set all parameters correctly";
      this->mScanType = manualScanType[0];
      this->mMin = manualMin;
      this->mMax = manualMax;
      this->mMin2 = manualMin2;
      this->mMax2 = manualMax2;
      this->mStep = manualStep;                 // 1 by default
      this->mStep2 = manualStep2;               // 1 by default
      this->mStrobeWindow = manualStrobeWindow; // 5 = 125 ns by default
      this->N_RANGE = (mMax - mMin) / mStep + 1;
      this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
      if (saveTree) {
        this->initThresholdTree();
      }
      this->mFitType = (mScanType == 'D' || mScanType == 'A' || mScanType == 'P' || mScanType == 'p' || mScanType == 't') ? NO_FIT : mFitType;
      this->mCheckExactRow = (mScanType == 'D' || mScanType == 'A') ? false : true;
      if (scaleNinj) {
        nInjScaled = nInj / 3;
      }
    } else {
      throw runtype;
    }
  }

  this->mX = new float[mScanType == 'r' ? N_RANGE2 : N_RANGE];
  for (short int i = ((mScanType == 'r') ? mMin2 : mMin); i <= ((mScanType == 'r') ? mMax2 / mStep2 : mMax / mStep); i++) {
    this->mX[i - (mScanType == 'r' ? mMin2 : mMin)] = (float)i + 0.5;
  }

  // Initialize objects for doing the threshold fits
  if (this->mFitType == FIT) {
    // Initialize the histogram used for error function fits
    // Will initialize the TF1 in setRunType (it is different for different runs)
    this->mFitHist = new TH1F(
      "mFitHist", "mFitHist", mScanType == 'r' ? N_RANGE2 : N_RANGE, mX[0] - 1., mX[(mScanType == 'r' ? N_RANGE2 : N_RANGE) - 1]);

    // Initialize correct fit function for the scan type
    this->mFitFunction = (this->mScanType == 'I')
                           ? new TF1("mFitFunction", erf_ithr, mMin, mMax, 2)
                           : new TF1("mFitFunction", erf, (mScanType == 'T' || mScanType == 'r') ? 3 : mMin, mScanType == 'r' ? mMax2 : mMax, 2);
    this->mFitFunction->SetParName(0, "Threshold");
    this->mFitFunction->SetParName(1, "Noise");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Calculate pulse parameters in 1D scan: time over threshold, rise time, ...
std::vector<float> ITSThresholdCalibrator::calculatePulseParams(const short int& chipID)
{

  int rt_mindel = -1, rt_maxdel = -1, tot_mindel = -1, tot_maxdel = -1;
  int sumRt = 0, sumSqRt = 0, countRt = 0, sumTot = 0, sumSqTot = 0, countTot = 0;

  for (auto itrow = mPixelHits[chipID].begin(); itrow != mPixelHits[chipID].end(); itrow++) { // loop over the chip rows
    short int row = itrow->first;
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {                                                                              // loop over the pixels on the row
      for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {                                                                       // loop over the strobe delays
        if (mPixelHits[chipID][row][col_i][0][sdel_i] > 0.1 * nInj && mPixelHits[chipID][row][col_i][0][sdel_i] < nInj && rt_mindel < 0) { // from left, first bin with 10% hits and 90% hits
          rt_mindel = (sdel_i * mStep) + 1;                                                                                                // + 1 because if delay = n, we get n+1 in reality (ALPIDE feature)
        }
        if (mPixelHits[chipID][row][col_i][0][sdel_i] >= 0.9 * nInj) { // for Rt max take the 90% point
          rt_maxdel = (sdel_i * mStep) + 1;
          break;
        }
      }
      for (short int sdel_i = 0; sdel_i < N_RANGE; sdel_i++) {
        if (mPixelHits[chipID][row][col_i][0][sdel_i] >= 0.5 * nInj) { // for ToT take the 50% point
          tot_mindel = (sdel_i * mStep) + 1;
          break;
        }
      }

      for (short int sdel_i = N_RANGE - 1; sdel_i >= 0; sdel_i--) { // from right, the first bin with 50% nInj hits
        if (mPixelHits[chipID][row][col_i][0][sdel_i] >= 0.5 * nInj) {
          tot_maxdel = (sdel_i * mStep) + 1;
          break;
        }
      }

      if (tot_maxdel > tot_mindel && tot_mindel >= 0 && tot_maxdel >= 0) {
        sumTot += tot_maxdel - tot_mindel - mStrobeWindow;
        sumSqTot += (tot_maxdel - tot_mindel - mStrobeWindow) * (tot_maxdel - tot_mindel - mStrobeWindow);
        countTot++;
      }

      if (rt_maxdel > rt_mindel && rt_maxdel > 0 && rt_mindel > 0) {
        sumRt += rt_maxdel - rt_mindel + mStrobeWindow;
        sumSqRt += (rt_maxdel - rt_mindel + mStrobeWindow) * (rt_maxdel - rt_mindel + mStrobeWindow);
        countRt++;
      }

      rt_mindel = -1;
      rt_maxdel = -1;
      tot_maxdel = -1;
      tot_mindel = -1;
    } // end loop over col_i
  }   // end loop over chip rows

  std::vector<float> output; // {avgRt, rmsRt, avgTot, rmsTot}
  // Avg Rt
  output.push_back(!countRt ? 0. : (float)sumRt / (float)countRt);
  // Rms Rt
  output.push_back(!countRt ? 0. : (std::sqrt((float)sumSqRt / (float)countRt - output[0] * output[0])) * 25.);
  output[0] *= 25.;
  // Avg ToT
  output.push_back(!countTot ? 0. : (float)sumTot / (float)countTot);
  // Rms ToT
  output.push_back(!countTot ? 0. : (std::sqrt((float)sumSqTot / (float)countTot - output[2] * output[2])) * 25.);
  output[2] *= 25.;

  return output;
}

//////////////////////////////////////////////////////////////////////////////
// Calculate pulse parameters in 2D scan
std::vector<float> ITSThresholdCalibrator::calculatePulseParams2D(const short int& chipID)
{
  long int sumTot = 0, sumSqTot = 0, countTot = 0;
  long int sumMinThr = 0, sumSqMinThr = 0, countMinThr = 0;
  long int sumMinThrDel = 0, sumSqMinThrDel = 0;
  long int sumMaxPl = 0, sumSqMaxPl = 0, countMaxPl = 0;
  long int sumMaxPlChg = 0, sumSqMaxPlChg = 0;

  for (auto itrow = mPixelHits[chipID].begin(); itrow != mPixelHits[chipID].end(); itrow++) { // loop over the chip rows
    short int row = itrow->first;
    for (short int col_i = 0; col_i < this->N_COL; col_i++) { // loop over the pixels on the row
      int minThr = 1e7, minThrDel = 1e7, maxPl = -1, maxPlChg = -1;
      int tot_mindel = 1e7;
      bool isFound = false;
      for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {     // loop over charges
        for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) { // loop over the strobe delays
          if (mPixelHits[chipID][row][col_i][chg_i][sdel_i] == nInj) { // minimum threshold charge and delay
            minThr = chg_i * mStep2;
            minThrDel = (sdel_i * mStep) + 1; // +1 because n->n+1 (as from alpide manual)
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }
      isFound = false;
      for (short int sdel_i = this->N_RANGE - 1; sdel_i >= 0; sdel_i--) { // loop over the strobe delays
        for (short int chg_i = this->N_RANGE2 - 1; chg_i >= 0; chg_i--) { // loop over charges
          if (mPixelHits[chipID][row][col_i][chg_i][sdel_i] == nInj) {    // max pulse length charge and delay
            maxPl = (sdel_i * mStep) + 1;
            maxPlChg = chg_i * mStep2;
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }
      isFound = false;
      for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {   // loop over the strobe delays
        for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {   // loop over charges
          if (mPixelHits[chipID][row][col_i][chg_i][sdel_i] == nInj) { // min delay for the ToT calculation
            tot_mindel = (sdel_i * mStep) + 1;
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }

      if (maxPl > tot_mindel && tot_mindel < 1e7 && maxPl >= 0) { // ToT
        sumTot += maxPl - tot_mindel - mStrobeWindow;
        sumSqTot += (maxPl - tot_mindel - mStrobeWindow) * (maxPl - tot_mindel - mStrobeWindow);
        countTot++;
      }

      if (minThr < 1e7) { // minimum threshold
        sumMinThr += minThr;
        sumSqMinThr += minThr * minThr;
        sumMinThrDel += minThrDel;
        sumSqMinThrDel += minThrDel * minThrDel;
        countMinThr++;
      }

      if (maxPl >= 0) { // pulse length
        sumMaxPl += maxPl;
        sumSqMaxPl += maxPl * maxPl;
        sumMaxPlChg += maxPlChg;
        sumSqMaxPlChg += maxPlChg * maxPlChg;
        countMaxPl++;
      }
    } // end loop over col_i
  }   // end loop over chip rows

  // Pulse shape 2D output: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  std::vector<long int> values = {sumTot, sumSqTot, countTot, sumMinThr, sumSqMinThr, countMinThr, sumMinThrDel, sumSqMinThrDel, countMinThr, sumMaxPl, sumSqMaxPl, countMaxPl, sumMaxPlChg, sumSqMaxPlChg, countMaxPl};
  std::vector<float> output;

  for (int i = 0; i < values.size(); i += 3) {
    // Avg
    output.push_back(!values[i + 2] ? 0. : (float)values[i] / (float)values[i + 2]);
    // Rms
    output.push_back(!values[i + 2] ? 0. : std::sqrt((float)values[i + 1] / (float)values[i + 2] - output[output.size() - 1] * output[output.size() - 1]));
    if (i == 0 || i == 6 || i == 9) {
      output[output.size() - 1] *= 25.;
      output[output.size() - 2] *= 25.;
    }
  }

  return output;
}
//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSThresholdCalibrator::extractAndUpdate(const short int& chipID, const short int& row)
{
  // In threshold scan case, reset mThresholdTree before writing to a new file
  if ((this->mRowCounter)++ == N_ROWS_PER_FILE) {
    // Finalize output and create a new TTree and ROOT file
    this->finalizeOutput();
    this->initThresholdTree();
    // Reset data counter for the next output file
    this->mRowCounter = 1;
  }

  // Extract threshold values and save to memory
  this->extractThresholdRow(chipID, row);

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
void ITSThresholdCalibrator::run(ProcessingContext& pc)
{
  if (mRunStopRequested) { // give up when run stop request arrived
    return;
  }

  updateTimeDependentParams(pc);

  // Calibration vector
  const auto calibs = pc.inputs().get<gsl::span<o2::itsmft::GBTCalibData>>("calib");
  const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");

  // Store some lengths for convenient looping
  const unsigned int nROF = (unsigned int)ROFs.size();

  // Loop over readout frames (usually only 1, sometimes 2)
  for (unsigned int iROF = 0; iROF < nROF; iROF++) {

    unsigned int rofIndex = ROFs[iROF].getFirstEntry();
    unsigned int rofNEntries = ROFs[iROF].getNEntries();

    // Find the correct charge, row, cw counter values for this ROF
    short int loopval = -1, realcharge = 0;
    short int row = -1;
    short int cwcnt = -1;
    bool isAllZero = true;
    short int ruIndex = -1;
    for (short int iRU = 0; iRU < this->N_RU; iRU++) {
      const auto& calib = calibs[iROF * this->N_RU + iRU];
      if (calib.calibUserField != 0) {
        mRuSet.insert(iRU);
        ruIndex = iRU;
        isAllZero = false;

        if (loopval >= 0) {
          LOG(warning) << "More than one charge detected!";
        }

        if (this->mRunType == -1) {
          mCdwVersion = isCRUITS ? 0 : ((short int)(calib.calibUserField >> 45)) & 0x7;
          LOG(info) << "CDW version: " << mCdwVersion;
          short int runtype = isCRUITS ? -2 : !mCdwVersion ? ((short int)(calib.calibUserField >> 24)) & 0xff
                                                           : ((short int)(calib.calibUserField >> 9)) & 0x7f;
          mConfDBv = !mCdwVersion ? ((short int)(calib.calibUserField >> 32)) & 0xffff : ((short int)(calib.calibUserField >> 32)) & 0x1fff; // confDB version
          this->setRunType(runtype);
          LOG(info) << "Calibrator will ship these run parameters to aggregator:";
          LOG(info) << "Run type  : " << mRunType;
          LOG(info) << "Scan type : " << mScanType;
          LOG(info) << "Fit type  : " << std::to_string(mFitType);
          LOG(info) << "DB version (ignore in TOT_CALIB & VRESET2D): " << mConfDBv;
        }
        this->mRunTypeUp = isCRUITS ? -1 : !mCdwVersion ? ((short int)(calib.calibUserField >> 24)) & 0xff
                                                        : ((short int)(calib.calibUserField >> 9)) & 0x7f;

        // count the zeros
        if (!mRunTypeUp) {
          mRunTypeRU[iRU]++;
          mRunTypeRUCopy[iRU]++;
        }
        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (this->mScanType == 'T') {
          // For threshold scan have to subtract from 170 to get charge value
          loopval = isCRUITS ? (short int)((calib.calibUserField >> 16) & 0xff) : !mCdwVersion ? (short int)(170 - (calib.calibUserField >> 16) & 0xff)
                                                                                               : (short int)(170 - (calib.calibUserField >> 16) & 0xffff);
        } else if (this->mScanType == 'D' || this->mScanType == 'A') { // Digital scan
          loopval = 0;
        } else { // VCASN / ITHR tuning and Pulse length scan (it's the strobe delay in this case), and vresetd scan
          loopval = !mCdwVersion ? (short int)((calib.calibUserField >> 16) & 0xff) : (short int)((calib.calibUserField >> 16) & 0xffff);
        }

        if (this->mScanType == 'p' || this->mScanType == 't' || this->mScanType == 'r') {
          realcharge = 170 - ((short int)(calib.calibUserField >> 32)) & 0x1fff; // not existing with CDW v0
        }

        // Last 16 bits should be the row (only uses up to 9 bits)
        row = !mCdwVersion ? (short int)(calib.calibUserField & 0xffff) : (short int)(calib.calibUserField & 0x1ff);
        // cw counter
        cwcnt = (short int)(calib.calibCounter);
        // count the last N injections
        short int checkVal = (mScanType == 'I') ? mMin : mMax;
        if ((mScanType != 'r' && mScanType != 'p' && mScanType != 't' && loopval == checkVal) ||
            (mScanType == 'r' && realcharge == mMax2) ||
            (mScanType == 'p' && realcharge == mMin2) ||
            (mScanType == 't' && loopval == checkVal && realcharge == mMax2)) {
          mCdwCntRU[iRU][row]++;
          mLoopVal[iRU][row] = loopval; // keep loop val (relevant for VRESET2D and TOT_1ROW scan only)
        }
        if (this->mVerboseOutput) {
          LOG(info) << "RU: " << iRU << " CDWcounter: " << cwcnt << " row: " << row << " Loopval: " << loopval << " realcharge: " << realcharge << " confDBv: " << mCdwVersion;
          LOG(info) << "NDIGITS: " << digits.size();
        }

        break;
      }
    }

    if (isCRUITS && isAllZero) {
      if (mRunType == -1) {
        short int runtype = -2;
        mConfDBv = 0;
        this->setRunType(runtype);
        LOG(info) << "Running with CRU_ITS data - Calibrator will ship these run parameters to aggregator:";
        LOG(info) << "Run type (non-sense) : " << mRunType;
        LOG(info) << "Scan type : " << mScanType;
        LOG(info) << "Fit type  : " << std::to_string(mFitType);
        LOG(info) << "DB version (non-sense): " << mConfDBv;
      }
      loopval = 0;
      realcharge = 0;
      row = 0;
      cwcnt = 0;
    }

    if (loopval > this->mMax || loopval < this->mMin || ((mScanType == 'p' || mScanType == 't' || mScanType == 'r') && (realcharge > this->mMax2 || realcharge < this->mMin2))) {
      if (this->mVerboseOutput) {
        LOG(warning) << "CW issues - loopval value " << loopval << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << N_RANGE << ")";
        if (mScanType == 'p' || mScanType == 'r' || mScanType == 't') {
          LOG(warning) << " and/or realcharge value " << realcharge << " out of range from min " << this->mMin2
                       << " and max " << this->mMax2 << " (range: " << N_RANGE2 << ")";
        }
      }
    } else {
      std::vector<short int> mChips;
      // loop to retrieve list of chips and start tagging bad dcols if the hits does not come from this row
      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) { // gets chipid
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }
        if (d.getRow() != row && mVerboseOutput) {
          LOG(info) << "iROF: " << iROF << " ChipID " << chipID << ": current row is " << d.getRow() << " (col = " << d.getColumn() << ") but the one in CW is " << row;
        }
        if (std::find(mChips.begin(), mChips.end(), chipID) == mChips.end()) {
          mChips.push_back(chipID);
        }
      }
      // loop to allocate memory only for allowed rows
      for (auto& chipID : mChips) {
        // mark active RU links
        short int ru = getRUID(chipID);
        mActiveLinks[ru][getLinkID(chipID, ru)] = true;
        // check rows and allocate memory
        if (!this->mPixelHits.count(chipID)) {
          if (mScanType == 'D' || mScanType == 'A') { // for digital and analog scan initialize the full matrix for each chipID
            for (int irow = 0; irow < 512; irow++) {
              this->mPixelHits[chipID][irow] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE2, std::vector<unsigned short int>(N_RANGE, 0)));
            }
          } else {
            this->mPixelHits[chipID][row] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE2, std::vector<unsigned short int>(N_RANGE, 0)));
          }
        } else if (!this->mPixelHits[chipID].count(row)) { // allocate memory for chip = chipID or for a row of this chipID
          this->mPixelHits[chipID][row] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE2, std::vector<unsigned short int>(N_RANGE, 0)));
        }
      }

      // loop to count hits from digits
      short int loopPoint = (loopval - this->mMin) / mStep;
      short int chgPoint = (realcharge - this->mMin2) / mStep2;
      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }

        if ((!mCheckExactRow || d.getRow() == row) && (mMeb < 0 || cwcnt % 3 == mMeb)) { // row has NOT to be forbidden and we ignore hits coming from other rows (potential masking issue on chip)
          // Increment the number of counts for this pixel
          this->mPixelHits[chipID][d.getRow()][col][chgPoint][loopPoint]++;
        }
      }
    } // if (charge)

    ////
    // Prepare the ChipDone object for QC + extract data if the row is completed
    if (ruIndex < 0) {
      continue;
    }
    short int nL = 0;
    for (int iL = 0; iL < 3; iL++) {
      if (mActiveLinks[ruIndex][iL]) {
        nL++; // count active links
      }
    }
    std::vector<short int> chipEnabled = getChipListFromRu(ruIndex, mActiveLinks[ruIndex]); // chip boundaries
    // Fill the chipDone info string
    if (mRunTypeRUCopy[ruIndex] == nInjScaled * nL) {
      for (short int iChip = 0; iChip < chipEnabled.size(); iChip++) {
        if ((chipEnabled[iChip] % mChipModBase) != mChipModSel) {
          continue;
        }
        addDatabaseEntry(chipEnabled[iChip], "", std::vector<float>(), true);
      }
      mRunTypeRUCopy[ruIndex] = 0; // reset here is safer (the other counter is reset in finalize)
    }
    // Check if scan of a row is finished: only for specific scans!
    bool passCondition = (mCdwCntRU[ruIndex][row] >= nInjScaled * nL);
    if (mScanType == 'p' || mScanType == 't') {
      passCondition = passCondition && (mLoopVal[ruIndex][row] == mMax);
      if (mVerboseOutput) {
        LOG(info) << "PassCondition: " << passCondition << " - (mCdwCntRU,mLoopVal) of RU" << ruIndex << " row " << row << " = (" << mCdwCntRU[ruIndex][row] << ", " << mLoopVal[ruIndex][row] << ")";
      }
    } else if (mVerboseOutput) {
      LOG(info) << "PassCondition: " << passCondition << " - mCdwCntRU of RU" << ruIndex << " row " << row << " = " << mCdwCntRU[ruIndex][row];
    }

    if (mScanType != 'D' && mScanType != 'A' && mScanType != 'P' && mScanType != 'R' && passCondition) {
      // extract data from the row
      for (short int iChip = 0; iChip < chipEnabled.size(); iChip++) {
        short int chipID = chipEnabled[iChip];
        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }
        if (!isDumpS || (std::find(chipDumpList.begin(), chipDumpList.end(), chipID) != chipDumpList.end() || !chipDumpList.size())) { // to dump s-curves as histograms
          if (mPixelHits.count(chipID)) {
            if (mPixelHits[chipID].count(row)) { // make sure the row exists
              extractAndUpdate(chipID, row);
              if (mScanType != 'p' && (mScanType != 'r' || mLoopVal[ruIndex][row] == mMax)) { // do not erase for scantype = p because in finalize() we have calculate2Dparams
                mPixelHits[chipID].erase(row);
              }
            }
          }
        }
      }
      mCdwCntRU[ruIndex][row] = 0; // reset
    }
  } // for (ROFs)

  if (!(this->mRunTypeUp)) {
    finalize();
    LOG(info) << "Shipping all outputs to aggregator (before endOfStream arrival!)";
    pc.outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning);
    pc.outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    pc.outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    pc.outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    pc.outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    pc.outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    pc.outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
    // reset the DCSconfigObject_t before next ship out
    mTuning.clear();
    mPixStat.clear();
    mChipDoneQc.clear();
  } else if (pc.transitionState() == TransitionHandlingState::Requested) {
    LOG(info) << "Run stop requested during the scan, sending output to aggregator and then stopping to process new data";
    mRunStopRequested = true;
    finalize();                                                                             // calculating average thresholds based on what's collected up to this moment
    pc.outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning); // dummy here
    pc.outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    pc.outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    pc.outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    pc.outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    pc.outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    pc.outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
    mChipDoneQc.clear();
    mPixStat.clear();
    mTuning.clear();
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Retrieve conf DB map from production ccdb
void ITSThresholdCalibrator::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CONFDBMAP", 0)) {
    LOG(info) << "Conf DB map retrieved from CCDB";
    mConfDBmap = (std::vector<int>*)obj;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Calculate the average threshold given a vector of threshold objects
void ITSThresholdCalibrator::findAverage(const std::array<long int, 6>& data, float& avgT, float& rmsT, float& avgN, float& rmsN)
{
  avgT = (!data[4]) ? 0. : (float)data[0] / (float)data[4];
  rmsT = (!data[4]) ? 0. : std::sqrt((float)data[1] / (float)data[4] - avgT * avgT);
  avgN = (!data[4]) ? 0. : (float)data[2] / (float)data[4];
  rmsN = (!data[4]) ? 0. : std::sqrt((float)data[3] / (float)data[4] - avgN * avgN);
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::addDatabaseEntry(
  const short int& chipID, const char* name, std::vector<float> data, bool isQC)
{

  /*--------------------------------------------------------------------
  // Format of *data
  // - Threshold scan: avgT, rmsT, avgN, rmsN, status
  // - ITHR/VCASN scan: avg value, rms value, 0, 0, status (0 are just placeholder since they will not be used)
  // - Dig/Ana scans: empty vector
  // - Pulse shape 1D: avgRt, rmsRt, avgToT, rmsToT
  // - Pulse shape 2D: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  */
  // Obtain specific chip information from the chip ID (layer, stave, ...)
  int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
  this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

  char stave[6];
  snprintf(stave, 6, "L%d_%02d", lay, sta);

  if (isQC) {
    o2::dcs::addConfigItem(this->mChipDoneQc, "O2ChipID", std::to_string(chipID));
    return;
  }

  // Get ConfDB id for the chip chipID
  int confDBid = (*mConfDBmap)[chipID];

  // Bad pix list and bad dcols for dig and ana scan
  if (this->mScanType == 'D' || this->mScanType == 'A') {
    short int vPixDcolCounter[512] = {0}; // count #bad_pix per dcol
    std::string dcolIDs = "";
    std::string pixIDs_Noisy = "";
    std::string pixIDs_Dead = "";
    std::string pixIDs_Ineff = "";
    std::vector<int>& v = PixelType == "Noisy" ? mNoisyPixID[chipID] : PixelType == "Dead" ? mDeadPixID[chipID]
                                                                                           : mIneffPixID[chipID];
    // Number of pixel types
    int n_pixel = v.size(), nDcols = 0;
    std::string ds = "-1"; // dummy string
    // find bad dcols and add them one by one
    if (PixelType == "Noisy") {
      for (int i = 0; i < v.size(); i++) {
        short int dcol = ((v[i] - v[i] % 1000) / 1000) / 2;
        vPixDcolCounter[dcol]++;
      }
      for (int i = 0; i < 512; i++) {
        if (vPixDcolCounter[i] > N_PIX_DCOL) {
          o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
          o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
          o2::dcs::addConfigItem(this->mTuning, "Dcol", std::to_string(i));
          o2::dcs::addConfigItem(this->mTuning, "Row", ds);
          o2::dcs::addConfigItem(this->mTuning, "Col", ds);

          dcolIDs += std::to_string(i) + '|'; // prepare string for second object for ccdb prod
          nDcols++;
        }
      }
    }

    if (this->mTagSinglePix) {
      if (PixelType == "Noisy") {
        for (int i = 0; i < v.size(); i++) {
          short int dcol = ((v[i] - v[i] % 1000) / 1000) / 2;
          if (vPixDcolCounter[dcol] > N_PIX_DCOL) { // single pixels must not be already in dcolIDs
            continue;
          }

          // Noisy pixel IDs
          pixIDs_Noisy += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Noisy += '|';
          }

          o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
          o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
          o2::dcs::addConfigItem(this->mTuning, "Dcol", ds);
          o2::dcs::addConfigItem(this->mTuning, "Row", std::to_string(v[i] % 1000));
          o2::dcs::addConfigItem(this->mTuning, "Col", std::to_string(int((v[i] - v[i] % 1000) / 1000)));
        }
      }

      if (PixelType == "Dead") {
        for (int i = 0; i < v.size(); i++) {
          pixIDs_Dead += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Dead += '|';
          }
        }
      }

      if (PixelType == "Ineff") {
        for (int i = 0; i < v.size(); i++) {
          pixIDs_Ineff += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Ineff += '|';
          }
        }
      }
    }

    if (!dcolIDs.empty()) {
      dcolIDs.pop_back(); // remove last pipe from the string
    } else {
      dcolIDs = "-1";
    }

    if (pixIDs_Noisy.empty()) {
      pixIDs_Noisy = "-1";
    }

    if (pixIDs_Dead.empty()) {
      pixIDs_Dead = "-1";
    }

    if (pixIDs_Ineff.empty()) {
      pixIDs_Ineff = "-1";
    }

    o2::dcs::addConfigItem(this->mPixStat, "O2ChipID", std::to_string(chipID));
    if (PixelType == "Dead" || PixelType == "Ineff") {
      o2::dcs::addConfigItem(this->mPixStat, "PixelType", PixelType);
      o2::dcs::addConfigItem(this->mPixStat, "PixelNos", n_pixel);
      o2::dcs::addConfigItem(this->mPixStat, "DcolNos", "-1");
    } else {
      o2::dcs::addConfigItem(this->mPixStat, "PixelType", PixelType);
      o2::dcs::addConfigItem(this->mPixStat, "PixelNos", n_pixel);
      o2::dcs::addConfigItem(this->mPixStat, "DcolNos", nDcols);
    }
  }
  if (this->mScanType != 'D' && this->mScanType != 'A' && this->mScanType != 'P' && this->mScanType != 'p') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, name, (strcmp(name, "ITHR") == 0 || strcmp(name, "VCASN") == 0) ? std::to_string((int)data[0]) : std::to_string(data[0]));
    o2::dcs::addConfigItem(this->mTuning, "Rms", std::to_string(data[1]));
    o2::dcs::addConfigItem(this->mTuning, "Status", std::to_string((int)data[4])); // percentage of unsuccess
  }
  if (this->mScanType == 'T') {
    o2::dcs::addConfigItem(this->mTuning, "Noise", std::to_string(data[2]));
    o2::dcs::addConfigItem(this->mTuning, "NoiseRms", std::to_string(data[3]));
  }

  if (this->mScanType == 'P') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, "Tot", std::to_string(data[2]));    // time over threshold
    o2::dcs::addConfigItem(this->mTuning, "TotRms", std::to_string(data[3])); // time over threshold rms
    o2::dcs::addConfigItem(this->mTuning, "Rt", std::to_string(data[0]));     // rise time
    o2::dcs::addConfigItem(this->mTuning, "RtRms", std::to_string(data[1]));  // rise time rms
  }

  //- Pulse shape 2D: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  if (this->mScanType == 'p') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, "Tot", std::to_string(data[0]));             // time over threshold
    o2::dcs::addConfigItem(this->mTuning, "TotRms", std::to_string(data[1]));          // time over threshold rms
    o2::dcs::addConfigItem(this->mTuning, "MinThrChg", std::to_string(data[2]));       // Min threshold charge
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgRms", std::to_string(data[3]));    // Min threshold charge rms
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgDel", std::to_string(data[4]));    // Min threshold charge delay
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgDelRms", std::to_string(data[5])); // Min threshold charge delay rms
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLen", std::to_string(data[6]));       // Max pulse length
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenRms", std::to_string(data[7]));    // Max pulse length rms
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenChg", std::to_string(data[8]));    // Max pulse length charge
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenChgRms", std::to_string(data[9])); // Max pulse length charge rms
  }

  return;
}

//___________________________________________________________________
void ITSThresholdCalibrator::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    mDataTakingContext = pc.services().get<DataTakingContext>();
  }
  mTimingInfo = pc.services().get<o2::framework::TimingInfo>();
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::finalize()
{
  // Add configuration item to output strings for CCDB
  const char* name = nullptr;
  std::set<int> thisRUs;

  if (mScanType == 'V' || mScanType == 'I' || mScanType == 'T') {
    // Loop over each chip and calculate avg and rms
    name = mScanType == 'V' ? "VCASN" : mScanType == 'I' ? "ITHR"
                                                         : "THR";
    if (mScanType == 'I') {
      // Only ITHR scan: assign default ITHR = 50 if chip has no avg ITHR
      for (auto& iRU : mRuSet) {
        if (mRunTypeRU[iRU] >= nInjScaled * getNumberOfActiveLinks(mActiveLinks[iRU]) || mRunStopRequested) {
          std::vector<short int> chipList = getChipListFromRu(iRU, mActiveLinks[iRU]);
          for (size_t i = 0; i < chipList.size(); i++) {
            if ((chipList[i] % mChipModBase) != mChipModSel) {
              continue;
            }
            if (!mThresholds.count(chipList[i])) {
              if (mVerboseOutput) {
                LOG(info) << "Setting ITHR = 50 for chip " << chipList[i];
              }
              std::vector<float> data = {50, 0, 0, 0, 0};
              addDatabaseEntry(chipList[i], name, data, false);
            }
          }
        }
      }
    }

    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      short int iRU = getRUID(it->first);
      if (!isCRUITS && (mRunTypeRU[iRU] < nInjScaled * getNumberOfActiveLinks(mActiveLinks[iRU]) && !mRunStopRequested)) {
        ++it;
        continue;
      }
      thisRUs.insert(iRU);
      float avgT, rmsT, avgN, rmsN, mpvT, outVal;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      outVal = avgT;
      if (isMpv) {
        mpvT = std::distance(mpvCounter[it->first].begin(), std::max_element(mpvCounter[it->first].begin(), mpvCounter[it->first].end())) + mMin;
        outVal = mpvT;
      }
      if (mVerboseOutput) {
        LOG(info) << "Average or mpv " << name << " of chip " << it->first << " = " << outVal << " e-";
      }
      float status = ((float)it->second[4] / (float)(it->second[4] + it->second[5])) * 100.; // percentage of successful threshold extractions
      if (status < mPercentageCut && (mScanType == 'I' || mScanType == 'V')) {
        if (mScanType == 'I') { // default ITHR if percentage of success < mPercentageCut
          outVal = 50.;
          if (mVerboseOutput) {
            LOG(info) << "Chip " << it->first << " status is " << status << ". Setting ITHR = 50";
          }
        } else { // better to not set any VCASN if the percentage of success < mPercentageCut
          it = this->mThresholds.erase(it);
          if (mVerboseOutput) {
            LOG(info) << "Chip " << it->first << " status is " << status << ". Ignoring this chip.";
          }
          continue;
        }
      }
      std::vector<float> data = {outVal, rmsT, avgN, rmsN, status};
      this->addDatabaseEntry(it->first, name, data, false);
      it = this->mThresholds.erase(it);
    }
  } else if (this->mScanType == 'D' || this->mScanType == 'A') {
    // Loop over each chip and calculate avg and rms
    name = "PixID";
    // Extract hits from the full matrix
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != this->mPixelHits.cend()) { // loop over chips collected
      short int iRU = getRUID(itchip->first);
      if (!isCRUITS && (mRunTypeRU[iRU] < nInjScaled * getNumberOfActiveLinks(mActiveLinks[iRU]) && !mRunStopRequested)) {
        ++itchip;
        continue;
      }
      thisRUs.insert(iRU);
      if (mVerboseOutput) {
        LOG(info) << "Extracting hits for the full matrix of chip " << itchip->first;
      }
      for (short int irow = 0; irow < 512; irow++) {
        this->extractAndUpdate(itchip->first, irow);
      }
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << itchip->first << " hits extracted";
      }
      ++itchip;
    }

    auto it = this->mNoisyPixID.cbegin();
    while (it != this->mNoisyPixID.cend()) {
      PixelType = "Noisy";
      if (mVerboseOutput) {
        LOG(info) << "Extracting noisy pixels in the full matrix of chip " << it->first;
      }
      this->addDatabaseEntry(it->first, name, std::vector<float>(), false); // all zeros are not used here
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << it->first << " done";
      }
      it = this->mNoisyPixID.erase(it);
    }

    auto it_d = this->mDeadPixID.cbegin();
    while (it_d != this->mDeadPixID.cend()) {
      if (mVerboseOutput) {
        LOG(info) << "Extracting dead pixels in the full matrix of chip " << it_d->first;
      }
      PixelType = "Dead";
      this->addDatabaseEntry(it_d->first, name, std::vector<float>(), false); // all zeros are not used here
      it_d = this->mDeadPixID.erase(it_d);
    }

    auto it_ineff = this->mIneffPixID.cbegin();
    while (it_ineff != this->mIneffPixID.cend()) {
      if (mVerboseOutput) {
        LOG(info) << "Extracting inefficient pixels in the full matrix of chip " << it_ineff->first;
      }
      PixelType = "Ineff";
      this->addDatabaseEntry(it_ineff->first, name, std::vector<float>(), false);
      it_ineff = this->mIneffPixID.erase(it_ineff);
    }
  } else if (this->mScanType == 'P' || this->mScanType == 'p' || mScanType == 'R') { // pulse length scan 1D and 2D, vresetd scan 1D (2D already extracted in run())
    name = "Pulse";
    // extract hits for the available row(s)
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != mPixelHits.cend()) {
      int iRU = getRUID(itchip->first);
      if (!mRunStopRequested && mRunTypeRU[iRU] < nInjScaled * getNumberOfActiveLinks(mActiveLinks[iRU])) {
        ++itchip;
        continue;
      }
      thisRUs.insert(iRU);
      if (mVerboseOutput) {
        LOG(info) << "Extracting hits from pulse shape scan or vresetd scan, chip " << itchip->first;
      }

      if (mScanType != 'p') { // done already in run()
        auto itrow = this->mPixelHits[itchip->first].cbegin();
        while (itrow != mPixelHits[itchip->first].cend()) {    // in case there are multiple rows, for now it's 1 row
          this->extractAndUpdate(itchip->first, itrow->first); // fill the tree - for mScanType = p, it is done already in run()
          ++itrow;
        }
      }

      if (mCalculate2DParams && (mScanType == 'P' || mScanType == 'p')) {
        this->addDatabaseEntry(itchip->first, name, mScanType == 'P' ? calculatePulseParams(itchip->first) : calculatePulseParams2D(itchip->first), false);
      }
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << itchip->first << " hits extracted";
      }
      ++itchip;
    }
    // reset RU counters so that the chips which are done will not appear again in the DCSConfigObject
  }

  for (auto& ru : thisRUs) {
    mRunTypeRU[ru] = 0; // reset
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSThresholdCalibrator::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOGF(info, "endOfStream report:", mSelfName);
    if (isCRUITS) {
      finalize();
    }
    this->finalizeOutput();
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void ITSThresholdCalibrator::stop()
{
  if (!isEnded) {
    LOGF(info, "stop() report:", mSelfName);
    this->finalizeOutput();
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSThresholdCalibratorSpec(const ITSCalibInpConf& inpConf)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);
  // inputs.emplace_back("confdbmap", detOrig, "CONFDBMAP", 0, Lifetime::Condition,
  //                     o2::framework::ccdbParamSpec("ITS/Calib/Confdbmap"));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TSTR", inpConf.chipModSel);
  outputs.emplace_back("ITS", "PIXTYP", inpConf.chipModSel);
  outputs.emplace_back("ITS", "RUNT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "SCANT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "FITT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "CONFDBV", inpConf.chipModSel);
  outputs.emplace_back("ITS", "QCSTR", inpConf.chipModSel);

  return DataProcessorSpec{
    "its-calibrator_" + std::to_string(inpConf.chipModSel),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSThresholdCalibrator>(inpConf)},
    Options{{"fittype", VariantType::String, "derivative", {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}},
            {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
            {"output-dir", VariantType::String, "./", {"ROOT trees output directory"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"Metadata output directory"}},
            {"meta-type", VariantType::String, "", {"metadata type"}},
            {"nthreads", VariantType::Int, 1, {"Number of threads, default is 1"}},
            {"enable-cw-cnt-check", VariantType::Bool, false, {"Use to enable the check of the calib word counter row by row in addition to the hits"}},
            {"enable-single-pix-tag", VariantType::Bool, false, {"Use to enable tagging of single noisy pix in digital and analogue scan"}},
            {"ccdb-mgr-url", VariantType::String, "", {"CCDB url to download confDBmap"}},
            {"min-vcasn", VariantType::Int, 30, {"Min value of VCASN in vcasn scan, default is 30"}},
            {"max-vcasn", VariantType::Int, 70, {"Max value of VCASN in vcasn scan, default is 70"}},
            {"min-ithr", VariantType::Int, 25, {"Min value of ITHR in ithr scan, default is 25"}},
            {"max-ithr", VariantType::Int, 100, {"Max value of ITHR in ithr scan, default is 100"}},
            {"manual-mode", VariantType::Bool, false, {"Flag to activate the manual mode in case run type is not recognized"}},
            {"manual-min", VariantType::Int, 0, {"Min value of the variable used for the scan: use only in manual mode"}},
            {"manual-max", VariantType::Int, 50, {"Max value of the variable used for the scan: use only in manual mode"}},
            {"manual-min2", VariantType::Int, 0, {"Min2 value of the 2nd variable (if any) used for the scan (ex: charge in tot_calib): use only in manual mode"}},
            {"manual-max2", VariantType::Int, 50, {"Max2 value of the 2nd variable (if any) used for the scan (ex: charge in tot_calib): use only in manual mode"}},
            {"manual-step", VariantType::Int, 1, {"Step value: defines the steps between manual-min and manual-max. Default is 1. Use only in manual mode"}},
            {"manual-step2", VariantType::Int, 1, {"Step2 value: defines the steps between manual-min2 and manual-max2. Default is 1. Use only in manual mode"}},
            {"manual-scantype", VariantType::String, "T", {"scan type, can be D, T, I, V, P, p: use only in manual mode"}},
            {"manual-strobewindow", VariantType::Int, 5, {"strobe duration in clock cycles, default is 5 = 125 ns: use only in manual mode"}},
            {"save-tree", VariantType::Bool, false, {"Flag to save ROOT tree on disk: use only in manual mode"}},
            {"scale-ninj", VariantType::Bool, false, {"Flag to activate the scale of the number of injects to be used to count hits from specific MEBs: use only in manual mode and in combination with --meb-select"}},
            {"enable-mpv", VariantType::Bool, false, {"Flag to enable calculation of most-probable value in vcasn/ithr scans"}},
            {"enable-cru-its", VariantType::Bool, false, {"Flag to enable the analysis of raw data on disk produced by CRU_ITS IB commissioning tools"}},
            {"ninj", VariantType::Int, 50, {"Number of injections per change, default is 50"}},
            {"dump-scurves", VariantType::Bool, false, {"Dump any s-curve to disk in ROOT file. Works only with fit option."}},
            {"max-dump", VariantType::Int, -1, {"Maximum number of s-curves to dump in ROOT file per chip. Works with fit option and dump-scurves flag enabled. Default: dump all"}},
            {"chip-dump", VariantType::String, "", {"Dump s-curves only for these Chip IDs (0 to 24119). If multiple IDs, write them separated by comma. Default is empty string: dump all"}},
            {"calculate-slope", VariantType::Bool, false, {"For Pulse Shape 2D: if enabled it calculate the slope of the charge vs strobe delay trend for each pixel and fill it in the output tree"}},
            {"charge-a", VariantType::Int, 0, {"To use with --calculate-slope, it defines the charge (in DAC) for the 1st point used for the slope calculation"}},
            {"charge-b", VariantType::Int, 0, {"To use with --calculate-slope, it defines the charge (in DAC) for the 2nd point used for the slope calculation"}},
            {"meb-select", VariantType::Int, -1, {"Select from which multi-event buffer consider the hits: 0,1 or 2"}},
            {"s-curve-col-step", VariantType::Int, 8, {"save s-curves points to tree every s-curve-col-step  pixels on 1 row"}},
            {"percentage-cut", VariantType::Int, 25, {"discard chip in ITHR/VCASN scan if the percentage of success is less than this cut"}}}};
}
} // namespace its
} // namespace o2
