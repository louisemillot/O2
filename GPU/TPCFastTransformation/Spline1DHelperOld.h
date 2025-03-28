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

/// \file  Spline1DHelperOld.h
/// \brief Definition of Spline1DHelperOld class

/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline1DHelperOld_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline1DHelperOld_H

#include <cmath>
#include <vector>

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "Spline1D.h"
#include <functional>
#include <string>

namespace o2
{
namespace gpu
{
///
/// The Spline1DHelperOld class is to initialize parameters for Spline1D class
///
template <typename DataT>
class Spline1DHelperOld
{
 public:
  ///
  /// \brief Helper structure for 1D spline construction
  ///
  struct DataPoint {
    double u;      ///< u coordinate
    double cS0;    ///< a coefficient for s0
    double cZ0;    ///< a coefficient for s'0
    double cS1;    ///< a coefficient for s1
    double cZ1;    ///< a coefficient for s'1
    int32_t iKnot; ///< index of the left knot of the segment
    bool isKnot;   ///< is the point placed at a knot
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  Spline1DHelperOld();

  /// Copy constructor: disabled
  Spline1DHelperOld(const Spline1DHelperOld&) = default;

  /// Assignment operator: disabled
  Spline1DHelperOld& operator=(const Spline1DHelperOld&) = default;

  /// Destructor
  ~Spline1DHelperOld() = default;

  /// _______________  Main functionality  ________________________

  void bandGauss(double A[], double b[], int32_t n);

  /// Create best-fit spline parameters for a given input function F
  void approximateDataPoints(Spline1DContainer<DataT>& spline,
                             double xMin, double xMax,
                             double x[], double f[], int32_t nDataPoints);

  /// Create best-fit spline parameters for a given input function F
  void approximateFunction(Spline1DContainer<DataT>& spline,
                           double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
                           int32_t nAuxiliaryDataPoints = 4);

  /// Create best-fit spline parameters gradually for a given input function F
  void approximateFunctionGradually(Spline1DContainer<DataT>& spline,
                                    double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
                                    int32_t nAuxiliaryDataPoints = 4);

  /// Create classic spline parameters for a given input function F
  void approximateFunctionClassic(Spline1DContainer<DataT>& spline,
                                  double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F);

  /// _______________   Interface for a step-wise construction of the best-fit spline   ________________________

  /// precompute everything needed for the construction
  int32_t setSpline(const Spline1DContainer<DataT>& spline, int32_t nFdimensions, int32_t nAuxiliaryDataPoints);

  /// precompute everything needed for the construction
  int32_t setSpline(const Spline1DContainer<DataT>& spline, int32_t nFdimensions, double xMin, double xMax, double vx[], int32_t nDataPoints);

  /// approximate std::function, output in Fparameters
  void approximateFunction(DataT* Fparameters, double xMin, double xMax, std::function<void(double x, double f[])> F) const;

  /// approximate std::function gradually, output in Fparameters
  void approximateFunctionGradually(DataT* Fparameters, double xMin, double xMax, std::function<void(double x, double f[])> F) const;

  /// number of data points
  int32_t getNumberOfDataPoints() const { return mDataPoints.size(); }

  /// approximate a function given as an array of values at data points
  void approximateFunction(DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// gradually approximate a function given as an array of values at data points
  void approximateFunctionGradually(DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim */]) const;

  /// a tool for the gradual approximation: set spline values S_i at knots == function values
  void copySfromDataPoints(DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// a tool for the gradual approximation:
  /// calibrate spline derivatives D_i using already calibrated spline values S_i
  void approximateDerivatives(DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// _______________  Utilities   ________________________

  const Spline1D<double>& getSpline() const { return mSpline; }

  int32_t getKnotDataPoint(int32_t iknot) const { return mKnotDataPoints[iknot]; }

  const DataPoint& getDataPoint(int32_t ip) const { return mDataPoints[ip]; }

  /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
  /// over the spline values Sl, Sr and the slopes Dl, Dr
  static void getScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                               double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                 double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficientsLeft(const typename Spline1D<double>::Knot& knotL,
                                     double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficientsRight(const typename Spline1D<double>::Knot& knotL,
                                      double& cSl, double& cDl, double& cSr, double& cDr);
  static void getDDDScoefficients(const typename Spline1D<double>::Knot& knotL,
                                  double& cSl, double& cDl, double& cSr, double& cDr);

  ///  Gives error string
  const char* getLastError() const { return mError.c_str(); }

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the Spline1D class functionality
  static int32_t test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

 private:
  /// Stores an error message
  int32_t storeError(int32_t code, const char* msg);

  std::string mError = ""; ///< error string

  /// helpers for the construction of 1D spline

  Spline1D<double> mSpline;             ///< copy of the spline
  int32_t mFdimensions;                 ///< n of F dimensions
  std::vector<DataPoint> mDataPoints;   ///< measurement points
  std::vector<int32_t> mKnotDataPoints; ///< which measurement points are at knots
  std::vector<double> mLSMmatrixFull;   ///< a matrix to convert the measurements into the spline parameters with the LSM method
  std::vector<double> mLSMmatrixSderivatives;
  std::vector<double> mLSMmatrixSvalues;

  ClassDefNV(Spline1DHelperOld, 0);
};

} // namespace gpu
} // namespace o2

#endif
