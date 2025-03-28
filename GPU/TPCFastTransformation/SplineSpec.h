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

/// \file  SplineSpec.h
/// \brief Definition of SplineSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINESPEC_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINESPEC_H

#include "Spline1D.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"
#include "SplineUtil.h"

#if !defined(__ROOTCLING__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC)
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

class TFile;

namespace o2
{
namespace gpu
{

/// ==================================================================================================
/// The class SplineContainer is a base Spline class.
/// It contains all the class members and those methods which only depends on the DataT data type.
/// It also contains all non-inlined methods with the implementation in SplineSpec.cxx file.
///
/// DataT is a data type, which is supposed to be either double or float.
/// For other possible data types one has to add the corresponding instantiation line
/// at the end of the SplineSpec.cxx file
///
template <typename DataT>
class SplineContainer : public FlatObject
{
 public:
  typedef typename Spline1D<DataT>::SafetyLevel SafetyLevel;
  typedef typename Spline1D<DataT>::Knot Knot;

  /// _____________  Version control __________________________

  /// Version control
  GPUd() static constexpr int32_t getVersion() { return (1 << 16) + Spline1D<DataT>::getVersion(); }

  /// _____________  C++ constructors / destructors __________________________

  /// Default constructor
  SplineContainer() = default;

  /// Disable all other constructors
  SplineContainer(const SplineContainer&) = delete;

  /// Destructor
  ~SplineContainer() = default;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline
  void approximateFunction(const double xMin[/* mXdim */], const double xMax[/* mXdim */],
                           std::function<void(const double x[/* mXdim */], double f[/*mYdim*/])> F,
                           const int32_t nAuxiliaryDataPoints[/* mXdim */] = nullptr);
#endif

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int32_t writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static SplineContainer* readFromFile(TFile& inpf, const char* name);
#endif

  /// _______________  Getters   ________________________

  /// Get number of X dimensions
  GPUd() int32_t getXdimensions() const { return mXdim; }

  /// Get number of Y dimensions
  GPUd() int32_t getYdimensions() const { return mYdim; }

  /// Get minimal required alignment for the spline parameters
  GPUd() static constexpr size_t getParameterAlignmentBytes() { return 16; }

  /// Number of parameters
  GPUd() int32_t getNumberOfParameters() const { return this->calcNumberOfParameters(mYdim); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return sizeof(DataT) * this->getNumberOfParameters(); }

  /// Get a number of knots
  GPUd() int32_t getNumberOfKnots() const { return mNknots; }

  /// Number of parameters per knot
  GPUd() int32_t getNumberOfParametersPerKnot() const { return calcNumberOfParametersPerKnot(mYdim); }

  /// Get 1-D grid for dimX dimension
  GPUd() const Spline1D<DataT>& getGrid(int32_t dimX) const { return mGrid[dimX]; }

  /// Get u[] coordinate of i-th knot
  GPUd() void getKnotU(int32_t iKnot, int32_t u[/* mXdim */]) const;

  /// Get index of a knot (iKnot1,iKnot2,..,iKnotN)
  GPUd() int32_t getKnotIndex(const int32_t iKnot[/* mXdim */]) const;

  /// Get spline parameters
  GPUd() DataT* getParameters() { return mParameters; }

  /// Get spline parameters const
  GPUd() const DataT* getParameters() const { return mParameters; }

  /// _______________  Technical stuff  ________________________

  /// Get offset of Grid[dimX] flat data in the flat buffer
  GPUd() size_t getGridOffset(int32_t dimX) const { return mGrid[dimX].getFlatBufferPtr() - mFlatBufferPtr; }

  /// Set X range
  GPUd() void setXrange(const DataT xMin[/* mXdim */], const DataT xMax[/* mXdim */]);

  /// Print method
  void print() const;

  ///  _______________  Expert tools  _______________

  /// Number of parameters for given Y dimensions
  GPUd() int32_t calcNumberOfParameters(int32_t nYdim) const
  {
    return calcNumberOfParametersPerKnot(nYdim) * getNumberOfKnots();
  }

  /// Number of parameters per knot
  GPUd() int32_t calcNumberOfParametersPerKnot(int32_t nYdim) const
  {
    return (1 << mXdim) * nYdim; // 2^mXdim parameters per Y dimension
  }

  ///_______________  Test tools  _______________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int32_t test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const SplineContainer& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);
#endif

  using FlatObject::releaseInternalBuffer;

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 protected:
#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline
  void recreate(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */]);

  /// Constructor for an irregular spline
  void recreate(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */], const int32_t* const knotU[/* nXdim */]);
#endif

  /// _____________  Data members  ____________

  int32_t mXdim = 0;   ///< dimentionality of X
  int32_t mYdim = 0;   ///< dimentionality of Y
  int32_t mNknots = 0; ///< number of spline knots

  Spline1D<DataT>* mGrid; //! (transient!!) mXdim grids
  DataT* mParameters;     //! (transient!!) F-dependent parameters of the spline

  ClassDefNV(SplineContainer, 1);
};

template <typename DataT>
GPUdi() void SplineContainer<DataT>::getKnotU(int32_t iKnot, int32_t u[/* mXdim */]) const
{
  /// Get u[] coordinate of i-th knot
  for (int32_t dim = 0; dim < mXdim; dim++) {
    int32_t n = mGrid[dim].getNumberOfKnots();
    u[dim] = mGrid[dim].getKnot(iKnot % n).getU();
    iKnot /= n;
  }
}

template <typename DataT>
GPUdi() int32_t SplineContainer<DataT>::getKnotIndex(const int32_t iKnot[/* mXdim */]) const
{
  /// Get index of a knot (iKnot1,iKnot2,..,iKnotN)
  int32_t ind = iKnot[0];
  int32_t n = 1;
  for (int32_t dim = 1; dim < mXdim; dim++) {
    n *= mGrid[dim - 1].getNumberOfKnots();
    ind += n * iKnot[dim];
  }
  return ind;
}

template <typename DataT>
GPUdi() void SplineContainer<DataT>::
  setXrange(const DataT xMin[/* mXdim */], const DataT xMax[/* mXdim */])
{
  /// Set X range
  for (int32_t i = 0; i < mXdim; i++) {
    mGrid[i].setXrange(xMin[i], xMax[i]);
  }
}

/// ==================================================================================================
///
/// SplineSpec class declares different specializations of the Spline class.
/// (See Spline.h for the description.)
///
/// The specializations depend on the value of Spline's template parameters XdimT and YdimT.
/// specializations have different constructors and slightly different declarations of methods.
///
/// The meaning of the template parameters:
///
/// \param DataT data type: float or double
/// \param XdimT
///    XdimT > 0 : the number of X dimensions is known at the compile time and is equal to XdimT
///    XdimT = 0 : the number of X dimensions will be set in the runtime
///    XdimT < 0 : the number of X dimensions will be set in the runtime, and it will not exceed abs(XdimT)
/// \param YdimT same for the X dimensions
/// \param SpecT specialisation number:
///  0 - a parent class for all other specializations
///  1 - nXdim>0, nYdim>0: both nXdim and nYdim are set at the compile time
///  2 - at least one of the dimensions must be set during runtime
///  3 - specialization where nYdim==1 (a small add-on on top of the other specs)
///
template <typename DataT, int32_t XdimT, int32_t YdimT, int32_t SpecT>
class SplineSpec;

/// ==================================================================================================
/// Specialization 0 declares common methods for all other Spline specializations.
/// Implementations of the methods may depend on the YdimT value.
///
template <typename DataT, int32_t XdimT, int32_t YdimT>
class SplineSpec<DataT, XdimT, YdimT, 0> : public SplineContainer<DataT>
{
  typedef SplineContainer<DataT> TBase;

 public:
  typedef typename TBase::SafetyLevel SafetyLevel;
  typedef typename TBase::Knot Knot;

  /// _______________  Interpolation math   ________________________

  /// Get interpolated value S(x)
  GPUd() void interpolate(const DataT x[/*mXdim*/], GPUgeneric() DataT S[/*mYdim*/]) const
  {
    const auto nXdimTmp = SplineUtil::getNdim<XdimT>(mXdim);
    const auto nXdim = nXdimTmp.get();
    const auto maxXdimTmp = SplineUtil::getMaxNdim<XdimT>(mXdim);
    DataT u[maxXdimTmp.get()];
    for (int32_t i = 0; i < nXdim; i++) {
      u[i] = mGrid[i].convXtoU(x[i]);
    }
    interpolateU<SafetyLevel::kSafe>(mXdim, mYdim, mParameters, u, S);
  }

  /// Get interpolated value for S(u):inpXdim->inpYdim using spline parameters Parameters
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateU(int32_t inpXdim, int32_t inpYdim, GPUgeneric() const DataT Parameters[],
                           const DataT u[/*inpXdim*/], GPUgeneric() DataT S[/*inpYdim*/]) const
  {
    const auto nXdimTmp = SplineUtil::getNdim<XdimT>(mXdim);
    const auto nXdim = nXdimTmp.get();
    const auto maxXdimTmp = SplineUtil::getMaxNdim<XdimT>(mXdim);
    const auto maxXdim = maxXdimTmp.get();
    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(mYdim);
    const auto nYdim = nYdimTmp.get();
    const auto maxYdimTmp = SplineUtil::getMaxNdim<XdimT>(mYdim);
    const auto maxYdim = maxYdimTmp.get();

    // const auto nParameters = 1 << (2 * nXdim);         //total Nr of Parameters necessary for one interpolation
    const auto nKnotParametersPerY = 1 << nXdim;       // Nr of Parameters per Knot per Y dimension
    const auto nKnotParameters = (1 << nXdim) * nYdim; // Nr of Parameters per Knot

    DataT iParameters[(1 << (2 * maxXdim)) * maxYdim]; // Array for all parameters

    // get the indices of the "most left" Knot:

    int32_t indices[maxXdim]; // indices of the 'most left' knot
    for (int32_t i = 0; i < nXdim; i++) {
      indices[i] = mGrid[i].getLeftKnotIndexForU(u[i]);
    }
    // get all the needed parameters into one array iParameters[nParameters]:
    int32_t indicestmp[maxXdim];
    for (int32_t i = 0; i < nKnotParametersPerY; i++) { // for every necessary Knot
      for (int32_t k = 0; k < nXdim; k++) {
        indicestmp[k] = indices[k] + (i / (1 << k)) % 2; // get the knot-indices in every dimension (mirrored order binary counting)
      }
      int32_t index = TBase::getKnotIndex(indicestmp); // get index of the current Knot

      for (int32_t j = 0; j < nKnotParameters; j++) { // and fill the iparameter array with according parameters
        iParameters[i * nKnotParameters + j] = Parameters[index * nKnotParameters + j];
      }
    }
    // now start with the interpolation loop:

    constexpr auto maxInterpolations = (1 << (2 * maxXdim - 2)) * maxYdim;

    DataT S0[maxInterpolations];
    DataT D0[maxInterpolations];
    DataT S1[maxInterpolations];
    DataT D1[maxInterpolations];

    int32_t nInterpolations = (1 << (2 * nXdim - 2)) * nYdim;
    int32_t nKnots = 1 << (nXdim);

    for (int32_t d = 0; d < nXdim; d++) {            // for every dimension
      DataT* pointer[4] = {S0, D0, S1, D1};          // pointers for interpolation arrays S0, D0, S1, D1 point to Arraystart
      for (int32_t i = 0; i < nKnots; i++) {         // for every knot
        for (int32_t j = 0; j < nKnots; j++) {       // for every parametertype
          int32_t pointernr = 2 * (i % 2) + (j % 2); // to which array should it be delivered
          for (int32_t k = 0; k < nYdim; k++) {
            pointer[pointernr][0] = iParameters[(i * nKnots + j) * nYdim + k];
            pointer[pointernr]++;
          }
        } // end for j (every parametertype)
      } // end for i (every knot)

      const typename Spline1D<DataT>::Knot& knotL = mGrid[d].getKnot(indices[d]);
      DataT coordinate = u[d];
      typedef Spline1DSpec<DataT, 0, 0> TGridX;
      const TGridX& gridX = *((const TGridX*)&(mGrid[d]));
      gridX.interpolateU(nInterpolations, knotL, S0, D0, S1, D1, coordinate, iParameters);
      nInterpolations /= 4;
      nKnots /= 2;
    } // end d (every dimension)

    for (int32_t i = 0; i < nYdim; i++) {
      S[i] = iParameters[i]; // write into result-array
      // LOG(info)<<iParameters[i] <<", ";
    }
  } // end interpolateU

 protected:
  using TBase::mGrid;
  using TBase::mParameters;
  using TBase::mXdim;
  using TBase::mYdim;
  using TBase::TBase; // inherit constructors and hide them
};

/// ==================================================================================================
/// Specialization 1: XdimT>0, YdimT>0 where the number of dimensions is taken from template parameters
/// at the compile time
///
template <typename DataT, int32_t XdimT, int32_t YdimT>
class SplineSpec<DataT, XdimT, YdimT, 1>
  : public SplineSpec<DataT, XdimT, YdimT, 0>
{
  typedef SplineContainer<DataT> TVeryBase;
  typedef SplineSpec<DataT, XdimT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  SplineSpec() : SplineSpec(nullptr) {}

  /// Constructor for a regular spline
  SplineSpec(const int32_t nKnots[/*XdimT*/]) : TBase()
  {
    recreate(nKnots);
  }
  /// Constructor for an irregular spline
  SplineSpec(const int32_t nKnots[/*XdimT*/], const int32_t* const knotU[/*XdimT*/])
    : TBase()
  {
    recreate(nKnots, knotU);
  }
  /// Copy constructor
  SplineSpec(const SplineSpec& v) : TBase()
  {
    TBase::cloneFromObject(v, nullptr);
  }
  /// Constructor for a regular spline
  void recreate(const int32_t nKnots[/*XdimT*/])
  {
    TBase::recreate(XdimT, YdimT, nKnots);
  }

  /// Constructor for an irregular spline
  void recreate(const int32_t nKnots[/*XdimT*/], const int32_t* const knotU[/*XdimT*/])
  {
    TBase::recreate(XdimT, YdimT, nKnots, knotU);
  }
#endif

  /// Get number of X dimensions
  GPUd() constexpr int32_t getXdimensions() const { return XdimT; }

  /// Get number of Y dimensions
  GPUd() constexpr int32_t getYdimensions() const { return YdimT; }

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  /// Get interpolated value for an YdimT-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateU(GPUgeneric() const DataT Parameters[],
                           const DataT u[/*XdimT*/], GPUgeneric() DataT S[/*YdimT*/]) const
  {
    TBase::template interpolateU<SafeT>(XdimT, YdimT, Parameters, u, S);
  }

  /// _______________  Suppress some parent class methods   ________________________
 private:
#if !defined(GPUCA_GPUCODE)
  using TBase::recreate;
#endif
  using TBase::interpolateU;
};

/// ==================================================================================================
/// Specialization 2 (XdimT<=0 || YdimT<=0) where at least one of the dimensions
/// must be set in the runtime via a constructor parameter
///
template <typename DataT, int32_t XdimT, int32_t YdimT>
class SplineSpec<DataT, XdimT, YdimT, 2>
  : public SplineSpec<DataT, XdimT, YdimT, 0>
{
  typedef SplineContainer<DataT> TVeryBase;
  typedef SplineSpec<DataT, XdimT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  SplineSpec() : SplineSpec((XdimT > 0 ? XdimT : 0), (YdimT > 0 ? YdimT : 0), nullptr) {}

  /// Constructor for a regular spline
  SplineSpec(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */]) : TBase()
  {
    this->recreate(nXdim, nYdim, nKnots);
  }

  /// Constructor for an irregular spline
  SplineSpec(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */], const int32_t* const knotU[/* nXdim */])
    : TBase()
  {
    this->recreate(nXdim, nYdim, nKnots, knotU);
  }

  /// Copy constructor
  SplineSpec(const SplineSpec& v) : TBase()
  {
    cloneFromObject(v, nullptr);
  }

  /// Constructor for a regular spline
  void recreate(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */])
  {
    checkDimensions(nXdim, nYdim);
    TBase::recreate(nXdim, nYdim, nKnots);
  }

  /// Constructor for an irregular spline
  void recreate(int32_t nXdim, int32_t nYdim, const int32_t nKnots[/* nXdim */], const int32_t* const knotU[/* nXdim */])
  {
    checkDimensions(nXdim, nYdim);
    TBase::recreate(nXdim, nYdim, nKnots, knotU);
  }

#endif

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  using TBase::interpolateU;

  /// Check dimensions
  void checkDimensions(int32_t& nXdim, int32_t& nYdim)
  {
    if (XdimT > 0 && nXdim != XdimT) {
      assert(0);
      nXdim = XdimT;
    }
    if (XdimT < 0 && nXdim > abs(XdimT)) {
      assert(0);
      nXdim = abs(XdimT);
    }
    if (nXdim < 0) {
      assert(0);
      nXdim = 0;
    }
    if (YdimT > 0 && nYdim != YdimT) {
      assert(0);
      nYdim = YdimT;
    }
    if (YdimT < 0 && nYdim > abs(YdimT)) {
      assert(0);
      nYdim = abs(YdimT);
    }
    if (nYdim < 0) {
      assert(0);
      nYdim = 0;
    }
  }
};

/// ==================================================================================================
/// Specialization 3 where the number of Y dimensions is 1.
///
template <typename DataT, int32_t XdimT>
class SplineSpec<DataT, XdimT, 1, 3>
  : public SplineSpec<DataT, XdimT, 1, SplineUtil::getSpec(XdimT, 999)>
{
  typedef SplineSpec<DataT, XdimT, 1, SplineUtil::getSpec(XdimT, 999)> TBase;

 public:
  using TBase::TBase; // inherit constructors

  /// Simplified interface for 1D: return the interpolated value
  GPUd() DataT interpolate(const DataT x[]) const
  {
    DataT S = 0.;
    TBase::interpolate(x, &S);
    return S;
  }

  // this parent method should be public anyhow,
  // but the compiler gets confused w/o this extra declaration
  using TBase::interpolate;
};

} // namespace gpu
} // namespace o2

#endif
