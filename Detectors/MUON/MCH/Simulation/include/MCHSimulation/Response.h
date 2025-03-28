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

#ifndef O2_MCH_SIMULATION_RESPONSE_H_
#define O2_MCH_SIMULATION_RESPONSE_H_

#include "DataFormatsMCH/Digit.h"
#include "MCHBase/MathiesonOriginal.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"

namespace o2
{
namespace mch
{

enum class Station {
  Type1,
  Type2345
};

class Response
{
 public:
  Response(Station station);
  ~Response() = default;
  float getChargeSpread() const { return mChargeSpread; }
  float getPitch() const { return mPitch; }
  float getSigmaIntegration() const { return mSigmaIntegration; }
  bool isAboveThreshold(float charge) const { return charge > mChargeThreshold; }
  bool isAngleEffect() const { return mAngleEffect; }
  bool isMagnetEffect() const { return mMagnetEffect; }

  /** Converts energy deposition into a charge.
   *
   * @param edepos deposited energy from Geant (in GeV)
   * @returns an equivalent charge (roughly in ADC units)
   *
   */
  float etocharge(float edepos) const;

  /** Compute the charge fraction in a rectangle area for a unit charge
   * occuring at position (0,0)
   *
   * @param xmin, xmax, ymin, ymax coordinates (in cm) defining the area
   */
  float chargePadfraction(float xmin, float xmax, float ymin, float ymax) const
  {
    return mMathieson.integrate(xmin, ymin, xmax, ymax);
  }

  /// return wire coordinate closest to x
  float getAnod(float x) const;

  /// return a randomized charge correlation between cathodes
  float chargeCorr() const;

  /// compute the number of samples corresponding to the charge in ADC units
  uint32_t nSamples(float charge) const;

  /// compute deteriation of y-resolution due to track inclination and B-field
  float inclandbfield(float thetawire, float betagamma, float bx) const;

 private:
  Station mStation{};             ///< Station type
  MathiesonOriginal mMathieson{}; ///< Mathieson function
  float mPitch = 0.f;             ///< anode-cathode pitch (cm)
  float mChargeSlope = 0.f;       ///< charge slope used in E to charge conversion
  float mChargeSpread = 0.f;      ///< width of the charge distribution (cm)
  float mSigmaIntegration = 0.f;  ///< number of sigmas used for charge distribution
  float mChargeCorr = 0.f;        ///< amplitude of charge correlation between cathodes
  float mChargeThreshold = 0.f;   ///< minimum fraction of charge considered
  bool mAngleEffect = true;       ///< switch for angle effect influencing charge deposition
  bool mMagnetEffect = true;      ///< switch for magnetic field influencing charge deposition

  /// Ratio of particle mean eloss with respect MIP's Khalil Boudjemline, sep 2003, PhD.Thesis and Particle Data Book
  float eLossRatio(float logbetagamma) const;
  /// ToDo: check Aliroot formula vs PDG, if really log_10 and not ln or bug in Aliroot

  /// Angle effect in tracking chambers at theta =10 degres as a function of ElossRatio (Khalil BOUDJEMLINE sep 2003 Ph.D Thesis) (in micrometers)
  float angleEffect10(float elossratio) const;

  /// Angle effect: Normalisation form theta=10 degres to theta between 0 and 10 (Khalil BOUDJEMLINE sep 2003 Ph.D Thesis)
  /// Angle with respect to the wires assuming that chambers are perpendicular to the z axis.
  float angleEffectNorma(float angle) const;

  /// Magnetic field effect: Normalisation form theta=16 degres (eq. 10 degrees B=0) to theta between -20 and 20 (Lamia Benhabib jun 2006 )
  /// Angle with respect to the wires assuming that chambers are perpendicular to the z axis.
  float magAngleEffectNorma(float angle, float bfield) const;
};
} // namespace mch
} // namespace o2
#endif
