/*****************************************************************************\
 * (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration          *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
 *                                                                             *
 * In applying this licence CERN does not waive the privileges and immunities  *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <array>
#include <cstdint>

namespace Allen::Rich {
  /// Number of RICH detectors
  inline constexpr std::uint16_t NRiches = 2u;

  /// Number of photon-detector panels per RICH detector
  inline constexpr std::uint16_t NPDPanelsPerRICH = 2u;

  /// Total number of RICH photon-detector panels
  inline constexpr std::uint16_t NTotalPDPanels = NRiches * NPDPanelsPerRICH;

  /// Fixed-size storage with one entry per RICH detector
  template<typename T>
  using DetectorArray = std::array<T, NRiches>;

  /// Fixed-size storage with one entry per photon-detector panel
  template<typename T>
  using PanelArray = std::array<T, NPDPanelsPerRICH>;

  /// Number of RICH particle types, including the below-threshold hypothesis
  inline constexpr std::uint16_t NParticleTypes = 7u;

  /// Number of physical RICH particle types
  inline constexpr std::uint16_t NRealParticleTypes = NParticleTypes - 1u;

  /// RICH particle ID mass hypotheses
  enum ParticleIDType : std::int8_t {
    Unknown = -1,  ///< Unknown particle type
    Electron,      ///< Represents e+ or e-
    Muon,          ///< Represents mu+ or mu-
    Pion,          ///< Represents pi+ or pi-
    Kaon,          ///< Represents K+ or K-
    Proton,        ///< Represents Pr+ or Pr-
    Deuteron,      ///< Represents d+ or d-
    BelowThreshold ///< Particle type is below threshold
  };

  /// Fixed-size storage with one entry per RICH particle hypothesis
  template<typename T>
  using ParticleArray = std::array<T, NParticleTypes>;

  /// Container of particle ID types
  using Particles = ParticleArray<ParticleIDType>;

  /// Fixed-size storage with one entry per physical RICH particle hypothesis
  template<typename T>
  using RealParticleArray = std::array<T, NRealParticleTypes>;

  /// Container of physical particle ID types
  using RealParticles = RealParticleArray<ParticleIDType>;
} // namespace Allen::Rich

namespace Allen::Rich::Detector {
  enum DetectorType : std::int8_t {
    InvalidDetector = -1, //< Unspecified Detector
    Rich1 = 0,            //< RICH1 detector
    Rich2 = 1,            //< RICH2 detector
    Rich = 1              //< Single RICH detector
  };

  /// Detector side
  enum Side : std::int8_t {
    InvalidSide = -1, //< Invalid side
    // RICH1
    top = 0,    //< Upper panel in RICH1
    bottom = 1, //< Lower panel in RICH1
    // RICH2
    left = 0,  //< Left panel in RICH2
    right = 1, //< Right panel in RICH2
    aside = 0, //< A-Side panel in RICH2
    cside = 1, //< C-Side panel in RICH2
    // Generic
    firstSide = 0, //< Upper panel in RICH1 or Left panel in RICH2
    secondSide = 1 //< Lower panel in RICH1 or Right panel in RICH2
  };

  enum RadiatorType : std::int8_t {
    InvalidRadiator = -1, ///< Unspecified radiator type
    Rich1Gas = 1,         ///< Gaseous RICH1 radiator
    Rich2Gas = 2,         ///< Gaseous RICH2 radiator
    C4F10 = 1,            ///< Gaseous RICH1 radiator (to be removed)
    CF4 = 2,              ///< Gaseous RICH2 radiator (to be removed)
    // background types
    GasQuartzWin = 3,  ///< Quartz windows to the gas radiator volumes
    HPDQuartzWin = 4,  ///< HPD Quartz windows
    Nitrogen = 5,      ///< Nitrogen volume
    AerogelFilter = 6, ///< Aerogel filter material //TODO: I suppose this material should go too?
    CO2 = 7,           ///< Carbon dioxide
    PMTQuartzWin = 8   ///< MAPMT Quartz windows
  };

  /// Container of detector types
  using Detectors = Allen::Rich::DetectorArray<DetectorType>;

  /// Container of panel sides
  using Sides = Allen::Rich::PanelArray<Side>;

  /// Container of active radiator types
  using Radiators = Allen::Rich::DetectorArray<RadiatorType>;
} // namespace Allen::Rich::Detector
