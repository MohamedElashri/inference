/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "Constants.cuh"
#include "MVAModelsManager.h"
#include "InputTools.h"

#include "Beamline.h"
#include "EcalGeometry.h"
#include "MagneticField.h"
#include "MagneticFieldPolarity.h"
#include "MuonGeometry.h"
#include "MuonLookupTables.h"
#include "RichGeometry.h"
#include "RichCableMapping.h"
#include "RichPDMDBMapping.h"
#include "SciFiGeometry.h"
#include "UTBoards.h"
#include "UTGeometry.h"
#include "UTLookupTables.h"
#include "VPGeometry.h"

namespace Allen::Conditions {
  struct ConstantsCondition {
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-constants";
#else
    inline static std::string const DefaultLocation = "AllenConditions-constants";
#endif

    template<typename PARENT, typename... DEPENDENCIES>
    static auto addConditionDerivationImpl(PARENT* parent)
    {
      auto updater = parent->template service<AllenUpdater>("AllenUpdater", true);
      if (!updater) {
        parent->error() << "Failed to retrieve AllenUpdater" << endmsg;
#ifdef USE_DD4HEP
        return false;
#else
        return static_cast<std::size_t>(-1); // NoDerivation
#endif
      }

      // Register the condition derivations:
      (DEPENDENCIES::addConditionDerivation(parent), ...);

      // And the constants:
      return parent->addConditionDerivation(
        {DEPENDENCIES::DefaultLocation...},
        ConstantsCondition::DefaultLocation,
        [updater](DEPENDENCIES const&... dependencies) {
          auto& constants = updater->getConstants();

          // TODO: move these to their own conditions, and proper memory management..
          std::string geometry_path = updater->getParamDir() + "/data";

          std::vector<float> muon_field_of_interest_params;
          read_muon_field_of_interest(
            muon_field_of_interest_params, geometry_path + "/allen_muon_field_of_interest_params.bin");
          constants.reserve_and_initialize(muon_field_of_interest_params, geometry_path);

          ParKalmanReader parKalmanFilter_reader {geometry_path + "/ParametrizedKalmanFit/25v1/params.json"};
          constants.initialize_kalman_pars_constants(
            parKalmanFilter_reader.VP_pars(),
            parKalmanFilter_reader.VPUT_pars(),
            parKalmanFilter_reader.UT_pars(),
            parKalmanFilter_reader.T_pars(),
            parKalmanFilter_reader.UTTF_pars(),
            parKalmanFilter_reader.TFT_pars(),
            parKalmanFilter_reader.UT_layer(),
            parKalmanFilter_reader.T_layer(),
            parKalmanFilter_reader.UTT_META());

          (dependencies.update_constants(constants), ...);
          (updater->dump(dependencies), ...);

          return constants;
        });
    }

    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return addConditionDerivationImpl<
        PARENT,
        Allen::Conditions::Beamline,
        Allen::Conditions::EcalGeometry,
        Allen::Conditions::MagneticField,
        Allen::Conditions::MagneticFieldPolarity,
        Allen::Conditions::MuonGeometry,
        Allen::Conditions::MuonLookupTables,
        Allen::Conditions::Rich1Geometry,
        Allen::Conditions::Rich2Geometry,
        Allen::Conditions::RichCableMapping,
        Allen::Conditions::RichPDMDBMapping,
        Allen::Conditions::SciFiGeometry,
        Allen::Conditions::UTBoards,
        Allen::Conditions::UTGeometry,
        Allen::Conditions::UTLookupTables,
        Allen::Conditions::VPGeometry>(parent);
    }
  };
} // namespace Allen::Conditions
