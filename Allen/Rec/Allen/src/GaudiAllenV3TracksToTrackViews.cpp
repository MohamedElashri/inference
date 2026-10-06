/*****************************************************************************\
* (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// Gaudi
#include "Gaudi/Accumulators.h"
#include "GaudiAlg/Transformer.h"
#include "GaudiKernel/StdArrayAsProperty.h"

// LHCb
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/TrackEnums.h"
#include "Event/UniqueIDGenerator.h"
#include "Event/StateParameters.h"
#include <Kernel/EventLocalAllocator.h>

// Allen
#include "Logger.h"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiConsolidated.cuh"
#include "ParKalmanFittedTrack.cuh"
#include "States.cuh"
#include "ParticleTypes.cuh"

#include <AIDA/IHistogram1D.h>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <functional>

using InTracks = LHCb::Event::v3::Tracks;
using SL = LHCb::Event::v3::Tracks::StateLocation;
using InTrackType = LHCb::Event::v3::TrackType;
template<typename T>
using allen_t = std::vector<T, LHCb::Allocators::EventLocal<T>>;
using OffsetsType = allen_t<unsigned>;
namespace InTag = LHCb::Event::v3::Tag;

namespace GaudiAllen::Converters::v3 {
  namespace { // TODO: move this definition to common header
    struct beamline_states {
      using type = Allen::Views::Physics::KalmanStates;
      // currently unused, comment to avoid clang warning
      // static constexpr auto keyname = "allen_beamline_states_view";
    };

    struct endvelo_states {
      using type = Allen::Views::Physics::KalmanStates;
      // currently unused, comment to avoid clang warning
      // static constexpr auto keyname = "allen_endvelo_states_view";
    };

    struct rich1_front_states {
      using type = SimpleKalmanState;
      static constexpr auto keyname = "allen_kalman_R1_F_view";
    };

    struct rich1_back_states {
      using type = SimpleKalmanState;
      static constexpr auto keyname = "allen_kalman_R1_B_view";
    };

    struct rich2_front_states {
      using type = SimpleKalmanState;
      static constexpr auto keyname = "allen_kalman_R2_F_view";
    };

    struct rich2_back_states {
      using type = SimpleKalmanState;
      static constexpr auto keyname = "allen_kalman_R2_B_view";
    };

    template<typename AllenInput>
    auto get_output_name()
    {
      if constexpr (
        std::is_same_v<AllenInput, beamline_states> || std::is_same_v<AllenInput, endvelo_states> ||
        std::is_same_v<AllenInput, rich1_front_states> || std::is_same_v<AllenInput, rich1_back_states> ||
        std::is_same_v<AllenInput, rich2_front_states> || std::is_same_v<AllenInput, rich2_back_states>) {
        return AllenInput::keyname;
      }
      else {
        return "allen_tracks_mec";
      }
    }

    template<typename KeyValue, typename... AllenInput>
    auto get_output_names()
    {
      return std::make_tuple(
        KeyValue {"track_offsets", ""}, KeyValue {"num_tracks", ""}, (KeyValue {get_output_name<AllenInput>(), ""})...);
    }
  } // namespace

  /**
   * The first template parameter is assumed to be a view of track types.
   * If present, all other parameters are assumed to be views of states.
   *
   * Number of output containers is deduced from input type
   * - Two track containers for Velo input (forward and backward)
   * - One track container for all other types of input
   */
  template<typename AllenTracks, typename... AllenStates>
  class GaudiAllenV3TracksToTrackViews final
    : public Gaudi::Functional::MultiTransformer<
        std::tuple<OffsetsType, OffsetsType, allen_t<AllenTracks>, allen_t<typename AllenStates::type>...>(
          InTracks const&)> {

  public:
    using OutType = std::tuple<OffsetsType, OffsetsType, allen_t<AllenTracks>, allen_t<typename AllenStates::type>...>;
    using base_class = Gaudi::Functional::MultiTransformer<OutType(InTracks const&)>;
    using KeyValue = typename base_class::KeyValue;

    /// Standard constructor
    GaudiAllenV3TracksToTrackViews(const std::string& name, ISvcLocator* pSvcLocator) :
      base_class(
        name,
        pSvcLocator,
        // Inputs
        {KeyValue {"InputTracks", ""}},
        // Outputs
        get_output_names<KeyValue, AllenTracks, AllenStates...>())
    {}

    /// Algorithm execution
    OutType operator()(InTracks const& tracks) const override
    {
      OutType output;

      // Make offsets and num tracks
      auto& offsets = std::get<0>(output);
      offsets.resize(2);
      offsets[0] = 0;
      offsets[1] = tracks.size();

      auto& num_tracks = std::get<1>(output);
      num_tracks.resize(1);
      num_tracks[0] = tracks.size();

      // Make view
      // TODO

      // Make states, TODO: make this generic ?
      auto& r1_front_states = std::get<3>(output);
      auto& r1_end_states = std::get<4>(output);
      auto& r2_front_states = std::get<5>(output);
      auto& r2_end_states = std::get<6>(output);

      r1_front_states.reserve(tracks.size());
      r1_end_states.reserve(tracks.size());
      r2_front_states.reserve(tracks.size());
      r2_end_states.reserve(tracks.size());

      if constexpr (sizeof...(AllenStates) > 0) {
        for (const auto& track : tracks.scalar()) {
          const auto make_allen_state = [&](const SL location) {
            if (!track.has_state(location)) {
              ++m_missing_rich_states;
              return SimpleKalmanState {};
            }

            const auto& state = track.template field<InTag::States>()[track.state_index(location)];
            const SimpleKalmanState allen_state {
              state.x().cast(),
              state.y().cast(),
              state.z().cast(),
              state.tx().cast(),
              state.ty().cast(),
              state.qOverP().cast()};
            if (
              !std::isfinite(allen_state.x) || !std::isfinite(allen_state.y) || !std::isfinite(allen_state.z) ||
              !std::isfinite(allen_state.tx) || !std::isfinite(allen_state.ty) || !std::isfinite(allen_state.qop)) {
              ++m_nonfinite_rich_states;
              return SimpleKalmanState {};
            }
            ++m_converted_rich_states;
            return allen_state;
          };

          r1_front_states.emplace_back(make_allen_state(SL::BegRich1));
          r1_end_states.emplace_back(make_allen_state(SL::EndRich1));
          r2_front_states.emplace_back(make_allen_state(SL::BegRich2));
          r2_end_states.emplace_back(make_allen_state(SL::EndRich2));
        }
      }
      return output;
    }

  private:
    mutable Gaudi::Accumulators::Counter<> m_converted_rich_states {this, "Converted RICH track states"};
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_missing_rich_states {
      this,
      "Input track is missing a required RICH state; using a default state"};
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_nonfinite_rich_states {
      this,
      "Input track has a non-finite RICH state; using a default state"};
  };

  using GaudiAllenV3TracksToMEBasicParticlesRichStates = GaudiAllenV3TracksToTrackViews<
    Allen::Views::Physics::MultiEventBasicParticles,
    rich1_front_states,
    rich1_back_states,
    rich2_front_states,
    rich2_back_states>;
  DECLARE_COMPONENT_WITH_ID(
    GaudiAllenV3TracksToMEBasicParticlesRichStates,
    "GaudiAllenV3TracksToMEBasicParticlesRichStates")
} // namespace GaudiAllen::Converters::v3
