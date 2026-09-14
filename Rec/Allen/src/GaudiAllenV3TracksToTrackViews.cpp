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

// LHCb
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/TrackEnums.h"
#include "Event/StateParameters.h"

// Allen
#include "States.cuh"
#include "ParticleTypes.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

#include <algorithm>
#include <cmath>

using InTracks = LHCb::Event::v3::Tracks;
using SL = InTracks::StateLocation;
namespace InTag = LHCb::Event::v3::Tag;

namespace GaudiAllen::Converters::v3 {

  /**
   * Gather per-event v3 Long tracks from multiple event stores,
   * extract Rich SimpleKalmanState arrays (R1F, R1B, R2F, R2B),
   * and produce concatenated device buffers for the multi-event slice.
   */
  class GaudiAllenV3TracksToMEBasicParticlesRichStates final
    : public LHCb::Algorithm::GatherEvent::MultiTransformer<std::tuple<
        Allen::device_buffer<unsigned>,
        Allen::host_buffer<unsigned>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>>(const InTracks&)> {

  public:
    GaudiAllenV3TracksToMEBasicParticlesRichStates(const std::string& name, ISvcLocator* pSvcLocator) :
      MultiTransformer(
        name,
        pSvcLocator,
        {KeyValue {"InputTracks", ""}},
        {KeyValue {"dev_offsets_tracks", ""},
         KeyValue {"host_number_of_tracks", ""},
         KeyValue {"dev_kalman_R1_F_view", ""},
         KeyValue {"dev_kalman_R1_B_view", ""},
         KeyValue {"dev_kalman_R2_F_view", ""},
         KeyValue {"dev_kalman_R2_B_view", ""}})
    {}

    std::tuple<
      Allen::device_buffer<unsigned>,
      Allen::host_buffer<unsigned>,
      Allen::device_buffer<SimpleKalmanState>,
      Allen::device_buffer<SimpleKalmanState>,
      Allen::device_buffer<SimpleKalmanState>,
      Allen::device_buffer<SimpleKalmanState>>
    operator()(const EventContext& ctx, const std::span<const InTracks*>& tracks_span) const override
    {
      const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
      const unsigned n_events = tracks_span.size();

      // Per-event track offsets
      Allen::host_buffer<unsigned> h_offsets {n_events + 1, ctxExt->memory_managers};
      h_offsets[0] = 0;
      for (unsigned e = 0; e < n_events; ++e)
        h_offsets[e + 1] = h_offsets[e] + tracks_span[e]->size();
      const unsigned n_total = h_offsets[n_events];

      Allen::host_buffer<unsigned> h_n_tracks {1, ctxExt->memory_managers};
      h_n_tracks[0] = n_total;

      // Allocate host buffers for states
      Allen::host_buffer<SimpleKalmanState> h_r1f {n_total, ctxExt->memory_managers};
      Allen::host_buffer<SimpleKalmanState> h_r1b {n_total, ctxExt->memory_managers};
      Allen::host_buffer<SimpleKalmanState> h_r2f {n_total, ctxExt->memory_managers};
      Allen::host_buffer<SimpleKalmanState> h_r2b {n_total, ctxExt->memory_managers};

      // Fill states on host with validity checks
      unsigned offset = 0;
      for (unsigned e = 0; e < n_events; ++e) {
        for (const auto& track : tracks_span[e]->scalar()) {

          auto make_allen_state = [&](SL location) -> SimpleKalmanState {
            if (!track.has_state(location)) {
              ++m_missing_rich_states;
              return {};
            }
            const auto& state = track.template field<InTag::States>()[track.state_index(location)];
            SimpleKalmanState s {
              state.x().cast(),
              state.y().cast(),
              state.z().cast(),
              state.tx().cast(),
              state.ty().cast(),
              state.qOverP().cast()};
            if (
              !std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z) || !std::isfinite(s.tx) ||
              !std::isfinite(s.ty) || !std::isfinite(s.qop)) {
              ++m_nonfinite_rich_states;
              return {};
            }
            ++m_converted_rich_states;
            return s;
          };

          h_r1f[offset] = make_allen_state(SL::BegRich1);
          h_r1b[offset] = make_allen_state(SL::EndRich1);
          h_r2f[offset] = make_allen_state(SL::BegRich2);
          h_r2b[offset] = make_allen_state(SL::EndRich2);
          ++offset;
        }
      }

      // Copy to device
      Allen::device_buffer<unsigned> d_offsets {ctxExt->memory_managers};
      Allen::device_buffer<SimpleKalmanState> d_r1f {ctxExt->memory_managers};
      Allen::device_buffer<SimpleKalmanState> d_r1b {ctxExt->memory_managers};
      Allen::device_buffer<SimpleKalmanState> d_r2f {ctxExt->memory_managers};
      Allen::device_buffer<SimpleKalmanState> d_r2b {ctxExt->memory_managers};
      h_offsets.copy_to(d_offsets);
      h_r1f.copy_to(d_r1f);
      h_r1b.copy_to(d_r1b);
      h_r2f.copy_to(d_r2f);
      h_r2b.copy_to(d_r2b);

      return {
        std::move(d_offsets),
        std::move(h_n_tracks),
        std::move(d_r1f),
        std::move(d_r1b),
        std::move(d_r2f),
        std::move(d_r2b)};
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

  DECLARE_COMPONENT_WITH_ID(
    GaudiAllenV3TracksToMEBasicParticlesRichStates,
    "GaudiAllenV3TracksToMEBasicParticlesRichStates")
} // namespace GaudiAllen::Converters::v3
