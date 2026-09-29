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
/**
 * Convert Allen UT (VeloUT) tracks (raw device buffers) into LHCb::Event::v3::Tracks
 *
 * Multi-event: receives per-slice raw device buffers, copies to host,
 * directly iterates over per-event tracks using flat offset arrays,
 * scatters per-event Upstream track containers to event stores.
 */

#include "GaudiAlg/Transformer.h"

#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/TrackEnums.h"
#include "Event/UniqueIDGenerator.h"
#include "Event/StateParameters.h"
#include "Kernel/LHCbID.h"

#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "States.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"
#include "KalmanVeloStateWithQoP.h"

#include <algorithm>

namespace {
  // State locations for Upstream tracks
  using UpstreamStates = LHCb::Event::v3::
    available_states_t<LHCb::Event::v3::TrackType::Upstream, LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>;
} // anonymous namespace

// ==================================================================

class ConvertAllenUTToV3Tracks final : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<OutTracks>(
                                         const Allen::device_buffer<char>&,     // UT hits data (float array)
                                         const Allen::device_buffer<unsigned>&, // UT track offsets (N+1)
                                         const Allen::device_buffer<unsigned>&, // UT track-hit offsets (cumulative)
                                         const Allen::device_buffer<unsigned>&, // UT→Velo track index indirection
                                         const Allen::device_buffer<char>&,     // Velo hits data (for Velo segments)
                                         const Allen::device_buffer<unsigned>&, // Velo track offsets (N+1)
                                         const Allen::device_buffer<unsigned>&, // Velo track-hit offsets (cumulative)
                                         const Allen::device_buffer<float>&, // UT track params (qop, x, z, tx floats)
                                         const Allen::device_buffer<char>&,  // beamline state data
                                         const Allen::device_buffer<char>&,  // endvelo state data
                                         const LHCb::UniqueIDGenerator&)> {

public:
  ConvertAllenUTToV3Tracks(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"dev_ut_hits_data", ""},
       KeyValue {"dev_ut_track_offsets", ""},
       KeyValue {"dev_ut_track_hit_offsets", ""},
       KeyValue {"dev_ut_track_velo_indices", ""},
       KeyValue {"dev_velo_hits_data", ""},
       KeyValue {"dev_velo_track_offsets", ""},
       KeyValue {"dev_velo_track_hit_offsets", ""},
       KeyValue {"dev_ut_track_params_data", ""},
       KeyValue {"dev_ut_beamline_state_data", ""},
       KeyValue {"dev_ut_endvelo_state_data", ""},
       KeyValue {"InputUniqueIDGenerator", LHCb::UniqueIDGeneratorLocation::Default}},
      {KeyValue {"OutputTracks", ""}})
  {}

  std::tuple<std::vector<OutTracks>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<char>& dev_ut_hits,
    const Allen::device_buffer<unsigned>& dev_ut_track_offsets,
    const Allen::device_buffer<unsigned>& dev_ut_hit_offsets,
    const Allen::device_buffer<unsigned>& dev_velo_indices,
    const Allen::device_buffer<char>& dev_velo_hits,
    const Allen::device_buffer<unsigned>& dev_velo_track_offsets,
    const Allen::device_buffer<unsigned>& dev_velo_hit_offsets,
    const Allen::device_buffer<float>& dev_track_params,
    const Allen::device_buffer<char>& dev_beamline_state_data,
    const Allen::device_buffer<char>& dev_endvelo_state_data,
    const LHCb::UniqueIDGenerator& unique_id_gen) const override
  {
    auto h_ut_hits = dev_ut_hits.to_host();
    auto h_ut_track_offsets = dev_ut_track_offsets.to_host();
    auto h_ut_hit_offsets = dev_ut_hit_offsets.to_host();
    auto h_velo_indices = dev_velo_indices.to_host();
    auto h_velo_hits = dev_velo_hits.to_host();
    auto h_velo_offsets = dev_velo_track_offsets.to_host();
    auto h_velo_hit_offsets = dev_velo_hit_offsets.to_host();
    auto h_track_params = dev_track_params.to_host();
    auto h_beamline_data = dev_beamline_state_data.to_host();
    auto h_endvelo_data = dev_endvelo_state_data.to_host();

    const unsigned n_events = h_ut_track_offsets.size() - 1;
    const unsigned n_ut_tracks_total = h_ut_track_offsets[n_events];
    const unsigned n_ut_hits_total = h_ut_hit_offsets[n_ut_tracks_total];

    // UT hits: 6 float sections + 1 uint32 section, all of length n_ut_hits_total
    const float* ut_float_base = reinterpret_cast<const float*>(h_ut_hits.data());
    const uint32_t* ut_id_base = reinterpret_cast<const uint32_t*>(ut_float_base + 6 * n_ut_hits_total);

    // Velo hits (indexed globally by Velo track index via dev_velo_indices)
    const unsigned n_velo_hits_total =
      h_velo_hit_offsets.size() > 0 ? h_velo_hit_offsets[h_velo_hit_offsets.size() - 1] : 0;
    Velo::Consolidated::ConstHits velo_hits {h_velo_hits.data(), 0, n_velo_hits_total};

    // Track params: 4 floats per UT track (qop, x, z, tx)
    const float* params = h_track_params.data();

    std::vector<OutTracks> all_tracks;
    all_tracks.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {

      const unsigned t_begin = h_ut_track_offsets[evt];
      const unsigned t_end = h_ut_track_offsets[evt + 1];

      // State views indexed by global Velo track index (via velo_indices)
      Allen::Views::Physics::KalmanStates beamline_states(h_beamline_data.data(), h_velo_offsets.data(), evt, n_events);
      Allen::Views::Physics::KalmanStates endvelo_states(h_endvelo_data.data(), h_velo_offsets.data(), evt, n_events);

      auto zn = Zipping::generateZipIdentifier();
      OutTracks out(LHCb::Event::v3::TrackType::Upstream, unique_id_gen, zn);

      for (unsigned t = t_begin; t < t_end; ++t) {

        const unsigned velo_idx = h_velo_indices[t];

        // ---- UT hits ----
        const unsigned ut_hit_begin = h_ut_hit_offsets[t];
        const unsigned ut_hit_end = h_ut_hit_offsets[t + 1];
        const unsigned n_ut_hits = ut_hit_end - ut_hit_begin;

        // ---- Velo hits (indexed by Velo track index) ----
        const unsigned velo_hit_begin = h_velo_hit_offsets[h_velo_offsets[evt] + velo_idx];
        const unsigned velo_hit_end = h_velo_hit_offsets[h_velo_offsets[evt] + velo_idx + 1];
        const unsigned n_velo_hits = velo_hit_end - velo_hit_begin;

        const unsigned n_total_hits = n_ut_hits + n_velo_hits;

        // ---- extract hit LHCbIDs (UT first, then Velo) ----
        std::vector<LHCb::LHCbID> lhcb_ids;
        lhcb_ids.reserve(n_total_hits);

        for (unsigned h = velo_hit_begin; h < velo_hit_end; ++h) {
          [[maybe_unused]] const auto id = lhcb_ids.emplace_back(velo_hits.id(h));
          assert(id.isVP());
        }

        for (unsigned h = ut_hit_begin; h < ut_hit_end; ++h) {
          [[maybe_unused]] const auto id = lhcb_ids.emplace_back(ut_id_base[h]);
          assert(id.isUT());
        }

        // ---- z of first and last measurement ----
        // First measurement: last Velo hit (closest to beam)
        // Last measurement: first UT hit (furthest from beam)
        const float first_meas_z =
          (n_velo_hits > 0) ? velo_hits.z(velo_hit_end - 1) : ut_float_base[2 * n_ut_hits_total + ut_hit_begin];
        const float last_meas_z = ut_float_base[2 * n_ut_hits_total + ut_hit_end - 1];

        // ---- states (indexed by global Velo track index) ----
        KalmanVeloState beamline = beamline_states.state(velo_idx);
        KalmanVeloState endvelo = endvelo_states.state(velo_idx);

        // qop from track params
        const float qop = params[4 * t]; // qop
        const float qopVar = m_qopvar_rel * qop * qop;

        std::vector<KalmanVeloStateWithQoP> states {{beamline, qop, qopVar}, {endvelo, qop, qopVar}};

        // ---- create track ----
        auto newTrack = out.template emplace_back<SIMDWrapper::InstructionSet::Scalar>();

        // Hits: UT then Velo (matches LHCbID order above)
        newTrack.template field<OutTag::UTHits>().resize(n_ut_hits);
        for (unsigned i = 0; i < n_ut_hits; ++i)
          newTrack.template field<OutTag::UTHits>()[i].template field<OutTag::LHCbID>().set(lhcb_ids[i]);

        newTrack.template field<OutTag::VPHits>().resize(n_velo_hits);
        for (unsigned i = 0; i < n_velo_hits; ++i)
          newTrack.template field<OutTag::VPHits>()[i].template field<OutTag::LHCbID>().set(lhcb_ids[n_ut_hits + i]);

        // metadata
        newTrack.template field<OutTag::history>().set(LHCb::Event::Enum::Track::History::PrVeloUT);
        using int_v = decltype(newTrack.template field<OutTag::UniqueID>().get());
        newTrack.template field<OutTag::UniqueID>().set(unique_id_gen.generate<int_v>().value());
        newTrack.template field<OutTag::Chi2>().set(0.f);
        newTrack.template field<OutTag::nDoF>().set(0);

        // states
        update_all_states(newTrack, states, first_meas_z, last_meas_z, UpstreamStates {});
      }

      all_tracks.emplace_back(std::move(out));
    }

    return std::make_tuple(std::move(all_tracks));
  }

private:
  Gaudi::Property<float> m_qopvar_rel {this, "relQoPVar", 0.1, "Default relative qop variance"};
};

DECLARE_COMPONENT(ConvertAllenUTToV3Tracks)
