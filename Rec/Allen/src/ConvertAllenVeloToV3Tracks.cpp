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
 * Convert Allen Velo tracks (raw device buffers) into LHCb::Event::v3::Tracks
 *
 * Multi-event: receives per-slice raw device buffers, copies to host,
 * directly iterates over per-event tracks using flat offset arrays,
 * scatters per-event forward+backward track containers to event stores.
 */

// Gaudi
#include "GaudiAlg/Transformer.h"

// LHCb
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/TrackEnums.h"
#include "Event/UniqueIDGenerator.h"
#include "Event/StateParameters.h"
#include "Kernel/LHCbID.h"

// Allen
#include "VeloEventModel.cuh"
#include "VeloDefinitions.cuh"
#include "States.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"
#include "KalmanVeloStateWithQoP.h"

namespace {
  using VeloStates = LHCb::Event::v3::
    available_states_t<LHCb::Event::v3::TrackType::Velo, LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>;
  using VeloBwdStates = LHCb::Event::v3::
    available_states_t<LHCb::Event::v3::TrackType::VeloBackward, LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>;
} // anonymous namespace

class ConvertAllenVeloToV3Tracks final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<OutTracks, OutTracks>(
      const Allen::device_buffer<char>&,     // velo hits data
      const Allen::device_buffer<unsigned>&, // track offsets  (N+1, cumulative per event)
      const Allen::device_buffer<unsigned>&, // track-hit offsets (cumulative per track)
      const Allen::device_buffer<char>&,     // beamline state data
      const LHCb::UniqueIDGenerator&)> {

public:
  ConvertAllenVeloToV3Tracks(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"dev_velo_hits_data", ""},
       KeyValue {"dev_velo_track_offsets", ""},
       KeyValue {"dev_velo_track_hit_offsets", ""},
       KeyValue {"dev_velo_beamline_state_data", ""},
       KeyValue {"InputUniqueIDGenerator", LHCb::UniqueIDGeneratorLocation::Default}},
      {KeyValue {"OutputTracksForward", ""}, KeyValue {"OutputTracksBackward", ""}})
  {}

  std::tuple<std::vector<OutTracks>, std::vector<OutTracks>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<char>& dev_hits,
    const Allen::device_buffer<unsigned>& dev_track_offsets,
    const Allen::device_buffer<unsigned>& dev_track_hit_offsets,
    const Allen::device_buffer<char>& dev_state_data,
    const LHCb::UniqueIDGenerator& unique_id_gen) const override
  {
    // --- copy all device buffers to host ---
    auto h_hits = dev_hits.to_host();
    auto h_track_offsets = dev_track_offsets.to_host();
    auto h_hit_offsets = dev_track_hit_offsets.to_host();
    auto h_state_data = dev_state_data.to_host();

    const unsigned n_events = h_track_offsets.size() - 1;
    const unsigned n_tracks_total = h_track_offsets[n_events];
    const unsigned n_hits_total = h_hit_offsets[n_tracks_total];

    // Global hit container (handles the half_t → float conversion)
    Velo::ConstClusters all_hits {h_hits.data(), n_hits_total};

    // Output vectors (one entry per event)
    std::vector<OutTracks> out_fwd, out_bwd;
    out_fwd.reserve(n_events);
    out_bwd.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {

      const unsigned t_begin = h_track_offsets[evt];
      const unsigned t_end = h_track_offsets[evt + 1];

      // Build KalmanStates view for this event
      Allen::Views::Physics::KalmanStates kalman_states(h_state_data.data(), h_track_offsets.data(), evt, n_events);

      // Create forward and backward output containers for this event
      auto zn = Zipping::generateZipIdentifier();
      OutTracks fwd(LHCb::Event::v3::TrackType::Velo, unique_id_gen, zn);
      OutTracks bwd(
        LHCb::Event::v3::TrackType::VeloBackward,
        LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter,
        true,
        unique_id_gen,
        zn);

      for (unsigned t = t_begin; t < t_end; ++t) {

        const unsigned hit_begin = h_hit_offsets[t];
        const unsigned hit_end = h_hit_offsets[t + 1];
        const unsigned n_hits = hit_end - hit_begin;

        // ---- extract hit LHCbIDs ----
        std::vector<LHCb::LHCbID> lhcb_ids;
        lhcb_ids.reserve(n_hits);
        for (unsigned h = hit_begin; h < hit_end; ++h) {
          lhcb_ids.emplace_back(all_hits.id(h));
        }

        // First and last hit z (for state extrapolation)
        const float first_meas_z = all_hits.z(hit_end - 1); // last hit in container = first measurement
        const float last_meas_z = all_hits.z(hit_begin);    // first hit = last measurement

        // ---- beamline state ----
        KalmanVeloState beamline = kalman_states.state(t - t_begin);
        const bool backward = beamline.z() > last_meas_z;

        // qop from charge and pT
        // charge sign from row (bits 0-7) of first hit's VP channel ID
        const int firstRow = static_cast<int>(all_hits.id(hit_begin) & 0xFF);
        const float charge = (firstRow % 2 == 0 ? -1.f : 1.f);
        const float tx1 = beamline.tx(), ty1 = beamline.ty();
        const float slope2 = std::max(tx1 * tx1 + ty1 * ty1, 1.e-20f);
        const float qop = charge / (m_ptVelo * std::sqrt(1.f + 1.f / slope2));
        const float qopVar = m_qopvar_rel * qop * qop;

        std::vector<KalmanVeloStateWithQoP> states {{beamline, qop, qopVar}};

        // ---- select forward or backward container ----
        OutTracks& out = backward ? bwd : fwd;
        auto newTrack = out.template emplace_back<SIMDWrapper::InstructionSet::Scalar>();

        // ---- fill LHCbIDs ----
        newTrack.template field<OutTag::VPHits>().resize(n_hits);
        for (unsigned i = 0; i < n_hits; ++i) {
          newTrack.template field<OutTag::VPHits>()[i].template field<OutTag::LHCbID>().set(lhcb_ids[i]);
        }

        // ---- track metadata ----
        newTrack.template field<OutTag::history>().set(LHCb::Event::Enum::Track::History::PrPixel);
        using int_v = decltype(newTrack.template field<OutTag::UniqueID>().get());
        newTrack.template field<OutTag::UniqueID>().set(unique_id_gen.generate<int_v>().value());
        newTrack.template field<OutTag::Chi2>().set(0.f);
        newTrack.template field<OutTag::nDoF>().set(0);

        // ---- states ----
        if (backward)
          update_all_states(newTrack, states, first_meas_z, last_meas_z, VeloBwdStates {});
        else
          update_all_states(newTrack, states, first_meas_z, last_meas_z, VeloStates {});
      }

      out_fwd.emplace_back(std::move(fwd));
      out_bwd.emplace_back(std::move(bwd));
    }

    return {std::move(out_fwd), std::move(out_bwd)};
  }

private:
  Gaudi::Property<float> m_ptVelo {this, "ptVelo", 400 * Allen::Units::MeV, "Default pT for Velo tracks"};
  Gaudi::Property<float> m_qopvar_rel {this, "relQoPVar", 0.1, "Default relative qop variance"};
};

DECLARE_COMPONENT(ConvertAllenVeloToV3Tracks)
