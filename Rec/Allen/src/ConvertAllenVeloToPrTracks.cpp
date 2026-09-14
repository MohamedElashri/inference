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
 * Convert Allen Velo tracks (raw device buffers) into LHCb::Pr::Velo::Tracks
 *
 * Multi-event: receives per-slice raw device buffers, copies to host,
 * directly iterates over per-event tracks using flat offset arrays,
 * scatters per-event forward+backward PrVelo track containers to event stores.
 */

#include "GaudiAlg/Transformer.h"

#include "Event/Track.h"
#include "Event/UniqueIDGenerator.h"
#include "Event/PrVeloTracks.h"
#include "Kernel/LHCbID.h"

#include "VeloEventModel.cuh"
#include "VeloDefinitions.cuh"
#include "States.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"
#include "KalmanVeloStateWithQoP.h"

class ConvertAllenVeloToPrTracks final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<LHCb::Pr::Velo::Tracks, LHCb::Pr::Velo::Tracks>(
      const Allen::device_buffer<char>&,     // velo hits data
      const Allen::device_buffer<unsigned>&, // track offsets  (N+1)
      const Allen::device_buffer<unsigned>&, // track-hit offsets (cumulative)
      const Allen::device_buffer<char>&,     // beamline state data
      const Allen::device_buffer<char>&)> {  // endvelo state data

public:
  ConvertAllenVeloToPrTracks(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"dev_velo_hits_data", ""},
       KeyValue {"dev_velo_track_offsets", ""},
       KeyValue {"dev_velo_track_hit_offsets", ""},
       KeyValue {"dev_velo_beamline_state_data", ""},
       KeyValue {"dev_velo_endvelo_state_data", ""}},
      {KeyValue {"OutputTracksForward", ""}, KeyValue {"OutputTracksBackward", ""}})
  {}

  std::tuple<std::vector<LHCb::Pr::Velo::Tracks>, std::vector<LHCb::Pr::Velo::Tracks>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<char>& dev_hits,
    const Allen::device_buffer<unsigned>& dev_track_offsets,
    const Allen::device_buffer<unsigned>& dev_track_hit_offsets,
    const Allen::device_buffer<char>& dev_beamline_state_data,
    const Allen::device_buffer<char>& dev_endvelo_state_data) const override
  {
    auto h_hits = dev_hits.to_host();
    auto h_track_offsets = dev_track_offsets.to_host();
    auto h_hit_offsets = dev_track_hit_offsets.to_host();
    auto h_beamline_data = dev_beamline_state_data.to_host();
    auto h_endvelo_data = dev_endvelo_state_data.to_host();

    const unsigned n_events = h_track_offsets.size() - 1;
    const unsigned n_tracks_total = h_track_offsets[n_events];
    const unsigned n_hits_total = h_hit_offsets[n_tracks_total];

    Velo::ConstClusters all_hits {h_hits.data(), n_hits_total};

    std::vector<LHCb::Pr::Velo::Tracks> out_fwd, out_bwd;
    out_fwd.reserve(n_events);
    out_bwd.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {

      const unsigned t_begin = h_track_offsets[evt];
      const unsigned t_end = h_track_offsets[evt + 1];

      // Per-event state views
      Allen::Views::Physics::KalmanStates beamline_states(
        h_beamline_data.data(), h_track_offsets.data(), evt, n_events);
      Allen::Views::Physics::KalmanStates endvelo_states(h_endvelo_data.data(), h_track_offsets.data(), evt, n_events);

      auto zn = Zipping::generateZipIdentifier();
      LHCb::Pr::Velo::Tracks fwd {false, zn};
      LHCb::Pr::Velo::Tracks bwd {true, zn};

      for (unsigned t = t_begin; t < t_end; ++t) {

        const unsigned hit_begin = h_hit_offsets[t];
        const unsigned hit_end = h_hit_offsets[t + 1];
        const unsigned n_hits = hit_end - hit_begin;

        // ---- LHCbIDs ----
        std::vector<LHCb::LHCbID> lhcb_ids;
        lhcb_ids.reserve(n_hits);
        for (unsigned h = hit_begin; h < hit_end; ++h)
          lhcb_ids.emplace_back(all_hits.id(h));

        const float last_meas_z = all_hits.z(hit_begin);

        // ---- states ----
        const unsigned local_t = t - t_begin;
        KalmanVeloState beamline = beamline_states.state(local_t);
        KalmanVeloState endvelo = endvelo_states.state(local_t);
        const bool backward = beamline.z() > last_meas_z;

        // qop from charge and pT
        const int firstRow = static_cast<int>(all_hits.id(hit_begin) & 0xFF);
        const float charge = (firstRow % 2 == 0 ? -1.f : 1.f);
        const float tx1 = beamline.tx(), ty1 = beamline.ty();
        const float slope2 = std::max(tx1 * tx1 + ty1 * ty1, 1.e-20f);
        const float qop = charge / (m_ptVelo * std::sqrt(1.f + 1.f / slope2));
        const float qopVar = m_qopvar_rel * qop * qop;

        std::vector<KalmanVeloStateWithQoP> states {{beamline, qop, qopVar}, {endvelo, qop, qopVar}};

        // ---- create track ----
        LHCb::Pr::Velo::Tracks& out = backward ? bwd : fwd;
        auto newTrack = out.template emplace_back<SIMDWrapper::InstructionSet::Scalar>();

        // hits
        newTrack.template field<LHCb::Pr::Velo::Tag::Hits>().resize(n_hits);
        for (unsigned i = 0; i < n_hits; ++i)
          newTrack.template field<LHCb::Pr::Velo::Tag::Hits>()[i].template field<LHCb::Pr::Velo::Tag::LHCbID>().set(
            lhcb_ids[i]);

        // states: ClosestToBeam and EndVelo
        newTrack.template field<LHCb::Pr::Velo::Tag::States>(0).setPosition(
          states[0].x(), states[0].y(), states[0].z());
        newTrack.template field<LHCb::Pr::Velo::Tag::States>(0).setDirection(states[0].tx(), states[0].ty());
        newTrack.setStateCovXY(
          SL::ClosestToBeam,
          LHCb::LinAlg::Vec<SIMDWrapper::scalar::float_v, 3> {states[0].c00(), states[0].c20(), states[0].c22()},
          LHCb::LinAlg::Vec<SIMDWrapper::scalar::float_v, 3> {states[0].c11(), states[0].c31(), states[0].c33()});

        newTrack.template field<LHCb::Pr::Velo::Tag::States>(1).setPosition(
          states[1].x(), states[1].y(), states[1].z());
        newTrack.template field<LHCb::Pr::Velo::Tag::States>(1).setDirection(states[1].tx(), states[1].ty());
        newTrack.setStateCovXY(
          SL::EndVelo,
          LHCb::LinAlg::Vec<SIMDWrapper::scalar::float_v, 3> {states[1].c00(), states[1].c20(), states[1].c22()},
          LHCb::LinAlg::Vec<SIMDWrapper::scalar::float_v, 3> {states[1].c11(), states[1].c31(), states[1].c33()});
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

DECLARE_COMPONENT(ConvertAllenVeloToPrTracks)
