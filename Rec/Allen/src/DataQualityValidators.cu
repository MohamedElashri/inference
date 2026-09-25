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
// ----------------------------------------------------------------------------
// Offline Data Quality Validator (ODQV), migrated from the Allen host/device
// validators to Gaudi algorithms so that the tuples are written through the
// standard Gaudi tuple service.
//
//   ConvertAllenLongTracksDQ      : MultiEventBasicParticles -> per-event POD
//   DataQualityValidatorVelo      : Velo track states
//   DataQualityValidatorPV        : primary vertices
//   DataQualityValidatorOccupancy : event occupancy
//   DataQualityValidatorLong      : long-track quantities
// ----------------------------------------------------------------------------
// Gaudi
#include "GaudiAlg/GaudiTupleAlg.h"
#include "LHCbAlgs/Consumer.h"

// Allen
#include "AllenBuffer.cuh"
#include "EventTransformer.h"
#include "KinUtils.cuh"
#include "MuonDefinitions.cuh"
#include "ParticleTypes.cuh"
#include "PV_Definitions.cuh"
#include "SciFiDefinitions.cuh"
#include "TargetFunction.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"

// Standard
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {
  template<typename Fn>
  auto global_function(const Fn& fn)
  {
    return GlobalFunction<Fn> {fn};
  }

  struct DQLongTrackInfo {
    float ip_x = 0.f, ip_y = 0.f, ip_chi2 = 0.f, chi2 = 0.f;
    int is_muon = 0, is_electron = 0;
    float cov[4][4] = {{0}};
    float tx = 0.f, ty = 0.f, qop = 0.f, eta = 0.f, pt = 0.f;
  };

  using MEParticles = Allen::Views::Physics::MultiEventBasicParticles;

  __global__ void extract_long_track_offsets_k(const MEParticles* particles, unsigned* offsets)
  {
    for (unsigned i = threadIdx.x; i < particles->number_of_events(); i += blockDim.x) {
      offsets[i] = particles->container(i).offset();
      if (i == particles->number_of_events() - 1) {
        offsets[i + 1] = particles->container(i).offset() + particles->container(i).size();
      }
    }
  }

  __global__ void extract_long_track_dq_k(const MEParticles* particles, DQLongTrackInfo* out)
  {
    const unsigned event_number = blockIdx.x;
    const auto view = particles->container(event_number);
    const unsigned offset = view.offset();
    const unsigned number_of_tracks = view.size();

    for (unsigned i = threadIdx.x; i < number_of_tracks; i += blockDim.x) {
      const auto track = view.particle(i);

      DQLongTrackInfo info;
      info.ip_chi2 = track.ip_chi2();
      info.ip_x = track.ip_x();
      info.ip_y = track.ip_y();
      info.chi2 = track.chi2();
      info.is_muon = track.is_muon();
      info.is_electron = track.is_electron();

      const auto state = track.state();
      info.cov[0][0] = state.c00();
      info.cov[1][1] = state.c11();
      info.cov[2][0] = state.c20();
      info.cov[2][2] = state.c22();
      info.cov[3][1] = state.c31();
      info.cov[3][3] = state.c33();
      info.tx = state.tx();
      info.ty = state.ty();
      info.qop = state.qop();
      info.eta = state.eta();
      info.pt = state.pt();

      out[offset + i] = info;
    }
  }

  template<typename Tuple>
  auto make_column(Tuple& tuple)
  {
    return [&tuple](const std::string& name, const auto& value) { (void) tuple->column(name, value); };
  }
} // namespace

// ================================================================
//  Long tracks: multi-event view -> per-event POD info
// ================================================================

class ConvertAllenLongTracksDQ final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<std::vector<DQLongTrackInfo>>(
      const Allen::device_buffer<MEParticles>&)> {
public:
  using base_class = LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<std::vector<DQLongTrackInfo>>(
    const Allen::device_buffer<MEParticles>&)>;
  using KeyValue = typename base_class::KeyValue;

  ConvertAllenLongTracksDQ(const std::string& name, ISvcLocator* pSvcLocator) :
    base_class(name, pSvcLocator, {KeyValue {"dev_particle_container", ""}}, {KeyValue {"DQLongTrackInfo", ""}})
  {}

  std::tuple<std::vector<std::vector<DQLongTrackInfo>>> operator()(
    const EventContext& ctx,
    const Allen::device_buffer<MEParticles>& dev_particles) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
    const Allen::Context& context = ctxExt->allen_context;
    const unsigned n_events = ctxExt->number_of_events;

    Allen::device_buffer<unsigned> dev_offsets {n_events + 1, ctxExt->memory_managers};
    global_function(extract_long_track_offsets_k)(dim3(1), dim3(1), context)(dev_particles.data(), dev_offsets.data());
    auto h_offsets = dev_offsets.to_host();
    const unsigned n_total = h_offsets[n_events];

    Allen::device_buffer<DQLongTrackInfo> dev_info {n_total, ctxExt->memory_managers};
    if (n_total != 0) {
      global_function(extract_long_track_dq_k)(dim3(n_events), dim3(256), context)(
        dev_particles.data(), dev_info.data());
    }
    auto h_info = dev_info.to_host();

    std::vector<std::vector<DQLongTrackInfo>> all;
    all.reserve(n_events);
    for (unsigned event = 0; event < n_events; ++event) {
      all.emplace_back(h_info.begin() + h_offsets[event], h_info.begin() + h_offsets[event + 1]);
    }

    return std::make_tuple(std::move(all));
  }
};

DECLARE_COMPONENT(ConvertAllenLongTracksDQ)

// ================================================================
//  Tuple validators
// ================================================================

class DataQualityValidatorVelo final : public LHCb::Algorithm::Consumer<
                                         void(
                                           const Allen::device_buffer<unsigned>&,
                                           const Allen::device_buffer<unsigned>&,
                                           const Allen::device_buffer<unsigned>&,
                                           const Allen::device_buffer<char>&,
                                           const Allen::device_buffer<char>&,
                                           const Allen::host_buffer<unsigned>&),
                                         Gaudi::Functional::Traits::BaseClass_t<GaudiTupleAlg>> {
public:
  DataQualityValidatorVelo(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer(
      name,
      pSvcLocator,
      {KeyValue {"dev_offsets_velo_tracks", ""},
       KeyValue {"dev_offsets_all_velo_tracks", ""},
       KeyValue {"dev_offsets_velo_track_hit_number", ""},
       KeyValue {"dev_velo_track_hits", ""},
       KeyValue {"dev_velo_kalman_states", ""},
       KeyValue {"host_number_of_events", ""}})
  {
    std::ignore = setProperty("NTuplePrint", false);
  }

  void operator()(
    const Allen::device_buffer<unsigned>& dev_offsets_velo_tracks,
    const Allen::device_buffer<unsigned>& dev_offsets_all_velo_tracks,
    const Allen::device_buffer<unsigned>& dev_offsets_velo_track_hit_number,
    [[maybe_unused]] const Allen::device_buffer<char>& dev_velo_track_hits,
    const Allen::device_buffer<char>& dev_velo_kalman_states,
    const Allen::host_buffer<unsigned>& host_number_of_events) const override
  {
    std::lock_guard<std::mutex> guard(m_mutex);

    const auto event_velo_tracks_offsets = dev_offsets_velo_tracks.to_host();
    const auto offsets_all_velo_tracks = dev_offsets_all_velo_tracks.to_host();
    const auto offsets_velo_track_hit_number = dev_offsets_velo_track_hit_number.to_host();
    const auto velo_states_base = dev_velo_kalman_states.to_host();

    Tuple tree = nTuple("velo_states", "Velo track states");
    auto col = make_column(tree);

    const unsigned number_of_events = host_number_of_events[0];
    for (unsigned evnum = 0; evnum < number_of_events; ++evnum) {
      const auto velo_tracks_offset = event_velo_tracks_offsets[evnum];
      Velo::Consolidated::ConstTracks velo_tracks {
        offsets_all_velo_tracks.data(), offsets_velo_track_hit_number.data(), evnum, number_of_events};
      const unsigned n_velo_states = velo_tracks.number_of_tracks(evnum);

      Velo::Consolidated::ConstStates velo_states {velo_states_base.data(), velo_tracks.total_number_of_tracks()};

      for (unsigned i_track = 0; i_track < n_velo_states; i_track++) {
        const auto state = velo_states.get(velo_tracks_offset + i_track);
        const float tx = state.tx();
        const float ty = state.ty();
        const float rho = std::sqrt(tx * tx + ty * ty);
        col("tx", tx);
        col("ty", ty);
        col("rho", rho);
        col("eta", eta_from_rho(rho));
        col("phi", std::atan2(ty, tx));
        col("n_hits_per_track", static_cast<int>(velo_tracks.number_of_hits(i_track)));
        std::ignore = tree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorVelo");
      }
    }
  }

private:
  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
  mutable std::mutex m_mutex;
};

DECLARE_COMPONENT(DataQualityValidatorVelo)

class DataQualityValidatorPV final : public LHCb::Algorithm::Consumer<
                                       void(
                                         const Allen::device_buffer<PV::Vertex>&,
                                         const Allen::device_buffer<unsigned>&,
                                         const Allen::host_buffer<unsigned>&),
                                       Gaudi::Functional::Traits::BaseClass_t<GaudiTupleAlg>> {
public:
  DataQualityValidatorPV(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer(
      name,
      pSvcLocator,
      {KeyValue {"dev_multi_fit_vertices", ""},
       KeyValue {"dev_number_of_multi_fit_vertices", ""},
       KeyValue {"host_number_of_events", ""}})
  {
    std::ignore = setProperty("NTuplePrint", false);
  }

  void operator()(
    const Allen::device_buffer<PV::Vertex>& dev_multi_fit_vertices,
    const Allen::device_buffer<unsigned>& dev_number_of_multi_fit_vertices,
    const Allen::host_buffer<unsigned>& host_number_of_events) const override
  {
    std::lock_guard<std::mutex> guard(m_mutex);

    const auto PVs = dev_multi_fit_vertices.to_host();
    const auto n_pvs = dev_number_of_multi_fit_vertices.to_host();

    Tuple tree = nTuple("PVs", "Primary vertices");
    Tuple eventTree = nTuple("PV_event", "Primary vertices per event");
    Tuple PVpairsTree = nTuple("PV_pairs", "Primary vertex pairs");
    auto col = make_column(tree);
    auto eventCol = make_column(eventTree);
    auto pairsCol = make_column(PVpairsTree);

    const unsigned number_of_events = host_number_of_events[0];
    for (unsigned evnum = 0; evnum < number_of_events; ++evnum) {
      const int nPVs = n_pvs[evnum];
      const unsigned pv_offset = evnum * PV::max_number_vertices;

      float PVdistance_min, PVdistance_max, PVdistance_mean;
      if (nPVs <= 1) {
        PVdistance_max = -99.f;
        PVdistance_min = -99.f;
        PVdistance_mean = -99.f;
      }
      else {
        PVdistance_mean = 0.f;
        PVdistance_max = 0.f;
        PVdistance_min = std::numeric_limits<float>::infinity();
      }

      for (int i_vertex = 0; i_vertex < nPVs; i_vertex++) {
        const auto pv = PVs[i_vertex + pv_offset];
        const float pv_x = pv.position.x;
        const float pv_y = pv.position.y;
        const float pv_z = pv.position.z;

        col("pv_x", pv_x);
        col("pv_y", pv_y);
        col("pv_z", pv_z);
        col("pv_chi2", pv.chi2);
        col("pv_ndof", pv.ndof);
        col("pv_nTracks", static_cast<float>(pv.nTracks));
        col("cov00", pv.cov00);
        col("cov10", pv.cov10);
        col("cov11", pv.cov11);
        col("cov20", pv.cov20);
        col("cov21", pv.cov21);
        col("cov22", pv.cov22);
        std::ignore = tree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorPV");

        if (nPVs > 1) {
          for (int j = 0; j < i_vertex; ++j) {
            const auto otherVertex = PVs[j + pv_offset];
            const float delta_x = otherVertex.position.x - pv_x;
            const float delta_y = otherVertex.position.y - pv_y;
            const float delta_z = otherVertex.position.z - pv_z;
            const float delta_pos = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

            PVdistance_max = std::max(PVdistance_max, delta_pos);
            PVdistance_min = std::min(PVdistance_min, delta_pos);
            PVdistance_mean += delta_pos;

            pairsCol("PVdelta_z", delta_z);
            std::ignore = PVpairsTree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorPV");
          }
        }
      }

      if (nPVs > 1) {
        PVdistance_mean /= nPVs;
      }
      eventCol("n_pvs", nPVs);
      eventCol("PVdistance_min", PVdistance_min);
      eventCol("PVdistance_max", PVdistance_max);
      eventCol("PVdistance_mean", PVdistance_mean);
      std::ignore = eventTree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorPV");
    }
  }

private:
  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
  mutable std::mutex m_mutex;
};

DECLARE_COMPONENT(DataQualityValidatorPV)

class DataQualityValidatorOccupancy final : public LHCb::Algorithm::Consumer<
                                              void(
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::device_buffer<unsigned>&,
                                                const Allen::host_buffer<unsigned>&),
                                              Gaudi::Functional::Traits::BaseClass_t<GaudiTupleAlg>> {
public:
  DataQualityValidatorOccupancy(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer(
      name,
      pSvcLocator,
      {KeyValue {"dev_station_ocurrences_offset", ""},
       KeyValue {"dev_velo_offsets_estimated_input_size", ""},
       KeyValue {"dev_offsets_velo_tracks", ""},
       KeyValue {"dev_scifi_hit_offsets", ""},
       KeyValue {"dev_ecal_clusters_offsets", ""},
       KeyValue {"dev_scifi_seedsXZ", ""},
       KeyValue {"host_number_of_events", ""}})
  {
    std::ignore = setProperty("NTuplePrint", false);
  }

  void operator()(
    const Allen::device_buffer<unsigned>& dev_station_ocurrences_offset,
    const Allen::device_buffer<unsigned>& dev_velo_offsets_estimated_input_size,
    const Allen::device_buffer<unsigned>& dev_offsets_velo_tracks,
    const Allen::device_buffer<unsigned>& dev_scifi_hit_offsets,
    const Allen::device_buffer<unsigned>& dev_ecal_clusters_offsets,
    const Allen::device_buffer<unsigned>& dev_scifi_seedsXZ,
    const Allen::host_buffer<unsigned>& host_number_of_events) const override
  {
    std::lock_guard<std::mutex> guard(m_mutex);

    const auto scifi_tracks_offsets = dev_scifi_hit_offsets.to_host();
    const auto scifi_seeds = dev_scifi_seedsXZ.to_host();
    const auto velo_offsets_eis = dev_velo_offsets_estimated_input_size.to_host();
    const auto event_velo_tracks_offsets = dev_offsets_velo_tracks.to_host();
    const auto ecal_clusters = dev_ecal_clusters_offsets.to_host();
    const auto muon_offsets = dev_station_ocurrences_offset.to_host();

    Tuple eventTree = nTuple("occupancy", "Event occupancy");
    auto col = make_column(eventTree);

    constexpr std::array<int, 4> st_order {
      Muon::Constants::M5, Muon::Constants::M4, Muon::Constants::M3, Muon::Constants::M2};

    const unsigned number_of_events = host_number_of_events[0];
    for (unsigned evnum = 0; evnum < number_of_events; ++evnum) {
      SciFi::ConstHitCount scifi_hit_count {scifi_tracks_offsets.data(), evnum};
      const int n_scifi_hits = scifi_hit_count.event_number_of_hits();
      const int n_scifi_xz_seeds = scifi_seeds[evnum];

      const unsigned* module_pair_hit_start = velo_offsets_eis.data() + evnum * Velo::Constants::n_module_pairs;
      const unsigned event_hit_start = module_pair_hit_start[0];
      const int n_velo_hits = module_pair_hit_start[Velo::Constants::n_module_pairs] - event_hit_start;

      const auto velo_tracks_offset = event_velo_tracks_offsets[evnum];
      const int n_velo_tracks = event_velo_tracks_offsets[evnum + 1] - velo_tracks_offset;

      const int n_ecal_clusters = ecal_clusters[evnum + 1] - ecal_clusters[evnum];

      const auto station_ocurrences_offset = muon_offsets.data() + evnum * Muon::Constants::n_stations;
      int n_muon_hits = 0;
      for (const int& station : st_order) {
        const auto ocurrences_offset = station_ocurrences_offset[station];
        n_muon_hits += station_ocurrences_offset[station + 1] - ocurrences_offset;
      }

      col("n_scifi_hits", n_scifi_hits);
      col("n_scifi_xz_seeds", n_scifi_xz_seeds);
      col("n_velo_hits", n_velo_hits);
      col("n_velo_tracks", n_velo_tracks);
      col("n_ecal_clusters", n_ecal_clusters);
      col("n_muon_hits", n_muon_hits);
      std::ignore = eventTree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorOccupancy");
    }
  }

private:
  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
  mutable std::mutex m_mutex;
};

DECLARE_COMPONENT(DataQualityValidatorOccupancy)

class DataQualityValidatorLong final
  : public LHCb::Algorithm::
      Consumer<void(const std::vector<DQLongTrackInfo>&), Gaudi::Functional::Traits::BaseClass_t<GaudiTupleAlg>> {
public:
  DataQualityValidatorLong(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer(name, pSvcLocator, {KeyValue {"DQLongTrackInfo", ""}})
  {
    std::ignore = setProperty("NTuplePrint", false);
  }

  void operator()(const std::vector<DQLongTrackInfo>& long_track_infos) const override
  {
    std::lock_guard<std::mutex> guard(m_mutex);

    Tuple tree = nTuple("long_track_particles", "Long track particles");
    Tuple eventTree = nTuple("long_tracks_event", "Long tracks per event");
    auto col = make_column(tree);
    auto eventCol = make_column(eventTree);

    float prop_muon = 0.f;
    float prop_electron = 0.f;
    const int n_long_tracks = static_cast<int>(long_track_infos.size());

    for (int i_track = 0; i_track < n_long_tracks; i_track++) {
      const auto& info = long_track_infos[i_track];
      col("ip_chi2", info.ip_chi2);
      col("chi2", info.chi2);
      col("ip_x", info.ip_x);
      col("ip_y", info.ip_y);
      col("is_muon", info.is_muon);
      col("is_electron", info.is_electron);
      col("cov00", info.cov[0][0]);
      col("cov11", info.cov[1][1]);
      col("cov20", info.cov[2][0]);
      col("cov22", info.cov[2][2]);
      col("cov31", info.cov[3][1]);
      col("cov33", info.cov[3][3]);
      col("tx", info.tx);
      col("ty", info.ty);
      col("qop", info.qop);
      col("eta", info.eta);
      col("phi", std::atan2(info.ty, info.tx));
      col("pt", info.pt);

      if (info.is_muon == 1) prop_muon += 1.0f;
      if (info.is_electron == 1) prop_electron += 1.0f;

      std::ignore = tree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorLong");
    }

    if (n_long_tracks != 0) {
      prop_muon /= n_long_tracks;
      prop_electron /= n_long_tracks;
    }
    eventCol("prop_muon", prop_muon);
    eventCol("prop_electron", prop_electron);
    eventCol("n_long_tracks", n_long_tracks);
    std::ignore = eventTree->write().orThrow("Failed to fill ntuple", "DataQualityValidatorLong");
  }

private:
  mutable std::mutex m_mutex;
};

DECLARE_COMPONENT(DataQualityValidatorLong)
