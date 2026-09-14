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
 * Convert Allen Long tracks / BasicParticles into LHCb::Event::v3::Tracks.
 *
 * Multi-event.  Takes the MultiEventBasicParticles device view, runs GPU
 * kernels to extract LHCbIDs, segment counts, states, and z positions from
 * all segments into flat device buffers, copies everything to host, then
 * builds per-event v3 Long tracks.
 *
 * Templated on optional Rich state types:
 *   ConvertAllenLongToV3Tracks_Basic   = ConvertAllenLongToV3Tracks<>
 *   ConvertAllenLongToV3Tracks_Rich    = ConvertAllenLongToV3Tracks<rich1_front_states, ...>
 */

#include "GaudiAlg/Transformer.h"

#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/TrackEnums.h"
#include "Event/UniqueIDGenerator.h"
#include "Event/StateParameters.h"

#include "ParticleTypes.cuh"
#include "States.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"
#include "KalmanVeloStateWithQoP.h"
#include "PrefixSum.cuh"

#include <algorithm>

namespace {
  using OutTracks = LHCb::Event::v3::Tracks;
  namespace OutTag = LHCb::Event::v3::Tag;
  using SL = OutTracks::StateLocation;

  using LongStates = LHCb::Event::v3::
    available_states_t<LHCb::Event::v3::TrackType::Long, LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>;

  // ---- GPU helpers ----

  template<typename Fn>
  auto global_function(const Fn& fn)
  {
    return GlobalFunction<Fn> {fn};
  }

  void prefix_sum(Allen::host_buffer<unsigned>& dev_array)
  {
    unsigned sum = 0;
    for (unsigned i = 0; i < dev_array.size(); i++) {
      unsigned val = dev_array[i];
      dev_array[i] = sum;
      sum += val;
    }
  }

  // ---- Kernels ----

  __global__ void extract_track_offsets_k(
    const Allen::Views::Physics::MultiEventBasicParticles* particles,
    unsigned* track_offsets)
  {
    for (unsigned i = threadIdx.x; i < particles->number_of_events(); i += blockDim.x) {
      track_offsets[i] = particles->container(i).offset();
      if (i == particles->number_of_events() - 1) {
        track_offsets[i + 1] = particles->container(i).offset() + particles->container(i).size();
      }
    }
  }

  __global__ void extract_hit_counts_k(
    const Allen::Views::Physics::MultiEventBasicParticles* particles,
    unsigned* hit_offsets)
  {
    const unsigned event_number = blockIdx.x;
    for (unsigned i = threadIdx.x; i < particles->container(event_number).size(); i += blockDim.x) {
      const auto& particle = particles->container(event_number).particle(i);
      hit_offsets[particles->container(event_number).offset() + i] = particle.number_of_ids();
    }
  }

  __global__ void extract_particle_data_k(
    const Allen::Views::Physics::MultiEventBasicParticles* particles,
    const unsigned* hit_offsets,
    unsigned* lhcb_ids,
    unsigned* seg_counts,
    float* qop,
    float* chi2,
    unsigned* ndof,
    float* state_x,
    float* state_y,
    float* state_z,
    float* state_tx,
    float* state_ty,
    float* state_c00,
    float* state_c20,
    float* state_c22,
    float* state_c11,
    float* state_c31,
    float* state_c33,
    float* first_meas_z,
    float* last_meas_z)
  {
    const unsigned event_number = blockIdx.x;
    const auto& container = particles->container(event_number);
    const unsigned offset = container.offset();

    for (unsigned i = threadIdx.x; i < container.size(); i += blockDim.x) {
      const unsigned p = offset + i;
      const auto& particle = container.particle(i);
      const auto& track = particle.track();

      const unsigned n_ids = particle.number_of_ids();
      const unsigned id_start = hit_offsets[p];
      for (unsigned j = 0; j < n_ids; ++j)
        lhcb_ids[id_start + j] = particle.id(j);

      using seg = Allen::Views::Physics::Track::segment;
      seg_counts[4 * p + 0] = track.has<seg::velo>() ? track.number_of_segment_hits<seg::velo>() : 0;
      seg_counts[4 * p + 1] = track.has<seg::ut>() ? track.number_of_segment_hits<seg::ut>() : 0;
      seg_counts[4 * p + 2] = track.has<seg::scifi>() ? track.number_of_segment_hits<seg::scifi>() : 0;
      seg_counts[4 * p + 3] = track.has<seg::muon>() ? track.number_of_segment_hits<seg::muon>() : 0;

      auto state = particle.state();
      qop[p] = state.qop();
      chi2[p] = state.chi2();
      ndof[p] = state.ndof();
      state_x[p] = state.x();
      state_y[p] = state.y();
      state_z[p] = state.z();
      state_tx[p] = state.tx();
      state_ty[p] = state.ty();
      state_c00[p] = state.c00();
      state_c20[p] = state.c20();
      state_c22[p] = state.c22();
      state_c11[p] = state.c11();
      state_c31[p] = state.c31();
      state_c33[p] = state.c33();

      if (track.has<seg::velo>()) {
        const auto& velo_seg = track.track_segment<seg::velo>();
        first_meas_z[p] = velo_seg.hit(velo_seg.number_of_ids() - 1).z();
      }
      else
        first_meas_z[p] = state.z();

      if (track.has<seg::scifi>()) {
        const auto& scifi_seg = track.track_segment<seg::scifi>();
        last_meas_z[p] = scifi_seg.hit(0).z0();
      }
      else
        last_meas_z[p] = state.z();
    }
  }

} // anonymous namespace

// ---- Rich state tag types ----

struct rich1_front_states {
  using type = SimpleKalmanState;
  static constexpr auto keyname = "dev_rich_R1F_data";
};
struct rich1_back_states {
  using type = SimpleKalmanState;
  static constexpr auto keyname = "dev_rich_R1B_data";
};
struct rich2_front_states {
  using type = SimpleKalmanState;
  static constexpr auto keyname = "dev_rich_R2F_data";
};
struct rich2_back_states {
  using type = SimpleKalmanState;
  static constexpr auto keyname = "dev_rich_R2B_data";
};

template<typename T>
struct in_type {
  using type = T;
};
template<>
struct in_type<rich1_front_states> {
  using type = rich1_front_states::type;
};
template<>
struct in_type<rich1_back_states> {
  using type = rich1_back_states::type;
};
template<>
struct in_type<rich2_front_states> {
  using type = rich2_front_states::type;
};
template<>
struct in_type<rich2_back_states> {
  using type = rich2_back_states::type;
};
template<typename T>
using in_type_t = typename in_type<T>::type;

template<typename AllenInput>
auto get_state_keyname()
{
  if constexpr (std::is_same_v<AllenInput, rich1_front_states>) return rich1_front_states::keyname;
  if constexpr (std::is_same_v<AllenInput, rich1_back_states>) return rich1_back_states::keyname;
  if constexpr (std::is_same_v<AllenInput, rich2_front_states>) return rich2_front_states::keyname;
  if constexpr (std::is_same_v<AllenInput, rich2_back_states>) return rich2_back_states::keyname;
  return "";
}

template<typename KeyValue, typename... RichStates>
auto get_rich_input_names()
{
  return std::make_tuple(KeyValue {get_state_keyname<RichStates>(), ""}...);
}

// ==================================================================

template<typename... RichStates>
class ConvertAllenLongToV3Tracks final : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<OutTracks>(
                                           const Allen::device_buffer<Allen::Views::Physics::MultiEventBasicParticles>&,
                                           const Allen::device_buffer<in_type_t<RichStates>>&...,
                                           const LHCb::UniqueIDGenerator&)> {

public:
  using MEC = Allen::Views::Physics::MultiEventBasicParticles;

  using base_class = LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<OutTracks>(
    const Allen::device_buffer<MEC>&,
    const Allen::device_buffer<in_type_t<RichStates>>&...,
    const LHCb::UniqueIDGenerator&)>;
  using KeyValue = typename base_class::KeyValue;

  ConvertAllenLongToV3Tracks(const std::string& name, ISvcLocator* pSvcLocator) :
    base_class(
      name,
      pSvcLocator,
      std::tuple_cat(
        std::make_tuple(KeyValue {"dev_particle_container", ""}),
        get_rich_input_names<KeyValue, RichStates...>(),
        std::make_tuple(KeyValue {"InputUniqueIDGenerator", LHCb::UniqueIDGeneratorLocation::Default})),
      {KeyValue {"OutputTracks", ""}})
  {}

  std::tuple<std::vector<OutTracks>> operator()(
    const EventContext& ctx,
    const Allen::device_buffer<MEC>& dev_particles,
    const Allen::device_buffer<in_type_t<RichStates>>&... dev_rich_states,
    const LHCb::UniqueIDGenerator& unique_id_gen) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
    const Allen::Context& context = ctxExt->allen_context;
    unsigned n_events = ctxExt->number_of_events;

    // --- a) extract per-event track offsets ---
    Allen::device_buffer<unsigned> dev_track_offsets {n_events + 1, ctxExt->memory_managers};
    global_function(extract_track_offsets_k)(dim3(1), dim3(256), context)(
      dev_particles.data(), dev_track_offsets.data());
    auto h_track_offsets = dev_track_offsets.to_host();
    const unsigned n_tracks_total = h_track_offsets[n_events];

    // --- b) count LHCbIDs per particle ---
    Allen::device_buffer<unsigned> dev_hit_offsets {n_tracks_total + 1, ctxExt->memory_managers};
    global_function(extract_hit_counts_k)(dim3(n_events), dim3(256), context)(
      dev_particles.data(), dev_hit_offsets.data());
    auto h_hit_offsets = dev_hit_offsets.to_host();
    prefix_sum(h_hit_offsets);
    h_hit_offsets.copy_to(dev_hit_offsets);
    const unsigned n_hits_total = h_hit_offsets[n_tracks_total];

    // --- c) extract particle data ---
    Allen::device_buffer<unsigned> dev_lhcb_ids {n_hits_total, ctxExt->memory_managers};
    Allen::device_buffer<unsigned> dev_seg_counts {n_tracks_total * 4, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_qop {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_chi2 {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<unsigned> dev_ndof {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_sx {n_tracks_total, ctxExt->memory_managers},
      dev_sy {n_tracks_total, ctxExt->memory_managers}, dev_sz {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_stx {n_tracks_total, ctxExt->memory_managers},
      dev_sty {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_sc00 {n_tracks_total, ctxExt->memory_managers},
      dev_sc20 {n_tracks_total, ctxExt->memory_managers}, dev_sc22 {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_sc11 {n_tracks_total, ctxExt->memory_managers},
      dev_sc31 {n_tracks_total, ctxExt->memory_managers}, dev_sc33 {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_first_z {n_tracks_total, ctxExt->memory_managers};
    Allen::device_buffer<float> dev_last_z {n_tracks_total, ctxExt->memory_managers};

    global_function(extract_particle_data_k)(dim3(n_events), dim3(256), context)(
      dev_particles.data(),
      dev_hit_offsets.data(),
      dev_lhcb_ids.data(),
      dev_seg_counts.data(),
      dev_qop.data(),
      dev_chi2.data(),
      dev_ndof.data(),
      dev_sx.data(),
      dev_sy.data(),
      dev_sz.data(),
      dev_stx.data(),
      dev_sty.data(),
      dev_sc00.data(),
      dev_sc20.data(),
      dev_sc22.data(),
      dev_sc11.data(),
      dev_sc31.data(),
      dev_sc33.data(),
      dev_first_z.data(),
      dev_last_z.data());

    // --- copy outputs + Rich states to host ---
    auto h_lhcb_ids = dev_lhcb_ids.to_host();
    auto h_seg_counts = dev_seg_counts.to_host();
    auto h_qop = dev_qop.to_host();
    auto h_chi2 = dev_chi2.to_host();
    auto h_ndof = dev_ndof.to_host();
    auto h_sx = dev_sx.to_host(), h_sy = dev_sy.to_host(), h_sz = dev_sz.to_host();
    auto h_stx = dev_stx.to_host(), h_sty = dev_sty.to_host();
    auto h_sc00 = dev_sc00.to_host(), h_sc20 = dev_sc20.to_host(), h_sc22 = dev_sc22.to_host();
    auto h_sc11 = dev_sc11.to_host(), h_sc31 = dev_sc31.to_host(), h_sc33 = dev_sc33.to_host();
    auto h_first_z = dev_first_z.to_host();
    auto h_last_z = dev_last_z.to_host();

    auto rich_tuple = std::make_tuple(dev_rich_states.to_host()...);

    // --- build per-event v3 Long tracks ---
    std::vector<OutTracks> all_tracks;
    all_tracks.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned p_begin = h_track_offsets[evt];
      const unsigned p_end = h_track_offsets[evt + 1];

      auto zn = Zipping::generateZipIdentifier();
      OutTracks out(
        LHCb::Event::v3::TrackType::Long,
        LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter,
        false,
        unique_id_gen,
        zn);

      for (unsigned p = p_begin; p < p_end; ++p) {

        unsigned n_vp = h_seg_counts[4 * p + 0];
        unsigned n_ut = h_seg_counts[4 * p + 1];
        unsigned n_ft = h_seg_counts[4 * p + 2];
        unsigned n_mu = h_seg_counts[4 * p + 3];

        KalmanVeloState beamline(
          h_sx[p],
          h_sy[p],
          h_sz[p],
          h_stx[p],
          h_sty[p],
          h_sc00[p],
          h_sc20[p],
          h_sc22[p],
          h_sc11[p],
          h_sc31[p],
          h_sc33[p]);
        float qop = h_qop[p];
        float qopVar = m_qopvar_rel * qop * qop;

        std::vector<KalmanVeloStateWithQoP> states {{beamline, qop, qopVar}};

        // Append Rich states
        if constexpr (sizeof...(RichStates) > 0)
          append_rich(states, p, rich_tuple, std::index_sequence_for<RichStates...> {});

        auto newTrack = out.template emplace_back<SIMDWrapper::InstructionSet::Scalar>();
        unsigned id_off = h_hit_offsets[p];

        // VPHits
        newTrack.template field<OutTag::VPHits>().resize(n_vp);
        for (unsigned i = 0; i < n_vp; ++i)
          newTrack.template field<OutTag::VPHits>()[i].template field<OutTag::LHCbID>().set(
            LHCb::LHCbID {h_lhcb_ids[id_off + i]});
        id_off += n_vp;

        // UTHits
        newTrack.template field<OutTag::UTHits>().resize(n_ut);
        for (unsigned i = 0; i < n_ut; ++i)
          newTrack.template field<OutTag::UTHits>()[i].template field<OutTag::LHCbID>().set(
            LHCb::LHCbID {h_lhcb_ids[id_off + i]});
        id_off += n_ut;

        // FTHits
        newTrack.template field<OutTag::FTHits>().resize(n_ft);
        for (unsigned i = 0; i < n_ft; ++i)
          newTrack.template field<OutTag::FTHits>()[i].template field<OutTag::LHCbID>().set(
            LHCb::LHCbID {h_lhcb_ids[id_off + i]});
        id_off += n_ft;

        // MuonHits
        newTrack.template field<OutTag::MuonHits>().resize(n_mu);
        for (unsigned i = 0; i < n_mu; ++i)
          newTrack.template field<OutTag::MuonHits>()[i].template field<OutTag::LHCbID>().set(
            LHCb::LHCbID {h_lhcb_ids[id_off + i]});

        newTrack.template field<OutTag::history>().set(LHCb::Event::Enum::Track::History::PrForward);
        using int_v = decltype(newTrack.template field<OutTag::UniqueID>().get());
        newTrack.template field<OutTag::UniqueID>().set(unique_id_gen.generate<int_v>().value());
        newTrack.template field<OutTag::Chi2>().set(h_chi2[p]);
        newTrack.template field<OutTag::nDoF>().set(static_cast<int>(h_ndof[p]));

        update_all_states(newTrack, states, h_first_z[p], h_last_z[p], LongStates {});
      }

      all_tracks.emplace_back(std::move(out));
    }

    return std::make_tuple(std::move(all_tracks));
  }

private:
  Gaudi::Property<float> m_qopvar_rel {this, "relQoPVar", 0.1, "Default relative qop variance"};

  template<typename RichTuple, std::size_t... Is>
  static void append_rich(
    std::vector<KalmanVeloStateWithQoP>& states,
    unsigned p,
    const RichTuple& rich_tuple,
    std::index_sequence<Is...>)
  {
    (states.push_back({(KalmanVeloState) std::get<Is>(rich_tuple)[p], std::get<Is>(rich_tuple)[p].qop, 0.f}), ...);
  }
};

// ---- Instantiations ----

using ConvertAllenLongToV3Tracks_Basic = ConvertAllenLongToV3Tracks<>;
DECLARE_COMPONENT_WITH_ID(ConvertAllenLongToV3Tracks_Basic, "ConvertAllenLongToV3Tracks_Basic")

using ConvertAllenLongToV3Tracks_Rich =
  ConvertAllenLongToV3Tracks<rich1_front_states, rich1_back_states, rich2_front_states, rich2_back_states>;
DECLARE_COMPONENT_WITH_ID(ConvertAllenLongToV3Tracks_Rich, "ConvertAllenLongToV3Tracks_Rich")
