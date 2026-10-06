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
 * Convert Allen multi-body secondary-vertex composites into LHCb::Particles.
 *
 * Two steps, following the GaudiAllenSVsToRecVertexV2 / ConvertAllenLongToV3Tracks
 * multi-event patterns:
 *   (1) ConvertAllenComposites    — multi-event device view -> per-event POD data
 *   (2) AssociateAllenComposites  — per-event POD data -> LHCb::Particle/Vertex/ProtoParticle
 *
 * The GPU extraction is needed because MultiEventCompositeParticles is a view over
 * device memory; it cannot be iterated on the host directly.
 */
#include "GaudiAlg/Transformer.h"

#include "Event/Particle.h"
#include "Event/ProtoParticle.h"
#include "Event/Track.h"
#include "Event/Vertex.h"
#include "GaudiKernel/SmartRefVector.h"

#include "ParticleTypes.cuh"
#include "States.cuh"
#include "AllenBuffer.cuh"
#include "TargetFunction.cuh"
#include "EventTransformer.h"

#include <array>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

namespace {
  template<typename Fn>
  auto global_function(const Fn& fn)
  {
    return GlobalFunction<Fn> {fn};
  }

  constexpr unsigned max_composite_children = 4;
} // namespace

/// POD snapshot of one daughter, self-contained (no Allen view pointers).
struct AllenCompositeDaughter {
  unsigned track_index = 0;
  float px = 0.f;
  float py = 0.f;
  float pz = 0.f;
  int charge = 0;
  bool is_muon = false;
  bool is_electron = false;
};

/// POD snapshot of one composite.
struct AllenCompositeData {
  unsigned n_children = 0;
  float x = 0.f, y = 0.f, z = 0.f;
  float c00 = 0.f, c10 = 0.f, c11 = 0.f, c20 = 0.f, c21 = 0.f, c22 = 0.f;
  float chi2 = 0.f;
  unsigned ndof = 0;
  float mass = 0.f;
  float pv_x = 0.f, pv_y = 0.f, pv_z = 0.f;
  float pv_c00 = 0.f, pv_c10 = 0.f, pv_c11 = 0.f, pv_c20 = 0.f, pv_c21 = 0.f, pv_c22 = 0.f;
  float pv_chi2 = 0.f, pv_ndof = 0.f;
  std::array<AllenCompositeDaughter, max_composite_children> children {};
};

namespace {
  using MEC = Allen::Views::Physics::MultiEventCompositeParticles;

  __global__ void extract_composite_offsets_k(const MEC* composites, unsigned* offsets)
  {
    for (unsigned i = threadIdx.x; i < composites->number_of_events(); i += blockDim.x) {
      offsets[i] = composites->container(i).offset();
      if (i == composites->number_of_events() - 1) {
        offsets[i + 1] = composites->container(i).offset() + composites->container(i).size();
      }
    }
  }

  __global__ void extract_composites_k(const MEC* composites, AllenCompositeData* out)
  {
    const unsigned event = blockIdx.x;
    const auto& container = composites->container(event);
    const unsigned offset = container.offset();

    for (unsigned i = threadIdx.x; i < container.size(); i += blockDim.x) {
      const auto& composite = container.particle(i);
      const unsigned n_children = composite.number_of_children();
      if (n_children > max_composite_children) continue;

      AllenCompositeData data;
      data.n_children = n_children;

      if (composite.has_vertex()) {
        const auto vertex = composite.vertex();
        data.x = vertex.x();
        data.y = vertex.y();
        data.z = vertex.z();
        data.c00 = vertex.c00();
        data.c10 = vertex.c10();
        data.c11 = vertex.c11();
        data.c20 = vertex.c20();
        data.c21 = vertex.c21();
        data.c22 = vertex.c22();
        data.chi2 = vertex.chi2();
        data.ndof = vertex.ndof();
        data.mass = composite.m();
        data.pv_c00 = composite.pv().cov00;
        data.pv_c10 = composite.pv().cov10;
        data.pv_c11 = composite.pv().cov11;
        data.pv_c20 = composite.pv().cov20;
        data.pv_c21 = composite.pv().cov21;
        data.pv_c22 = composite.pv().cov22;
        data.pv_chi2 = composite.pv().chi2;
        data.pv_ndof = composite.pv().ndof;
        data.pv_x = composite.pv().position.x;
        data.pv_y = composite.pv().position.y;
        data.pv_z = composite.pv().position.z;
      }

      for (unsigned j = 0; j < n_children; ++j) {
        const auto* child = Allen::dyn_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(j));
        if (!child) continue;

        const auto state = child->state();
        AllenCompositeDaughter daughter;
        daughter.track_index = child->get_index();
        daughter.px = state.px();
        daughter.py = state.py();
        daughter.pz = state.pz();
        daughter.charge = state.charge();
        daughter.is_muon = child->is_muon();
        daughter.is_electron = child->is_electron();
        data.children[j] = daughter;
      }

      out[offset + i] = data;
    }
  }
} // namespace

// ================================================================
//  Step 1: multi-event device view -> per-event AllenCompositeData
// ================================================================

class ConvertAllenComposites final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<std::vector<AllenCompositeData>>(
      const Allen::device_buffer<MEC>&)> {
public:
  using base_class = LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<std::vector<AllenCompositeData>>(
    const Allen::device_buffer<MEC>&)>;
  using KeyValue = typename base_class::KeyValue;

  ConvertAllenComposites(const std::string& name, ISvcLocator* pSvcLocator) :
    base_class(name, pSvcLocator, {KeyValue {"dev_multi_event_composites", ""}}, {KeyValue {"AllenCompositeData", ""}})
  {}

  std::tuple<std::vector<std::vector<AllenCompositeData>>> operator()(
    const EventContext& ctx,
    const Allen::device_buffer<MEC>& dev_composites) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
    const Allen::Context& context = ctxExt->allen_context;
    const unsigned n_events = ctxExt->number_of_events;

    Allen::device_buffer<unsigned> dev_offsets {n_events + 1, ctxExt->memory_managers};
    global_function(extract_composite_offsets_k)(dim3(1), dim3(1), context)(dev_composites.data(), dev_offsets.data());
    auto h_offsets = dev_offsets.to_host();
    const unsigned n_total = h_offsets[n_events];

    Allen::device_buffer<AllenCompositeData> dev_data {n_total, ctxExt->memory_managers};
    if (n_total != 0) {
      global_function(extract_composites_k)(dim3(n_events), dim3(256), context)(dev_composites.data(), dev_data.data());
    }
    auto h_data = dev_data.to_host();

    std::vector<std::vector<AllenCompositeData>> all_composites;
    all_composites.reserve(n_events);
    for (unsigned event = 0; event < n_events; ++event) {
      all_composites.emplace_back(h_data.begin() + h_offsets[event], h_data.begin() + h_offsets[event + 1]);
    }

    return std::make_tuple(std::move(all_composites));
  }
};

DECLARE_COMPONENT(ConvertAllenComposites)

// ================================================================
//  Step 2: per-event AllenCompositeData -> LHCb objects
// ================================================================

namespace {
  float daughter_mass(const AllenCompositeDaughter& daughter)
  {
    if (daughter.is_muon) return Allen::mMu;
    if (daughter.is_electron) return Allen::mEl;
    return Allen::mPi;
  }

  LHCb::ParticleID daughter_pid(const AllenCompositeDaughter& daughter)
  {
    if (daughter.is_muon) return LHCb::ParticleID {daughter.charge > 0 ? -13 : 13};
    if (daughter.is_electron) return LHCb::ParticleID {daughter.charge > 0 ? -11 : 11};
    return LHCb::ParticleID {daughter.charge > 0 ? 211 : -211};
  }

  void fill_daughter(LHCb::Particle* particle, const AllenCompositeDaughter& daughter)
  {
    const float mass = daughter_mass(daughter);
    const float energy =
      std::sqrtf(daughter.px * daughter.px + daughter.py * daughter.py + daughter.pz * daughter.pz + mass * mass);
    particle->setParticleID(daughter_pid(daughter));
    particle->setMeasuredMass(static_cast<double>(mass));
    particle->setMomentum(Gaudi::LorentzVector {
      static_cast<double>(daughter.px),
      static_cast<double>(daughter.py),
      static_cast<double>(daughter.pz),
      static_cast<double>(energy)});
  }

  void fill_composite(LHCb::Particle* particle)
  {
    Gaudi::LorentzVector four_momentum;
    for (const auto& daughter : particle->daughters()) {
      four_momentum += daughter.target()->momentum();
    }
    particle->setMeasuredMass(four_momentum.M());
    particle->setMomentum(four_momentum);
  }
} // namespace

class AssociateAllenComposites final
  : public Gaudi::Functional::MultiTransformer<
      std::tuple<LHCb::Particles, LHCb::Particles, LHCb::ProtoParticles, LHCb::Vertices>(
        const std::vector<AllenCompositeData>&,
        const LHCb::Event::v1::Tracks&)> {
public:
  AssociateAllenComposites(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"AllenCompositeData", ""}, KeyValue {"LongTracks", "Allen/Out/ForwardTracks"}},
      {KeyValue {"CompositeParticles", ""},
       KeyValue {"DaughterParticles", ""},
       KeyValue {"ProtoParticles", ""},
       KeyValue {"Vertices", ""}})
  {}

  std::tuple<LHCb::Particles, LHCb::Particles, LHCb::ProtoParticles, LHCb::Vertices> operator()(
    const std::vector<AllenCompositeData>& composites,
    const LHCb::Event::v1::Tracks& tracks) const override
  {
    LHCb::Particles rec_composites;
    LHCb::Particles rec_daughters;
    LHCb::ProtoParticles proto_particles;
    LHCb::Vertices vertices;
    LHCb::VertexBases pvs;

    for (const auto& composite : composites) {
      auto mother = std::make_unique<LHCb::Particle>(LHCb::ParticleID {0});
      auto* mother_ptr = mother.get();
      rec_composites.insert(mother.release());

      for (unsigned i = 0; i < composite.n_children; ++i) {
        const auto& daughter = composite.children[i];

        auto* track = *(tracks.begin() + daughter.track_index);
        if (!track) continue;

        auto reco_daughter = std::make_unique<LHCb::Particle>();
        auto* daughter_ptr = reco_daughter.get();
        rec_daughters.insert(reco_daughter.release());

        auto proto = std::make_unique<LHCb::ProtoParticle>();
        auto* proto_ptr = proto.get();
        proto_particles.insert(proto.release());
        proto_ptr->setTrack(track);
        daughter_ptr->setProto(proto_ptr);

        fill_daughter(daughter_ptr, daughter);
        mother_ptr->addToDaughters(daughter_ptr);
      }

      // Allen currently has one vertex per composite.
      auto vertex = std::make_unique<LHCb::Vertex>();
      auto* vertex_ptr = vertex.get();
      vertices.insert(vertex.release());

      vertex_ptr->setPosition(Gaudi::XYZPoint {
        static_cast<double>(composite.x), static_cast<double>(composite.y), static_cast<double>(composite.z)});

      Gaudi::SymMatrix3x3 position_covariance;
      position_covariance(0, 0) = static_cast<double>(composite.c00);
      position_covariance(1, 0) = static_cast<double>(composite.c10);
      position_covariance(1, 1) = static_cast<double>(composite.c11);
      position_covariance(2, 0) = static_cast<double>(composite.c20);
      position_covariance(2, 1) = static_cast<double>(composite.c21);
      position_covariance(2, 2) = static_cast<double>(composite.c22);
      vertex_ptr->setCovMatrix(position_covariance);
      vertex_ptr->setChi2AndDoF(static_cast<double>(composite.chi2), static_cast<int>(composite.ndof));
      vertex_ptr->setTechnique(LHCb::Vertex::CreationMethod::VertexFitter);

      for (const auto& daughter : mother_ptr->daughters()) {
        vertex_ptr->addToOutgoingParticles(daughter);
      }
      mother_ptr->setEndVertex(vertex_ptr);

      // Create PV, fill it with info from Allen, and link it to the composite vertex.
      auto pv = std::make_unique<LHCb::VertexBase>();
      auto* pv_ptr = pv.get();
      pvs.insert(pv.release());
      pv_ptr->setPosition(Gaudi::XYZPoint {
        static_cast<double>(composite.pv_x), static_cast<double>(composite.pv_y), static_cast<double>(composite.pv_z)});
      Gaudi::SymMatrix3x3 pv_cov;
      pv_cov[0][0] = static_cast<double>(composite.pv_c00);
      pv_cov[1][0] = static_cast<double>(composite.pv_c10);
      pv_cov[1][1] = static_cast<double>(composite.pv_c11);
      pv_cov[2][0] = static_cast<double>(composite.pv_c20);
      pv_cov[2][1] = static_cast<double>(composite.pv_c21);
      pv_cov[2][2] = static_cast<double>(composite.pv_c22);
      pv_ptr->setCovMatrix(pv_cov);
      pv_ptr->setChi2(static_cast<double>(composite.pv_chi2));
      pv_ptr->setNDoF(static_cast<int>(composite.pv_ndof));

      mother_ptr->setPV(pv_ptr);
      fill_composite(mother_ptr);
    }

    return {std::move(rec_composites), std::move(rec_daughters), std::move(proto_particles), std::move(vertices)};
  }
};

DECLARE_COMPONENT(AssociateAllenComposites)
