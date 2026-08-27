/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "VertexingValidator.cuh"
#include <chrono>

#include "Event/ODIN.h"
#include "ODINBank.cuh"

#include <stdio.h>

INSTANTIATE_ALGORITHM(vertexing_validator::vertexing_validator_t)

namespace {
  // Function which helps transform a Allen::Views::Physics::BasicParticle into Checker::Track
  __device__ inline Checker::Track make_checker_track(const Allen::Views::Physics::BasicParticle* track)
  {
    Checker::Track t;

    const auto num_hits = track->number_of_ids();
    for (unsigned int ihit = 0; ihit < num_hits; ihit++) {
      const auto id = track->id(ihit);
      t.addId(id);
    }
    return t;
  }

} // namespace

__global__ void vertexing_validator::vertexing_validator(vertexing_validator::Parameters parameters)
{
  constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();

  // Basic
  const unsigned event_number = blockIdx.x;

  // Fill composites
  const auto composites = parameters.dev_multi_event_composites_view->container(event_number);
  auto checker_composites = parameters.dev_checker_composites + composites.offset();
  auto dumpping_objects = parameters.dev_dumpping_objects + composites.offset();
  {
    const auto num_composites = composites.size();
    for (unsigned composite_idx = threadIdx.x; composite_idx < num_composites; composite_idx += blockDim.x) {

      // Fetch data
      const auto composite = composites.particle(composite_idx);
      const auto secondary_vertex = composite.vertex();
      const unsigned int nChildren = composite.number_of_children();

      if (nChildren > VertexFit::max_tracks_per_sv) {
        printf(
          "Composite has a larger number of children than %i, the maximum defined by default \n",
          VertexFit::max_tracks_per_sv);
        continue;
      }

      // Array of children. Hard-coded to the maximum number of children per SV
      std::array<const Allen::Views::Physics::BasicParticle*, VertexFit::max_tracks_per_sv> children {};
      Checker::Composite c;
      c.nChildren = nChildren;
      for (unsigned int i_child = 0; i_child < nChildren; i_child++) {
        children[i_child] = static_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(i_child));
        if (!children[i_child]) {
          printf("Debug: couldn't cast the children from composite \n");
          continue;
        }
        c.Tracks[i_child] = make_checker_track(children[i_child]);
      }

      c.idx = composites.offset() + composite_idx;
      checker_composites[composite_idx] = c;

      // Fill dumpper
      Allen::DumppingSVs::genericSV dumpper;

      // Dmen: add Run Number too!
      const unsigned event_id = LHCb::ODIN {parameters.dev_odin_data[event_number]}.eventNumber();
      dumpper.ievent = event_id;
      dumpper.nChildren = nChildren;
      dumpper.x = secondary_vertex.x();
      dumpper.y = secondary_vertex.y();
      dumpper.z = secondary_vertex.z();
      dumpper.c00 = secondary_vertex.c00();
      dumpper.c11 = secondary_vertex.c11();
      dumpper.c22 = secondary_vertex.c22();
      dumpper.px = secondary_vertex.px();
      dumpper.py = secondary_vertex.py();
      dumpper.pz = secondary_vertex.pz();

      dumpper.pvx = composite.has_pv() ? composite.pv().position.x : NaN;
      dumpper.pvy = composite.has_pv() ? composite.pv().position.y : NaN;
      dumpper.pvz = composite.has_pv() ? composite.pv().position.z : NaN;

      // Reconstructed mass of the composite under pion hypothesis for its daughters
      dumpper.recMass = composite.m();

      // TODO: add this info when compatibility with downstream is ready
      // dumpper.armenteros_x = secondary_vertex.downstream_armentero_podolanski_x();
      // dumpper.armenteros_y = secondary_vertex.downstream_armentero_podolanski_y();
      // dumpper.quality = secondary_vertex.downstream_quality();

      for (unsigned int i = 0; i < nChildren; i++) {
        for (unsigned int j = 0; j < nChildren; j++) {
          dumpper.doca[i][j] = composite.doca(i, j);
        }
      }
      dumpper.fd = composite.fd();
      // TODO: dump fd chi2 when available
      dumpper.ctau = composite.ctau();
      dumpper.dz = composite.dz();
      dumpper.drho = composite.drho();
      dumpper.eta = composite.eta();
      dumpper.mcor = composite.mcor();
      dumpper.minip = composite.minip();
      dumpper.minp = composite.minp();
      dumpper.minpt = composite.minpt();
      dumpper.maxp = composite.maxp();
      dumpper.maxpt = composite.maxpt();
      dumpper.dira = composite.dira();
      dumpper.bpv_ip = composite.ip();

      for (unsigned i_child = 0; i_child < nChildren; i_child++) {

        dumpper.children[i_child].px = children[i_child]->state().px();
        dumpper.children[i_child].py = children[i_child]->state().py();
        dumpper.children[i_child].pz = children[i_child]->state().pz();
        dumpper.children[i_child].pt = children[i_child]->state().pt();
        dumpper.children[i_child].p = children[i_child]->state().p();
        dumpper.children[i_child].eta = children[i_child]->state().eta();
        dumpper.children[i_child].rho = children[i_child]->state().rho();
        dumpper.children[i_child].ip_chi2 = children[i_child]->ip_chi2();
        dumpper.children[i_child].chi2 = children[i_child]->chi2();
        dumpper.children[i_child].ndof = children[i_child]->ndof();
        dumpper.children[i_child].chi2ndof = children[i_child]->chi2() / children[i_child]->ndof();

        dumpper.children[i_child].is_electron = children[i_child]->is_electron();
        dumpper.children[i_child].is_muon = children[i_child]->is_muon();
      }

      dumpping_objects[composite_idx] = dumpper;
    }
  }
  __syncthreads();
}

void vertexing_validator::vertexing_validator_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_dumpping_objects_t>(arguments, first<host_number_of_vertices_t>(arguments));
  set_size<dev_checker_composites_t>(arguments, first<host_number_of_vertices_t>(arguments));
}

void vertexing_validator::vertexing_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  global_function(vertexing_validator)(first<host_number_of_events_t>(arguments), m_block_dim, context)(arguments);

  // Load result to host
  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  const auto host_checker_composites = make_host_buffer<dev_checker_composites_t>(arguments, context);
  const auto host_dumpping_objects = make_host_buffer<dev_dumpping_objects_t>(arguments, context);
  const auto event_composites_offsets = make_host_buffer<dev_offset_vertices_t>(arguments, context);
  std::vector<Checker::Composites> composites;
  composites.resize(event_list.size());

  for (size_t i = 0; i < event_list.size(); ++i) {
    const auto evnum = event_list[i];
    const auto event_offset = event_composites_offsets[evnum];
    const auto n_tracks = event_composites_offsets[evnum + 1] - event_offset;
    std::vector<Checker::Composite> event_composites = {
      host_checker_composites.begin() + event_offset, host_checker_composites.begin() + event_offset + n_tracks};
    composites[i] = event_composites;
  }

  // Obtain checker
  auto& checker = runtime_options.checker_invoker->checker<CompositeDumper>(name(), m_root_output_filename);

  // Fetch event level infos
  float host_polarity = constants.magnet_polarity;

  const auto slice_idx = static_cast<unsigned>(runtime_options.slice_index);
  const auto time = static_cast<unsigned>(
    (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
       .count()) &
    0x7FFFFFFF);

  // Define tupling
  checker.accumulate<Checker::Subdetector::SciFi>(
    *first<host_mc_events_t>(arguments),
    composites,
    event_list,
    [slice_idx, time, host_polarity, &host_dumpping_objects](
      const unsigned event_idx,
      const Checker::Composite& composite,
      const CompositeDumper::MatchedComposite_t& matched,
      CompositeDumper::Columns_t& data) {
      // constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();

      data["slice_idx"] = slice_idx;
      data["event_idx"] = event_idx;
      data["time"] = time;
      data["polarity"] = host_polarity;

      // Fetch dumpper
      const auto idx = composite.idx;
      const auto& dumpper = host_dumpping_objects[idx];
      unsigned nChildren = dumpper.nChildren;

      // Fill reconstructed info
      data["ievent"] = dumpper.ievent;
      data["composite_x"] = dumpper.x;
      data["composite_y"] = dumpper.y;
      data["composite_z"] = dumpper.z;
      data["composite_c00"] = dumpper.c00;
      data["composite_c11"] = dumpper.c11;
      data["composite_c22"] = dumpper.c22;
      data["composite_px"] = dumpper.px;
      data["composite_py"] = dumpper.py;
      data["composite_pz"] = dumpper.pz;
      data["composite_pvx"] = dumpper.pvx;
      data["composite_pvy"] = dumpper.pvy;
      data["composite_pvz"] = dumpper.pvz;

      data["composite_recMass"] = dumpper.recMass;
      for (unsigned i = 0; i < nChildren; i++) {
        for (unsigned j = 0; j < nChildren; j++) {
          data["composite_doca" + std::to_string(i) + std::to_string(j)] = dumpper.doca[i][j];
        }
      }

      data["composite_fd"] = dumpper.fd;
      data["composite_ctau"] = dumpper.ctau;
      data["composite_dz"] = dumpper.dz;
      data["composite_drho"] = dumpper.drho;
      data["composite_eta"] = dumpper.eta;
      data["composite_mcor"] = dumpper.mcor;
      data["composite_minip"] = dumpper.minip;
      data["composite_minp"] = dumpper.minp;
      data["composite_minpt"] = dumpper.minpt;
      data["composite_maxp"] = dumpper.maxp;
      data["composite_maxpt"] = dumpper.maxpt;
      data["composite_dira"] = dumpper.dira;
      data["composite_IP"] = dumpper.bpv_ip;

      data["composite_armenteros_x"] = dumpper.armenteros_x;
      data["composite_armenteros_y"] = dumpper.armenteros_y;
      data["composite_quality"] = dumpper.quality;

      std::vector<const MCParticle*> mcps(nChildren);
      const auto category = matched.category;

      data["category"] = category;
      for (unsigned i = 0; i < nChildren; i++) {
        mcps[i] = matched.tracks[i];
        data["d" + std::to_string(i) + "_px"] = dumpper.children[i].px;
        data["d" + std::to_string(i) + "_py"] = dumpper.children[i].py;
        data["d" + std::to_string(i) + "_pz"] = dumpper.children[i].pz;
        data["d" + std::to_string(i) + "_pt"] = dumpper.children[i].pt;
        data["d" + std::to_string(i) + "_p"] = dumpper.children[i].p;
        data["d" + std::to_string(i) + "_eta"] = dumpper.children[i].eta;
        data["d" + std::to_string(i) + "_rho"] = dumpper.children[i].rho;
        data["d" + std::to_string(i) + "_ipchi2"] = dumpper.children[i].ip_chi2;
        data["d" + std::to_string(i) + "_chi2"] = dumpper.children[i].chi2;
        data["d" + std::to_string(i) + "_ndf"] = dumpper.children[i].ndof;
        data["d" + std::to_string(i) + "_chi2ndof"] = dumpper.children[i].chi2ndof;

        data["d" + std::to_string(i) + "_is_muon"] = dumpper.children[i].is_muon;
        data["d" + std::to_string(i) + "_is_electron"] = dumpper.children[i].is_electron;
      }

      if (category == 0 || category == 1) {
        data["true_x"] = mcps[0]->ovtx_x;
        data["true_y"] = mcps[0]->ovtx_y;
        data["true_z"] = mcps[0]->ovtx_z;
        data["true_px"] = std::accumulate(
          mcps.begin(), mcps.end(), 0.0f, [](float sum, const auto* mcp) { return sum + mcp->pt * cosf(mcp->phi); });
        data["true_py"] = std::accumulate(
          mcps.begin(), mcps.end(), 0.0f, [](float sum, const auto* mcp) { return sum + mcp->pt * sinf(mcp->phi); });
        data["true_pz"] = std::accumulate(
          mcps.begin(), mcps.end(), 0.0f, [](float sum, const auto* mcp) { return sum + mcp->pt * sinhf(mcp->eta); });
        data["true_pid"] = mcps[0]->mother_pid;
        data["true_key"] = mcps[0]->motherKey;
        data["true_fromSignal"] = mcps[0]->fromSignal;
        data["true_OriginKey"] = mcps[0]->DecayOriginMother_key;
      }
      else {
        data["true_x"] = std::numeric_limits<float>::quiet_NaN();
        data["true_y"] = std::numeric_limits<float>::quiet_NaN();
        data["true_z"] = std::numeric_limits<float>::quiet_NaN();
        data["true_px"] = std::numeric_limits<float>::quiet_NaN();
        data["true_py"] = std::numeric_limits<float>::quiet_NaN();
        data["true_pz"] = std::numeric_limits<float>::quiet_NaN();
        data["true_pid"] = 0;
        data["true_fromSignal"] = false;
        data["true_key"] = -1;
        data["true_OriginKey"] = -1;
      }

      auto fill_mcp = [&data](const std::string prefix, const MCParticle* mcp) {
        if (mcp) {
          data[prefix + "_true_hasVelo"] = mcp->hasVelo;
          data[prefix + "_true_hasUT"] = mcp->hasUT;
          data[prefix + "_true_hasSciFi"] = mcp->hasSciFi;
          data[prefix + "_true_isLong"] = mcp->isLong;
          data[prefix + "_true_isDown"] = mcp->isDown;
          data[prefix + "_true_fromSignal"] = mcp->fromSignal;
          data[prefix + "_true_pid"] = mcp->pid;
          data[prefix + "_true_key"] = mcp->key;
          data[prefix + "_true_mother_pid"] = mcp->mother_pid;
          data[prefix + "_true_origin_pid"] = mcp->DecayOriginMother_pid;
          data[prefix + "_true_mother_key"] = mcp->motherKey;
          data[prefix + "_true_origin_key"] = mcp->DecayOriginMother_key;
          data[prefix + "_true_p"] = mcp->p;
          data[prefix + "_true_pt"] = mcp->pt;
          data[prefix + "_true_eta"] = mcp->eta;
          data[prefix + "_true_phi"] = mcp->phi;
          data[prefix + "_true_px"] = mcp->pt * cosf(mcp->phi);
          data[prefix + "_true_py"] = mcp->pt * sinf(mcp->phi);
          data[prefix + "_true_pz"] = mcp->pt * sinhf(mcp->eta);
        }
        else {
          data[prefix + "_true_hasVelo"] = false;
          data[prefix + "_true_hasUT"] = false;
          data[prefix + "_true_hasSciFi"] = false;
          data[prefix + "_true_isLong"] = false;
          data[prefix + "_true_isDown"] = false;
          data[prefix + "_true_fromSignal"] = false;
          data[prefix + "_true_pid"] = 0;
          data[prefix + "_true_key"] = 0;
          data[prefix + "_true_mother_pid"] = 0;
          data[prefix + "_true_origin_pid"] = 0;
          data[prefix + "_true_mother_key"] = -1;
          data[prefix + "_true_origin_key"] = -1;
          data[prefix + "_true_p"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_pt"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_eta"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_phi"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_px"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_py"] = std::numeric_limits<float>::quiet_NaN();
          data[prefix + "_true_pz"] = std::numeric_limits<float>::quiet_NaN();
        }
      };

      unsigned i = 0;
      for (const auto& mcp : mcps) {
        fill_mcp("d" + std::to_string(i), mcp);
        i++;
      }

      // For partial ghost case, we assign the vertex info from the non-ghost track

      if (category == 3) {
        const auto mcp_aux2 = mcps[0] ? mcps[0] : mcps[1];
        const auto mcp_aux1 = mcp_aux2 ? mcp_aux2 : mcps[2];
        const auto mcp = mcp_aux1 ? mcp_aux1 : mcps[3];
        data["true_pid"] = mcp->pid;
        data["true_key"] = mcp->motherKey;
        data["true_fromSignal"] = mcp->fromSignal;
        data["true_OriginKey"] = mcp->DecayOriginMother_key;
      }
    },
    // Composite Filter
    [](
      const unsigned, const Checker::Composite&, const CompositeDumper::MatchedComposite_t& // matched
    ) { return true; },
    // Event filter (Has reconstructible downstream tracks)
    [](const unsigned, const MCEvent&) { return true; });
}
