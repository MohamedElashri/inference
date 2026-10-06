/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "MakeSubBanks.cuh"
#include "CaloConstants.cuh"
#include "HltSubBanks.cuh"

INSTANTIATE_ALGORITHM(make_subbanks::make_subbanks_t)

void make_subbanks::make_subbanks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rb_substr_t>(arguments, first<host_substr_bank_size_t>(arguments));
  set_size<dev_rb_hits_t>(arguments, first<host_hits_bank_size_t>(arguments));
  set_size<dev_rb_stdinfo_t>(arguments, first<host_stdinfo_bank_size_t>(arguments));
  set_size<dev_rb_objtyp_t>(arguments, first<host_objtyp_bank_size_t>(arguments));
}

void make_subbanks::make_subbanks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_rb_substr_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_hits_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_stdinfo_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_objtyp_t>(arguments, 0, context);
  global_function(make_rb_substr)(dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
    arguments, first<host_number_of_events_t>(arguments), m_max_children_per_object);

  global_function(make_rb_hits)(dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
    arguments, m_max_children_per_object);
}

__global__ void make_subbanks::make_rb_substr(
  make_subbanks::Parameters parameters,
  const unsigned number_of_events,
  const unsigned n_children)
{

  for (unsigned event_number = blockIdx.x * blockDim.x + threadIdx.x; event_number < number_of_events;
       event_number += blockDim.x * gridDim.x) {

    const unsigned n_sels = parameters.dev_sel_count[event_number];
    if (n_sels == 0) continue;

    unsigned* event_rb_substr = parameters.dev_rb_substr + parameters.dev_rb_substr_offsets[event_number];
    const unsigned event_rb_substr_size =
      parameters.dev_rb_substr_offsets[event_number + 1] - parameters.dev_rb_substr_offsets[event_number];
    const unsigned n_lines = parameters.dev_number_of_active_lines[0];
    const unsigned* line_object_offsets = parameters.dev_max_objects_offsets + n_lines * event_number;
    const unsigned selected_object_offset = n_children * line_object_offsets[0];
    const unsigned n_tracks = parameters.dev_unique_track_count[event_number];
    const unsigned n_calos = parameters.dev_unique_calo_count[event_number];
    const unsigned n_svs = parameters.dev_unique_sv_count[event_number];

    const auto event_track_ptrs = parameters.dev_basic_particle_ptrs + selected_object_offset;
    const auto event_calo_ptrs = parameters.dev_neutral_basic_particle_ptrs + selected_object_offset;
    const auto event_sv_ptrs = parameters.dev_composite_particle_ptrs + selected_object_offset;
    const unsigned* event_unique_track_list = parameters.dev_unique_track_list + selected_object_offset;
    const unsigned* event_unique_calo_list = parameters.dev_unique_calo_list + selected_object_offset;
    const unsigned* event_unique_sv_list = parameters.dev_unique_sv_list + selected_object_offset;

    const unsigned* event_candidate_offsets =
      parameters.dev_candidate_offsets + event_number * parameters.dev_number_of_active_lines[0];
    const unsigned* event_sel_list = parameters.dev_sel_list + event_number * parameters.dev_number_of_active_lines[0];

    // Make the substructure bank.
    make_rb_substr_bank(
      event_rb_substr,
      event_rb_substr_size,
      n_children,
      n_sels,
      n_tracks,
      n_calos,
      n_svs,
      parameters.dev_substr_sel_size[event_number],
      parameters.dev_substr_sv_size[event_number],
      parameters.dev_substr_track_size[event_number],
      parameters.dev_multi_event_particle_containers.data(),
      event_track_ptrs,
      event_calo_ptrs,
      event_sv_ptrs,
      line_object_offsets,
      event_unique_track_list,
      event_unique_calo_list,
      event_unique_sv_list,
      event_candidate_offsets,
      event_sel_list,
      parameters.dev_sel_track_indices.data(),
      parameters.dev_sel_calo_indices.data(),
      parameters.dev_sel_sv_indices.data(),
      parameters.dev_track_duplicate_map.data(),
      parameters.dev_calo_duplicate_map.data(),
      parameters.dev_sv_duplicate_map.data());

    // Create the ObjTyp subbank.
    const unsigned objtyp_offset = parameters.dev_rb_objtyp_offsets[event_number];
    const unsigned objtyp_size = parameters.dev_rb_objtyp_offsets[event_number + 1] - objtyp_offset;
    const unsigned n_objtyps = objtyp_size - 1;
    unsigned* event_rb_objtyp = parameters.dev_rb_objtyp + objtyp_offset;
    make_rb_objtyp_bank(event_rb_objtyp, n_objtyps, n_sels, n_tracks, n_calos, n_svs);

    // Create the StdInfo bank.
    unsigned* event_rb_stdinfo = parameters.dev_rb_stdinfo + parameters.dev_rb_stdinfo_offsets[event_number];
    const unsigned stdinfo_size =
      parameters.dev_rb_stdinfo_offsets[event_number + 1] - parameters.dev_rb_stdinfo_offsets[event_number];
    make_rb_stdinfo_bank(
      event_rb_stdinfo,
      stdinfo_size,
      n_sels,
      n_tracks,
      n_calos,
      n_svs,
      event_sel_list,
      event_unique_track_list,
      event_unique_calo_list,
      event_unique_sv_list,
      event_track_ptrs,
      event_calo_ptrs,
      event_sv_ptrs);
  }
}

__global__ void make_subbanks::make_rb_hits(make_subbanks::Parameters parameters, const unsigned n_children)
{
  const unsigned event_number = blockIdx.x;

  // Hit "sequence" here refers to the hits associated to a single track. See
  // https://gitlab.cern.ch/lhcb/LHCb/-/blob/master/Hlt/HltDAQ/HltDAQ/HltSelRepRBHits.h
  const unsigned n_hit_sequences = parameters.dev_unique_track_count[event_number];
  unsigned* event_rb_hits = parameters.dev_rb_hits + parameters.dev_rb_hits_offsets[event_number];
  const unsigned bank_info_size = 1 + (n_hit_sequences / 2);
  const unsigned n_lines = parameters.dev_number_of_active_lines[0];
  const unsigned track_offset = n_children * parameters.dev_max_objects_offsets[event_number * n_lines];

  // Run sequentially over tracks and in parallel over hits. There will usually
  // only be ~1 selected track anyway.
  unsigned seq_begin = bank_info_size;
  for (unsigned i_seq = 0; i_seq < n_hit_sequences; i_seq++) {
    const unsigned track_index = parameters.dev_unique_track_list[track_offset + i_seq];
    const Allen::Views::Physics::BasicParticle* track = parameters.dev_basic_particle_ptrs[track_offset + track_index];
    const unsigned n_hits = track->number_of_ids();
    unsigned* hits_insert_pointer = event_rb_hits + seq_begin;

    for (unsigned i_hit = threadIdx.x; i_hit < n_hits; i_hit += blockDim.x) {
      hits_insert_pointer[i_hit] = track->id(i_hit);
    }

    if (threadIdx.x == 0) {
      const unsigned seq_end = seq_begin + n_hits;
      unsigned i_word = (i_seq + 1) / 2;
      unsigned i_part = (i_seq + 1) % 2;
      unsigned bits = i_part * 16;
      unsigned mask = 0xFFFFL << bits;
      event_rb_hits[i_word] = (event_rb_hits[i_word] & ~mask) | (seq_end << bits);
    }

    seq_begin += n_hits;
  }

  if (threadIdx.x == 0) {
    event_rb_hits[0] = (event_rb_hits[0] & ~0xFFFFL) | n_hit_sequences;
  }
}
