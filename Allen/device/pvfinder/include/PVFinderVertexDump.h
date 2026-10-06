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
#pragma once

#include "AlgorithmTypes.cuh"
#include "PV_Definitions.cuh"
#include <span>

// Writes the reconstructed primary vertices and the MC primary vertices of
// every event in the event list to a binary file, for event-by-event
// comparisons of PV reconstructions.
// Several instances (one per vertex collection) can write in one job; the
// records carry the batch (this instance's call count) and the event number
// in the batch to line them up, which identifies the same event in every
// instance of a single-stream job (-t 1).
//
// Record per event, little endian:
//   uint32 batch, uint32 event (first event of the slice + index in it),
//   uint32 n_rec, uint32 n_mc,
//   n_rec x float32[9]: x, y, z, cov00, cov11, cov22, chi2, ndof, nTracks
//   n_mc  x float64[4]: x, y, z, number of reconstructible tracks
namespace pvfinder_pv_dump {
  struct Parameters {
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_multi_final_vertices_t, PV::Vertex) dev_multi_final_vertices;
    DEVICE_INPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
    HOST_INPUT(host_mc_pv_banks_t, std::span<const char>) host_mc_pv_banks;
    HOST_INPUT(host_mc_pv_offsets_t, std::span<const unsigned>) host_mc_pv_offsets;
    HOST_INPUT(host_mc_pv_sizes_t, std::span<const unsigned>) host_mc_pv_sizes;
  };

  struct pvfinder_pv_dump_t : public ValidationAlgorithm, Parameters {
    inline void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const {}

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<std::string> m_output_filename {
      this,
      "output_filename",
      "pvs.bin",
      "file the vertices are written to (truncated at the first call)"};
    mutable unsigned m_batch = 0;
  };
} // namespace pvfinder_pv_dump
