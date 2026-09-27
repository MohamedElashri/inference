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
#include "PVFinderConstants.cuh"

// Primary-vertex z seeds from the PVFinder KDE. The output has the layout of
// pv_beamline_peak's, so the beamline PV track association and fit
// (pv_beamline_calculate_denom, pv_beamline_multi_fitter, pv_beamline_cleanup)
// run on these seeds unchanged.
//
// The peak finder is pv-finder's pv_locations_updated: a peak is a run of
// consecutive bins at or above `threshold`, split in two where the KDE rises
// again after a drop of more than split_min_drop and split_min_ratio between
// two bins; it is kept when it has at least `min_width` bins and their sum is
// at least `integral_threshold`. The seed is the KDE-weighted mean z of its
// bins (bin centres).
namespace pvfinder_peak {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    // [events][PVFinderConstants::KDE::n_bins], from pvfinder_unet.
    DEVICE_INPUT(dev_pvfinder_kde_output_t, float) dev_pvfinder_kde_output;
    DEVICE_OUTPUT(dev_zpeaks_t, float) dev_zpeaks;
    DEVICE_OUTPUT(dev_number_of_zpeaks_t, unsigned) dev_number_of_zpeaks;
  };

  // Largest block_dim.x (size of the kernel's per-thread shared arrays).
  static constexpr unsigned max_block_dim = 256;

  __global__ void pvfinder_peak(
    Parameters,
    const float threshold,
    const float integral_threshold,
    const unsigned min_width,
    const bool split_peaks);

  struct pvfinder_peak_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {32, 1, 1}, "block dimensions"};
    Allen::Property<float> m_threshold {
      this,
      "threshold",
      PVFinderConstants::Peak::threshold,
      "minimum KDE value of a bin in a peak"};
    Allen::Property<float> m_integral_threshold {
      this,
      "integral_threshold",
      PVFinderConstants::Peak::integral_threshold,
      "minimum sum of the KDE over a peak's bins"};
    Allen::Property<unsigned> m_min_width {
      this,
      "min_width",
      PVFinderConstants::Peak::min_width,
      "minimum number of bins in a peak"};
    Allen::Property<bool> m_split_peaks {
      this,
      "split_peaks",
      true,
      "split a run of bins above threshold where the KDE rises again after a "
      "significant drop (pv_locations_updated; false = pv_locations)"};
    // Validation dump: when non-empty, the first call writes the seeds of every
    // event of the slice to <dump_validation>/allen_zpeaks.bin: uint32 magic,
    // uint32 number of events, then per event uint32 number of seeds and
    // float32[PV::max_number_vertices] seeds (0 seeds for events not in the list).
    Allen::Property<std::string> m_dump_dir {
      this,
      "dump_validation",
      "",
      "if non-empty, dump the seeds of the first slice to this directory"};
    mutable bool m_dump_done = false;
  };
} // namespace pvfinder_peak
