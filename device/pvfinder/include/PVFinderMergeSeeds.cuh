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
#include "PVFinderConstants.cuh"

// PV seeds from PVFinder inside its z range, completed with another finder's
// seeds (the beamline PV finder's) outside it, for one PV association and fit
// over the whole z range. Both inputs and the output use the layout of
// pv_beamline_peak: PV::max_number_vertices z values per event and a count.
namespace pvfinder_merge_seeds {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_pvfinder_zpeaks_t, float) dev_pvfinder_zpeaks;
    DEVICE_INPUT(dev_pvfinder_number_of_zpeaks_t, unsigned) dev_pvfinder_number_of_zpeaks;
    DEVICE_INPUT(dev_other_zpeaks_t, float) dev_other_zpeaks;
    DEVICE_INPUT(dev_other_number_of_zpeaks_t, unsigned) dev_other_number_of_zpeaks;
    DEVICE_OUTPUT(dev_zpeaks_t, float) dev_zpeaks;
    DEVICE_OUTPUT(dev_number_of_zpeaks_t, unsigned) dev_number_of_zpeaks;
  };

  __global__ void pvfinder_merge_seeds(Parameters, const float z_min, const float z_max);

  struct pvfinder_merge_seeds_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {32, 1, 1}, "block dimensions"};
  };
} // namespace pvfinder_merge_seeds
