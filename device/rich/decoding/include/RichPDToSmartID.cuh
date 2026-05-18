/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
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
#include <RichSmartID.cuh>

namespace rich_pd_to_smartid {
  struct Parameters {
    DEVICE_INPUT(dev_rich_pd_offsets_t, unsigned) dev_rich_pd_offsets;
    HOST_INPUT(host_rich_total_number_of_hits_t, unsigned) host_rich_total_number_of_hits;
    DEVICE_INPUT(dev_pd_pixels_t, uint64_t) dev_pd_pixels;
    DEVICE_OUTPUT(dev_smart_ids_t, Allen::Rich::Decoding::SmartID) dev_smart_ids;
  };

  struct rich_pd_to_smartid_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<unsigned> m_current_rich {this, "current_rich", 1, "current rich"};
  };
} // namespace rich_pd_to_smartid
