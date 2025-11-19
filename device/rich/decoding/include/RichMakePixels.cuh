/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
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
#include <RichPixel.cuh>

namespace rich_make_pixels {
  struct Parameters {
    HOST_INPUT(host_rich_total_number_of_hits_t, unsigned) host_rich_total_number_of_hits;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_smart_ids_t, Allen::Rich::Decoding::SmartID) dev_smart_ids;
    DEVICE_INPUT(dev_rich_hit_offsets_t, unsigned) dev_rich_hit_offsets;
    DEVICE_OUTPUT(dev_rich_pixels_t, Allen::Rich::PixelReco::Pixel) dev_rich_pixels;
  };

  struct rich_make_pixels_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {128, 1, 1}, "block dimensions"};
    Allen::Property<unsigned> m_current_rich {this, "current_rich", 1, "current rich"};
  };
} // namespace rich_make_pixels
