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
#include "RichPhoton.cuh"

namespace rich_pix2track {
  struct Parameters {
    // We iterate over photons, and for each photon we need its pixelIdx and its owning track.
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    HOST_INPUT(host_number_of_photons_t, unsigned) host_number_of_photons;
    HOST_INPUT(host_number_of_pixels_t, unsigned) host_number_of_pixels;

    DEVICE_INPUT(dev_offsets_rich_photons_t, unsigned) dev_offsets_rich_photons;
    DEVICE_INPUT(dev_rich_photons_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons;

    //  map pixels to  tracks
    DEVICE_OUTPUT(dev_pix2track_offsets_t, unsigned) dev_pix2track_offsets;
    DEVICE_OUTPUT(dev_pix2track_t, unsigned) dev_pix2track;
    DEVICE_OUTPUT(dev_pix2photon_t, unsigned) dev_pix2photon;
  };

  struct rich_pix2track_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
  };
} // namespace rich_pix2track
