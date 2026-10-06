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

#pragma once

#include "SciFiEventModel.cuh"
#include "States.cuh"
#include "AlgorithmTypes.cuh"
#include "ParticleTypes.cuh"
#include "CheckerTracks.cuh"
#include "CheckerInvoker.h"
#include "CompositeDumper.h"
#include "DumppingSVs.cuh"
#include "VertexDefinitions.cuh"
#include "ODINBank.cuh"

namespace vertexing_validator {
  struct Parameters {

    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // MC
    HOST_INPUT(host_mc_events_t, const MCEvents*) host_mc_events;

    // output from vertexing
    HOST_INPUT(host_number_of_vertices_t, unsigned) host_number_of_vertices;
    DEVICE_INPUT(dev_offset_vertices_t, unsigned) dev_offset_vertices;
    DEVICE_INPUT(dev_multi_event_composites_view_t, Allen::Views::Physics::MultiEventCompositeParticles)
    dev_multi_event_composites_view;
    DEVICE_INPUT(dev_odin_data_t, ODINData) dev_odin_data;

    // Output
    DEVICE_OUTPUT(dev_checker_composites_t, Checker::Composite) dev_checker_composites;
    DEVICE_OUTPUT(dev_dumpping_objects_t, Allen::DumppingSVs::genericSV) dev_dumpping_objects;
  };

  __global__ void vertexing_validator(Parameters parameters);

  struct vertexing_validator_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
    Allen::Property<std::string> m_root_output_filename {
      this,
      "root_output_filename",
      "PrCheckerPlots.root",
      "root output filename"};
  };
} // namespace vertexing_validator
