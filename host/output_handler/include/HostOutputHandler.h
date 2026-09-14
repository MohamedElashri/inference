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

#include "BackendCommon.h"
#include "AlgorithmTypes.cuh"
#include <InputProvider.h>
#include <HltDecReport.cuh>
#include <RoutingBitsDefinition.h>
#include <TAE.h>
#include <Event/RawBank.h>
#include <BankTypes.h>
#include <mdf_header.hpp>
#include <raw_helpers.hpp>
#include <read_mdf.hpp>
#include <write_mdf.hpp>
#include <OutputManager.h>

#ifndef ALLEN_STANDALONE
#include <GaudiKernel/Service.h>
#include <Gaudi/Accumulators.h>
#endif

namespace host_output_handler {
  struct HLT1Outputs {
    std::span<bool const> selected_events;
    std::span<unsigned const> dec_reports;
    std::span<unsigned const> routing_bits;
    std::span<unsigned const> sel_reports;
    std::span<unsigned const> sel_reports_offsets;
    std::span<unsigned const> lumi_summaries;
    std::span<unsigned const> lumi_summary_offsets;
    std::span<TAE::TAEEvent const> tae_events;
  };

  struct OutputSizes {
    std::vector<size_t> input;
    std::vector<size_t> hlt;
    std::vector<size_t> tae;
    OutputSizes(size_t s)
    {
      for (auto* sizes : {&input, &hlt, &tae}) {
        sizes->resize(s);
      }
    }
  };

  struct Parameters {
    HOST_INPUT(host_global_decision_t, bool) selected_events;
    HOST_INPUT(host_dec_reports_t, unsigned) dec_reports;
    HOST_INPUT(host_routingbits_t, unsigned) routing_bits;
    HOST_INPUT(host_sel_reports_t, unsigned) sel_reports;
    HOST_INPUT(host_selrep_offsets_t, unsigned) sel_reports_offsets;
    HOST_INPUT(host_lumi_summaries_t, unsigned) lumi_summaries;
    HOST_INPUT(host_lumi_summary_offsets_t, unsigned) lumi_summaries_offsets;
    HOST_INPUT(host_tae_events_t, TAE::TAEEvent) tae_events;
  };

  struct host_output_handler_t : public HostAlgorithm, Parameters {
    void init();

    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const {}

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    std::span<char> buffer(const Allen::Context& ctx, size_t size) const;
    void write_buffer(const Allen::Context& ctx) const;

    std::tuple<bool, size_t> output_single_events(
      size_t const slice_index,
      size_t const event_offset,
      HLT1Outputs const& outputs,
      IInputProvider const* input_provider,
      const Allen::Context& ctx) const;

    std::tuple<bool, size_t> output_tae_events(
      size_t const slice_index,
      size_t const event_offset,
      HLT1Outputs const& outputs,
      IInputProvider const* input_provider,
      const Allen::Context& ctx) const;

    void event_sizes(
      OutputSizes& sizes,
      size_t const slice_index,
      HLT1Outputs const& outputs,
      IInputProvider const* input_provider,
      std::span<unsigned> const& selected_events,
      unsigned const start_event) const;

    void add_checksum(LHCb::MDFHeader*, std::span<char>) const;

    size_t add_banks(
      HLT1Outputs const& outputs,
      IInputProvider const* input_provider,
      unsigned const slice_index,
      unsigned const start_event,
      unsigned const event_number,
      unsigned const input_size,
      std::span<char> event_span) const;

    Allen::Property<size_t> m_output_batch_size {this, "output_batch_size", 10, "output batch size"};
    Allen::Property<bool> m_checksum {this, "do_checksum", false, "do mdf checksum"};

#ifndef ALLEN_STANDALONE
    std::unique_ptr<Gaudi::Accumulators::Counter<>> m_nprocessed;
    std::unique_ptr<Gaudi::Accumulators::Counter<>> m_noutput;
    std::unique_ptr<Gaudi::Accumulators::AveragingCounter<>> m_batch_size;
    std::unique_ptr<Gaudi::Accumulators::AveragingCounter<>> m_nbatches;
    std::unique_ptr<Gaudi::Accumulators::Counter<>> m_ntae;
#endif
  };
} // namespace host_output_handler
