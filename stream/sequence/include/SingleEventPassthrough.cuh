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

#include <vector>
#include <span>
#include <Store.cuh>
#include <InputReader.h>
#include <InputProvider.h>
#include <Event/RawBank.h>
#include <BankTypes.h>
#include <OutputManager.h>

#ifndef ALLEN_STANDALONE
#include <GaudiKernel/Service.h>
#include <Gaudi/Accumulators.h>
#endif

struct SingleEventPassthrough {
  SingleEventPassthrough(unsigned tck, unsigned task_id, unsigned passthrough_rbs, bool do_checksum) :
    m_tck {tck}, m_task_id {task_id}, m_passthrough_rbs {passthrough_rbs}, m_do_checksum {do_checksum}
  {
    init();
  }

  SingleEventPassthrough(const ConfigurationReader::Params& configuration)
  {
    // load configuration relevant to the large-event passthrough
    if (configuration.find("dec_reporter") != configuration.end()) {
      auto decrep_config = configuration.find("dec_reporter")->second;
      for (auto& [key, m] : {std::tuple {"tck", std::ref(m_tck)}, std::tuple {"task_id", std::ref(m_task_id)}}) {
        if (decrep_config.find(key) != decrep_config.end()) {
          m.get() = decrep_config[key].template get<unsigned>();
        }
      }
    }

    if (configuration.find("host_routingbits_writer") != configuration.end()) {
      auto rb_config = configuration.find("host_routingbits_writer")->second;
      if (rb_config.find("routingbit_map") != rb_config.end()) {
        for (auto [expr, bit] : rb_config["routingbit_map"].template get<std::map<std::string, unsigned>>()) {
          std::smatch result;
          if (std::regex_match(m_passthrough_line, result, std::regex {expr})) {
            m_passthrough_rbs |= 1u << bit;
          }
        }
      }
    }

    if (configuration.find("host_output_handler") != configuration.end()) {
      auto handler_config = configuration.find("host_output_handler")->second;
      if (handler_config.find("do_checksum") != handler_config.end()) {
        m_do_checksum = handler_config["do_checksum"].template get<bool>();
      }
    }
    init();
  }

  void write(
    size_t const slice_index,
    unsigned const start_event,
    IInputProvider const* input_provider,
    int producer_id) const;

  void write(size_t const slice_index, unsigned const start_event, IInputProvider const* input_provider) const
  {
    write(slice_index, start_event, input_provider, OutputManager::get()->n_producers() - 1);
  }

#ifndef ALLEN_STANDALONE
  void activateMonitoring(Service* svc);
#endif

private:
  void init();

  std::vector<char> m_banks;

  unsigned m_tck {0u};
  unsigned m_task_id {0u};
  unsigned m_passthrough_rbs {0u};
  bool m_do_checksum {false};

  const std::string m_passthrough_line = "Hlt1PassthroughLargeEvent";
  const unsigned m_passthrough_key = 0xe7682884;

#ifndef ALLEN_STANDALONE
  std::unique_ptr<Gaudi::Accumulators::Counter<>> m_npassthrough;
#endif
};
