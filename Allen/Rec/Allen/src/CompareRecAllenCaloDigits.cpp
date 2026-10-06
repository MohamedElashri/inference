/***************************************************************************** \
 * (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <string>
#include <vector>

// Gaudi
#include "GaudiAlg/Consumer.h"
#include "GaudiAlg/Transformer.h"

// Allen
#include "CaloDigit.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

// Calorimeter
#include <Event/CaloDigit.h>
#include <Event/CaloDigits_v2.h>

using AllenCaloDigits = std::vector<CaloDigit>;

// ==================================================================
//  Multi-event converter: device buffer → per-event CaloDigit vectors
// ==================================================================

class ConvertAllenCaloDigits final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<
      std::tuple<AllenCaloDigits>(const Allen::device_buffer<CaloDigit>&, const Allen::device_buffer<unsigned>&)> {

public:
  ConvertAllenCaloDigits(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"ecal_digits", ""}, KeyValue {"ecal_digit_offsets", ""}},
      {KeyValue {"AllenCaloDigits", ""}})
  {}

  std::tuple<std::vector<AllenCaloDigits>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<CaloDigit>& dev_digits,
    const Allen::device_buffer<unsigned>& dev_offsets) const override
  {
    auto h_digits = dev_digits.to_host();
    auto h_offsets = dev_offsets.to_host();

    const unsigned n_events = h_offsets.size() - 1;

    std::vector<AllenCaloDigits> all_digits;
    all_digits.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned begin = h_offsets[evt];
      const unsigned end = h_offsets[evt + 1];
      all_digits.emplace_back(h_digits.data() + begin, h_digits.data() + end);
    }

    return std::make_tuple(std::move(all_digits));
  }
};

DECLARE_COMPONENT(ConvertAllenCaloDigits)

// ==================================================================
//  Single-event comparison: AllenCaloDigits  vs  Rec Calo Digits
// ==================================================================

class CompareRecAllenCaloDigits final
  : public Gaudi::Functional::Consumer<void(const AllenCaloDigits&, LHCb::Event::Calo::Digits const&)> {

public:
  CompareRecAllenCaloDigits(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const AllenCaloDigits& allenDigits, LHCb::Event::Calo::Digits const& lhcbDigits) const override;
};

DECLARE_COMPONENT(CompareRecAllenCaloDigits)

CompareRecAllenCaloDigits::CompareRecAllenCaloDigits(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"AllenCaloDigits", ""}, KeyValue {"EcalDigits", LHCb::CaloDigitLocation::Ecal}})
{}

void CompareRecAllenCaloDigits::operator()(
  const AllenCaloDigits& allenDigits,
  LHCb::Event::Calo::Digits const& lhcbDigits) const
{
  for (const auto& d : lhcbDigits) {
    LHCb::Detector::Calo::Index idx {d.cellID()};
    unsigned digit_index = unsigned {idx};

    if (digit_index >= allenDigits.size() || d.adc() != allenDigits[digit_index].adc) {
      std::stringstream msg;
      error() << "LHCb digit at " << unsigned {idx} << " ADC " << d.adc() << " != Allen digit at " << digit_index
              << " ADC " << (digit_index < allenDigits.size() ? allenDigits[digit_index].adc : -1) << endmsg;
    }
  }
}
