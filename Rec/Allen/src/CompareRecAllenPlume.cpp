/***************************************************************************** \
 * (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <string>
#include <vector>
#include <ostream>
#include <map>

// Gaudi
#include "GaudiAlg/Consumer.h"
#include "Gaudi/Accumulators.h"

// Allen
#include "Plume.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

// PLUME
#include <Event/PlumeAdc.h>

// ==================================================================
//  Multi-event converter: device buffer → per-event Plume_ structs
// ==================================================================

class ConvertAllenPlume final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<Plume_>(const Allen::device_buffer<Plume_>&)> {

public:
  ConvertAllenPlume(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(name, pSvcLocator, {KeyValue {"plume_digits_Allen", ""}}, {KeyValue {"PlumeDigit", ""}})
  {}

  std::tuple<std::vector<Plume_>> operator()(const EventContext& /*ctx*/, const Allen::device_buffer<Plume_>& dev_plume)
    const override
  {
    auto h = dev_plume.to_host();
    std::vector<Plume_> out(h.size());
    for (unsigned i = 0; i < h.size(); ++i)
      out[i] = h[i];
    return std::make_tuple(std::move(out));
  }
};

DECLARE_COMPONENT(ConvertAllenPlume)

// ==================================================================
//  Single-event comparison: Plume_  vs  Rec PlumeAdcs
// ==================================================================

class CompareRecAllenPlume final : public Gaudi::Functional::Consumer<void(const Plume_&, LHCb::PlumeAdcs const&)> {

public:
  CompareRecAllenPlume(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const Plume_& allenDigits, LHCb::PlumeAdcs const& lhcbDigits) const override;

private:
  Gaudi::Property<int> m_pedestalOffset {this, "PedestalOffset", 256, "Offset to subtract from raw ADC counts."};
  std::map<unsigned int, unsigned int> m_map_reversed;
  std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_map_reversed_time;

  mutable Gaudi::Accumulators::Counter<> m_matched {this, "Matched HLT1/HLT2 ADC values"};
  mutable Gaudi::Accumulators::Counter<> m_error {this, "Not Matched HLT1/HLT2 ADC values"};
  mutable Gaudi::Accumulators::Counter<> m_matched_ovr {this, "Matched HLT1/HLT2 over-threshold bits"};
  mutable Gaudi::Accumulators::Counter<> m_error_ovr {this, "Not Matched HLT1/HLT2 over-threshold bits"};
  mutable Gaudi::Accumulators::Counter<> m_matched_time {this, "Matched HLT1/HLT2 TIME ADC values"};
  mutable Gaudi::Accumulators::Counter<> m_error_time {this, "Not Matched HLT1/HLT2 TIME ADC values"};
};

DECLARE_COMPONENT(CompareRecAllenPlume)

CompareRecAllenPlume::CompareRecAllenPlume(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"PlumeDigit", ""}, KeyValue {"plume_digits_Moore", LHCb::PlumeAdcLocation::Default}})
{
  std::transform(
    LHCb::Plume::lumiFebToLogicalChannel.begin(),
    LHCb::Plume::lumiFebToLogicalChannel.end(),
    std::inserter(m_map_reversed, m_map_reversed.end()),
    [](const auto& pair) { return std::make_pair(pair.second, pair.first); });

  std::for_each(
    LHCb::Plume::timingFebToLogicalChannel.begin(),
    LHCb::Plume::timingFebToLogicalChannel.end(),
    [&](const auto& febMap) {
      std::transform(
        febMap.begin(),
        febMap.end(),
        std::inserter(m_map_reversed_time, m_map_reversed_time.end()),
        [](const auto& entry) { return std::make_pair(entry.second, entry.first); });
    });
}

void CompareRecAllenPlume::operator()(const Plume_& allenDigits, LHCb::PlumeAdcs const& lhcbDigits) const
{
  const auto n_lumi_PMTs_per_FEB = 22;
  const auto n_channels_per_FEB = 32;
  const auto shift_all_lumi_PMTs = 44;

  for (auto lhcb_digit : lhcbDigits) {
    const auto ch_type = lhcb_digit->channelID().channelType();

    if (ch_type == LHCb::Detector::Plume::ChannelID::ChannelType::LUMI) {
      if (m_map_reversed.find(lhcb_digit->channelID().channelID()) == m_map_reversed.end()) {
        error() << "LHCb digit " << lhcb_digit->channelID().channelID() << " not found." << endmsg;
        ++m_error;
      }

      int idx_int = m_map_reversed.at(lhcb_digit->channelID().channelID());
      const auto feb = idx_int < n_lumi_PMTs_per_FEB ? 0 : 1;
      const auto n_ovt = idx_int - (feb * n_channels_per_FEB);
      bool ovt = ((allenDigits.ovr_th[feb] & (1 << (n_ovt))) >> (n_ovt));

      if (feb == 1) idx_int -= n_channels_per_FEB - n_lumi_PMTs_per_FEB;
      auto allen_adc = static_cast<int>(std::round(allenDigits.ADC_counts[idx_int])) - m_pedestalOffset;

      if (lhcb_digit->adc() == allen_adc)
        ++m_matched;
      else {
        ++m_error;
        error() << "ADC " << idx_int << " different at: LHCb " << lhcb_digit->adc() << ", Allen " << allen_adc
                << endmsg;
      }

      if (lhcb_digit->overThreshold() == ovt)
        ++m_matched_ovr;
      else {
        ++m_error_ovr;
        error() << "OverThreshold bit " << idx_int << " different at: LHCb " << lhcb_digit->overThreshold()
                << ", Allen " << ovt << endmsg;
      }
    }
    else if (ch_type == LHCb::Detector::Plume::ChannelID::ChannelType::TIME) {
      const auto chID = lhcb_digit->channelID().channelID();
      const auto chsubID = lhcb_digit->channelID().channelSubID();
      auto shift_ch = (chID == 11 || chID == 35) ? n_channels_per_FEB : 0;
      auto query = std::make_pair(chID, chsubID);

      if (m_map_reversed_time.find(query) != m_map_reversed_time.end()) {
        auto idx_int = m_map_reversed_time.at(query) + shift_ch + shift_all_lumi_PMTs;
        auto allen_adc = static_cast<int>(std::round(allenDigits.ADC_counts[idx_int])) - m_pedestalOffset;
        if (lhcb_digit->adc() == allen_adc)
          ++m_matched_time;
        else {
          ++m_error_time;
          error() << "ADC " << idx_int << " different at: LHCb " << lhcb_digit->adc() << ", Allen " << allen_adc
                  << endmsg;
        }
      }
      else {
        error() << "LHCb digit " << chID << " " << chsubID << " not found." << endmsg;
        ++m_error_time;
      }
    }
  }
}
