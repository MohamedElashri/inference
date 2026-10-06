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

// Gaudi
#include "GaudiAlg/Transformer.h"

// LHCb
#include "Event/RichPID.h"
#include "Event/Track.h"
#include "Kernel/RichParticleIDType.h"

// Allen
#include "AlgorithmConversionTools.h"
#include "RichParticleHypos.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

#include <memory>
#include <vector>

namespace GaudiAllen::Converters {

  inline constexpr Allen::Rich::ParticleArray<Rich::ParticleIDType> RecParticleTypes {
    Rich::Electron,
    Rich::Muon,
    Rich::Pion,
    Rich::Kaon,
    Rich::Proton,
    Rich::Deuteron,
    Rich::BelowThreshold};

  constexpr auto recParticleType(const Allen::Rich::ParticleIDType particle) noexcept
  {
    return particle == Allen::Rich::Unknown ? Rich::Unknown : RecParticleTypes[particle];
  }

  // ================================================================
  //  Raw per-track PID data (no track association)
  // ================================================================

  struct AllenRichPIDEntry {
    Allen::Rich::ParticleIDType bestPID = Allen::Rich::Unknown;
    Allen::Rich::HypoData<float> dlls {};
    bool usedR1 = false;
    bool usedR2 = false;
    Allen::Rich::ParticleHypos hyposR1 {};
    Allen::Rich::ParticleHypos hyposR2 {};
  };

  using AllenRichPIDEntries = std::vector<AllenRichPIDEntry>;

  // ================================================================
  //  Step 1: Multi-event converter → per-event raw PID entries
  // ================================================================

  class ConvertAllenRichPidToRec final
    : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenRichPIDEntries>(
        const Allen::device_buffer<unsigned>&,                      // track offsets (N+1)
        const Allen::device_buffer<Allen::Rich::ParticleIDType>&,   // best PID per track
        const Allen::device_buffer<Allen::Rich::HypoData<float>>&,  // DLLs per track
        const Allen::device_buffer<unsigned>&,                      // Rich1 photon offsets
        const Allen::device_buffer<unsigned>&,                      // Rich2 photon offsets
        const Allen::device_buffer<Allen::Rich::ParticleHypos>&,    // Rich1 hypos
        const Allen::device_buffer<Allen::Rich::ParticleHypos>&)> { // Rich2 hypos

  public:
    ConvertAllenRichPidToRec(const std::string& name, ISvcLocator* pSvcLocator) :
      MultiTransformer(
        name,
        pSvcLocator,
        {KeyValue {"rich_track_offsets", ""},
         KeyValue {"AllenBestPIDsLocation", "Allen/Rich/BestPID"},
         KeyValue {"AllenDLLsLocation", "Allen/Rich/DLLs"},
         KeyValue {"AllenPhotonOffsetsR1Location", "Allen/Rich/PhotonOffsetsR1"},
         KeyValue {"AllenPhotonOffsetsR2Location", "Allen/Rich/PhotonOffsetsR2"},
         KeyValue {"AllenHyposR1Location", "Allen/Rich/HyposR1"},
         KeyValue {"AllenHyposR2Location", "Allen/Rich/HyposR2"}},
        {KeyValue {"AllenRichPIDEntries", ""}})
    {}

    std::tuple<std::vector<AllenRichPIDEntries>> operator()(
      const EventContext& /*ctx*/,
      const Allen::device_buffer<unsigned>& dev_track_off,
      const Allen::device_buffer<Allen::Rich::ParticleIDType>& dev_pids,
      const Allen::device_buffer<Allen::Rich::HypoData<float>>& dev_dlls,
      const Allen::device_buffer<unsigned>& dev_off_r1,
      const Allen::device_buffer<unsigned>& dev_off_r2,
      const Allen::device_buffer<Allen::Rich::ParticleHypos>& dev_hyp_r1,
      const Allen::device_buffer<Allen::Rich::ParticleHypos>& dev_hyp_r2) const override
    {
      auto h_track_off = dev_track_off.to_host();
      auto h_pids = dev_pids.to_host();
      auto h_dlls = dev_dlls.to_host();
      auto h_off_r1 = dev_off_r1.to_host();
      auto h_off_r2 = dev_off_r2.to_host();
      auto h_hyp_r1 = dev_hyp_r1.to_host();
      auto h_hyp_r2 = dev_hyp_r2.to_host();

      const unsigned n_events = h_track_off.size() - 1;

      std::vector<AllenRichPIDEntries> all_entries;
      all_entries.reserve(n_events);

      for (unsigned evt = 0; evt < n_events; ++evt) {
        const unsigned t_begin = h_track_off[evt];
        const unsigned t_end = h_track_off[evt + 1];

        AllenRichPIDEntries entries;
        entries.reserve(t_end - t_begin);

        for (unsigned t = t_begin; t < t_end; ++t) {
          entries.push_back(
            {h_pids[t],
             h_dlls[t],
             h_off_r1[t + 1] > h_off_r1[t],
             h_off_r2[t + 1] > h_off_r2[t],
             h_hyp_r1[t],
             h_hyp_r2[t]});
        }

        all_entries.emplace_back(std::move(entries));
      }

      return std::make_tuple(std::move(all_entries));
    }
  };

  DECLARE_COMPONENT(ConvertAllenRichPidToRec)

  // ================================================================
  //  Step 2: Per-event transformer → RichPIDs with track association
  // ================================================================

  class AssociateAllenRichPIDs final
    : public Gaudi::Functional::Transformer<LHCb::RichPIDs(const AllenRichPIDEntries&, const LHCb::Track::Range&)> {

  public:
    AssociateAllenRichPIDs(const std::string& name, ISvcLocator* pSvcLocator) :
      Transformer(
        name,
        pSvcLocator,
        {KeyValue {"AllenRichPIDEntries", ""}, KeyValue {"TracksLocation", LHCb::TrackLocation::Default}},
        {KeyValue {"RichPIDsLocation", "Rec/Rich/AllenPIDs"}})
    {}

    LHCb::RichPIDs operator()(const AllenRichPIDEntries& entries, const LHCb::Track::Range& tracks) const override
    {
      LHCb::RichPIDs rPIDs;

      if (entries.size() != tracks.size()) {
        error() << "AssociateAllenRichPIDs: size mismatch: entries=" << entries.size() << " tracks=" << tracks.size()
                << endmsg;
        return rPIDs;
      }

      rPIDs.reserve(tracks.size());

      for (std::size_t i = 0; i < tracks.size(); ++i) {
        const auto* tk = tracks[i];
        const auto& e = entries[i];

        const auto bestH = recParticleType(e.bestPID);

        auto pid = std::make_unique<LHCb::RichPID>();
        pid->setTrack(tk);
        pid->setBestParticleID(bestH);

        pid->setUsedAerogel(false);
        pid->setUsedRich1Gas(e.usedR1);
        pid->setUsedRich2Gas(e.usedR2);

        for (const auto allenHypo : Allen::Rich::particles()) {
          const auto hypo = recParticleType(allenHypo);
          pid->setAboveThreshold(hypo, e.hyposR1.yield[allenHypo] > 0.f || e.hyposR2.yield[allenHypo] > 0.f);
        }

        auto& vDLLs = pid->particleLLValues();
        for (const auto allenHypo : Allen::Rich::particles()) {
          const auto hypo = recParticleType(allenHypo);
          vDLLs[hypo] = static_cast<LHCb::RichPID::DLL>(e.dlls[allenHypo]);
        }

        rPIDs.insert(std::move(pid), tk->key());
      }

      return rPIDs;
    }
  };

  DECLARE_COMPONENT(AssociateAllenRichPIDs)
} // namespace GaudiAllen::Converters
