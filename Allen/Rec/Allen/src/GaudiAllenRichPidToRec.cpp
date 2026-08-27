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
#include "Kernel/EventLocalAllocator.h"
#include "Kernel/RichParticleIDType.h"

// Allen
#include "AlgorithmConversionTools.h"
#include "RichParticleHypos.cuh"

// STL
#include <algorithm>
#include <memory>
#include <vector>

/**
 * Convert Allen RICH PID output into LHCb::RichPIDs
 * for comparison with the Rec PID checker (PIDQC).
 */

namespace GaudiAllen::Converters {

  namespace {
    using InTracks = LHCb::Track::Range;
    using AllenPIDs = Allen::parameter_vector<Allen::Rich::ParticleIDType>;
    using AllenDLLs = Allen::parameter_vector<Allen::Rich::HypoData<float>>;
    using AllenOffsets = Allen::parameter_vector<unsigned>;
    using AllenHypos = Allen::parameter_vector<Allen::Rich::ParticleHypos>;

    // This mapping is only required while the standalone Allen build forces Allen to maintain its own copy of the
    // RICH particle hypothesis enum. Remove it and use the upstream Rich::ParticleIDType directly once the standalone
    // build and the duplicated Allen types have been retired.
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
  } // namespace

  class GaudiAllenRichPidToRec final : public Gaudi::Functional::Transformer<LHCb::RichPIDs(
                                         const InTracks&,
                                         const AllenPIDs&,
                                         const AllenDLLs&,
                                         const AllenOffsets&,
                                         const AllenOffsets&,
                                         const AllenHypos&,
                                         const AllenHypos&)> {
  public:
    GaudiAllenRichPidToRec(const std::string& name, ISvcLocator* pSvcLocator) :
      Transformer(
        name,
        pSvcLocator,
        {KeyValue {"TracksLocation", LHCb::TrackLocation::Default},
         KeyValue {"AllenBestPIDsLocation", "Allen/Rich/BestPID"},
         KeyValue {"AllenDLLsLocation", "Allen/Rich/DLLs"},
         KeyValue {"AllenPhotonOffsetsR1Location", "Allen/Rich/PhotonOffsetsR1"},
         KeyValue {"AllenPhotonOffsetsR2Location", "Allen/Rich/PhotonOffsetsR2"},
         KeyValue {"AllenHyposR1Location", "Allen/Rich/HyposR1"},
         KeyValue {"AllenHyposR2Location", "Allen/Rich/HyposR2"}},
        {KeyValue {"RichPIDsLocation", "Rec/Rich/AllenPIDs"}})
    {}

    LHCb::RichPIDs operator()(
      const InTracks& tracks,
      const AllenPIDs& allenPIDs,
      const AllenDLLs& allenDLLs,
      const AllenOffsets& offsetsR1,
      const AllenOffsets& offsetsR2,
      const AllenHypos& hyposR1,
      const AllenHypos& hyposR2) const override
    {

      LHCb::RichPIDs rPIDs;
      rPIDs.reserve(tracks.size());

      // only required for one of the radiator approaches
      const bool haveR1 = (offsetsR1.size() > tracks.size());
      const bool haveR2 = (offsetsR2.size() > tracks.size());

      if (allenPIDs.size() != tracks.size() || allenDLLs.size() != tracks.size()) {
        error() << "GaudiAllenRichPidToRec: size mismatch:"
                << " tracks=" << tracks.size() << " allenPIDs=" << allenPIDs.size() << " allenDLLs=" << allenDLLs.size()
                << endmsg;
        return rPIDs;
      }

      for (std::size_t i = 0; i < tracks.size(); ++i) {
        // Allen pid info
        const auto* tk = tracks[i];
        const auto bestH = recParticleType(allenPIDs[i]);
        const auto& allenDLL = allenDLLs[i];

        // Rec style pid
        auto pid = std::make_unique<LHCb::RichPID>();
        pid->setTrack(tk);
        pid->setBestParticleID(bestH);

        // Radiator
        // A track used a radiator if it has at least one photon reconstructed there, determined from the per-track
        // photon offset arrays.
        const bool usedR1 = haveR1 && (offsetsR1[i + 1] > offsetsR1[i]);
        const bool usedR2 = haveR2 && (offsetsR2[i + 1] > offsetsR2[i]);
        pid->setUsedAerogel(false); // not used in Allen
        pid->setUsedRich1Gas(usedR1);
        pid->setUsedRich2Gas(usedR2);

        // Threshold
        Allen::Rich::HypoData<int> thresh {};
        for (const auto allenHypo : Allen::Rich::particles()) {
          const auto hypo = recParticleType(allenHypo);
          thresh[allenHypo] = hyposR1[i].yield[allenHypo] > 0.f || hyposR2[i].yield[allenHypo] > 0.f;
          pid->setAboveThreshold(hypo, hyposR1[i].yield[allenHypo] > 0.f || hyposR2[i].yield[allenHypo] > 0.f);
        }

        // DLLs
        // Allen normalises DLLs to pion: allenDLL[X] = logL(X) - logL(pi)
        auto& vDLLs = pid->particleLLValues();
        for (const auto allenHypo : Allen::Rich::particles()) {
          const auto hypo = recParticleType(allenHypo);
          vDLLs[hypo] = static_cast<LHCb::RichPID::DLL>(allenDLL[allenHypo]);
        }
        vDLLs[Rich::Pion] = 0.f; // probably redundant
        rPIDs.insert(std::move(pid), tk->key());
      }
      return rPIDs;
    }
  };
  DECLARE_COMPONENT(GaudiAllenRichPidToRec)
} // namespace GaudiAllen::Converters
