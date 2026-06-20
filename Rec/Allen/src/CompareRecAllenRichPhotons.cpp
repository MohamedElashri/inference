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

// Gaudi
#include "LHCbAlgs/Consumer.h"
#include "Gaudi/Accumulators.h"
#include <Gaudi/Accumulators/Histogram.h>
#include <Kernel/EventLocalAllocator.h>

// Rec
#include "RichFutureRecEvent/RichRecCherenkovPhotons.h"
#include "RichFutureRecEvent/RichRecRelations.h"
#include "RichFutureRecEvent/RichRecPhotonPredictedPixelSignals.h"
#include "RichUtils/FastMaths.h"
#include "RichUtils/ZipRange.h"

// Allen
#include "AlgorithmConversionTools.h"
#include "RichPhoton.cuh"
#include "RichParticleHypos.cuh"

// std
#include <iomanip>
#include <limits>
#include <map>
#include <vector>

using AllenRichPhoton = Allen::Rich::PhotonReco::Photon;

// Helper struct to unpack SIMD Rec photons
struct RecPhotonIndividual {
  unsigned trackID {};
  uint64_t smartID {};
  float ckTheta {};
  float ckPhi {};
  Rich::DetectorType rich {Rich::InvalidDetector};
  Rich::Future::HypoData<float> signals {};

  RecPhotonIndividual(unsigned tid, uint64_t sid, float theta, float phi, const Rich::DetectorType rich) :
    trackID(tid), smartID(sid), ckTheta(theta), ckPhi(phi), rich(rich)
  {}
};

void printPhotonAttributes(const std::string& label, float ckTheta, float ckPhi, uint64_t smartIDKey)
{
  std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10);
  std::cout << label << ": ";
  std::cout << "ckTheta=" << ckTheta << ", ";
  std::cout << "ckPhi=" << ckPhi << ", ";
  std::cout << "SID=" << smartIDKey << "\n";
}

/**
 * The idea of this test is to match rec and allen photons by their respective
 * pixel SmartIDs, as this will give us which photons exist both in rec and Allen
 * for a given CKAngles threshold
 * Keep in mind that a single pixel may have multiple photons associated both
 * in Allen and Rec
 **/
class CompareRecAllenRichPhotons final : public LHCb::Algorithm::Consumer<void(
                                           const Allen::parameter_vector<AllenRichPhoton>&,
                                           const Allen::parameter_vector<unsigned>&,
                                           const Allen::parameter_vector<Allen::Rich::HypoData<float>>&,
                                           const Allen::parameter_vector<AllenRichPhoton>&,
                                           const Allen::parameter_vector<unsigned>&,
                                           const Allen::parameter_vector<Allen::Rich::HypoData<float>>&,
                                           const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>&,
                                           const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>&,
                                           const Rich::Future::Rec::SIMDCherenkovPhoton::Vector&,
                                           const Rich::Future::Rec::Relations::PhotonToParents::Vector&,
                                           const Rich::Future::Rec::SIMDPhotonSignals::Vector&)> {

public:
  /// Standard constructor
  CompareRecAllenRichPhotons(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override;

  /// Algorithm execution
  void operator()(
    const Allen::parameter_vector<AllenRichPhoton>&,
    const Allen::parameter_vector<unsigned>&,
    const Allen::parameter_vector<Allen::Rich::HypoData<float>>&,
    const Allen::parameter_vector<AllenRichPhoton>&,
    const Allen::parameter_vector<unsigned>&,
    const Allen::parameter_vector<Allen::Rich::HypoData<float>>&,
    const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>&,
    const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>&,
    const Rich::Future::Rec::SIMDCherenkovPhoton::Vector&,
    const Rich::Future::Rec::Relations::PhotonToParents::Vector&,
    const Rich::Future::Rec::SIMDPhotonSignals::Vector&) const override;

private:
  // Expect containers from the same track/rich
  // match the photons based on pixel id
  template<typename MatchCount, typename AllenNotInRec, typename RecNotInAllen>
  void matchPhotonsForTrack(
    [[maybe_unused]] const unsigned trackID,
    const std::vector<AllenRichPhoton>& allenPhotons,
    const std::vector<RecPhotonIndividual>& recPhotons,
    const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>& allenPixelsSmartID,
    [[maybe_unused]] const std::vector<Allen::Rich::HypoData<float>>& allenPhotonPixelSignals,
    MatchCount& match_count,
    AllenNotInRec& allen_not_in_rec,
    RecNotInAllen& rec_not_in_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_allen_all,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec_all) const
  {
    // Track which photons have been matched
    std::vector<bool> allen_matched(allenPhotons.size(), false);
    std::vector<bool> rec_matched(recPhotons.size(), false);

    // match Allen and Rec photons
    for (size_t i = 0; i < allenPhotons.size(); ++i) {
      uint64_t smartID = allenPixelsSmartID[allenPhotons[i].pixelIdx].key();
      for (size_t j = 0; j < recPhotons.size(); ++j) {
        // Skip already matched Rec photons
        if (rec_matched[j]) continue;

        // Check match
        if (smartID == recPhotons[j].smartID) {
          allen_matched[i] = true;
          rec_matched[j] = true;
          ++match_count;

          ++ckThetaRec_allen[allenPhotons[i].ckTheta];
          ++ckThetaRec_rec[recPhotons[j].ckTheta];
          ++ckThetaRec_rec_allen[recPhotons[j].ckTheta - allenPhotons[i].ckTheta];

          break; // found Allen photon in rec
        }
      }
    }

    // Report unmatched photons
    for (size_t i = 0; i < allenPhotons.size(); ++i) {
      ++ckThetaRec_allen_all[allenPhotons[i].ckTheta];
      if (!allen_matched[i]) {
        ++allen_not_in_rec;
        // error() << "Allen photon not found in Rec (SmartID=" << smartID << ")" << endmsg;
        // printPhotonAttributes("Allen", allenPhotons[i].ckTheta, allenPhotons[i].ckPhi, smartID);
      }
    }

    for (size_t j = 0; j < recPhotons.size(); ++j) {
      ++ckThetaRec_rec_all[recPhotons[j].ckTheta];
      if (!rec_matched[j]) {
        ++rec_not_in_allen;
        // error() << "Rec photon not found in Allen (SmartID=" << smartID << ")" << endmsg;
        // printPhotonAttributes("Rec", recPhotons[j].ckTheta, recPhotons[j].ckPhi, smartID);
      }
    }
  }

private:
  mutable Gaudi::Accumulators::Counter<> m_allen_not_in_rec_r1 {this, "R1 Photons Allen not found in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_rec_not_in_allen_r1 {this, "R1 Photons Rec not found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_allen_reviewed_r1 {this, "R1 Photons Allen reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_rec_reviewed_r1 {this, "R1 Photons Rec reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_matched_photons_r1 {this, "R1 Photons Matched"};

  mutable Gaudi::Accumulators::Counter<> m_allen_not_in_rec_r2 {this, "R2 Photons Allen not found in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_rec_not_in_allen_r2 {this, "R2 Photons Rec not found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_allen_reviewed_r2 {this, "R2 Photons Allen reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_rec_reviewed_r2 {this, "R2 Photons Rec reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_matched_photons_r2 {this, "R2 Photons Matched"};

  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_rec_r1 {this, "R1 Tracks not used in Rec but in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_allen_r1 {this, "R1 Tracks not used in Allen but in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_used_both_r1 {this, "R1 Tracks used in both"};

  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_rec_r2 {this, "R2 Tracks not used in Rec but in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_allen_r2 {this, "R2 Tracks not used in Allen but in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_used_both_r2 {this, "R2 Tracks used in both"};

  mutable Gaudi::Accumulators::Counter<> m_n_tracks {this, "Tracks"};

  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_all_r1 {this, "R1 ckTheta Allen (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_all_r1 {this, "R1 ckTheta Rec (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_all_r2 {this, "R2 ckTheta Allen (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_all_r2 {this, "R2 ckTheta Rec (all)"};

  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_r1 {this, "R1 ckTheta Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_r1 {this, "R1 ckTheta Rec (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_allen_r1 {this, "R1 ckTheta Rec-Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_r2 {this, "R2 ckTheta Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_r2 {this, "R2 ckTheta Rec (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_allen_r2 {this, "R2 ckTheta Rec-Allen (matched)"};
};

DECLARE_COMPONENT(CompareRecAllenRichPhotons)

CompareRecAllenRichPhotons::CompareRecAllenRichPhotons(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"rich1_photons", ""},
     KeyValue {"rich1_photons_offsets", ""},
     KeyValue {"rich1_photons_pixel_signals", ""},
     KeyValue {"rich2_photons", ""},
     KeyValue {"rich2_photons_offsets", ""},
     KeyValue {"rich2_photons_pixel_signals", ""},
     KeyValue {"rich1_pixels_smartid", ""},
     KeyValue {"rich2_pixels_smartid", ""},
     KeyValue {"CherenkovPhotons", ""},
     KeyValue {"PhotonToParents", ""},
     KeyValue {"RecPhotonSignals", ""}})
{}

StatusCode CompareRecAllenRichPhotons::initialize()
{
  return Consumer::initialize().andThen([&] {
    m_ckThetaRec_allen_all_r1.setTitle("R1 ckTheta Allen (all)");
    m_ckThetaRec_rec_all_r1.setTitle("R1 ckTheta Rec (all)");
    m_ckThetaRec_allen_all_r2.setTitle("R2 ckTheta Allen (all)");
    m_ckThetaRec_rec_all_r2.setTitle("R2 ckTheta Rec (all)");

    m_ckThetaRec_allen_r1.setTitle("R1 ckTheta Allen (matched)");
    m_ckThetaRec_rec_r1.setTitle("R1 ckTheta Rec (matched)");
    m_ckThetaRec_rec_allen_r1.setTitle("R1 ckTheta Rec-Allen (matched)");
    m_ckThetaRec_allen_r2.setTitle("R2 ckTheta Allen (matched)");
    m_ckThetaRec_rec_r2.setTitle("R2 ckTheta Rec (matched)");
    m_ckThetaRec_rec_allen_r2.setTitle("R2 ckTheta Rec-Allen (matched)");

    using Axis1D = Gaudi::Accumulators::Axis<double>;
    m_ckThetaRec_allen_all_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_all_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_allen_all_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_all_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});

    m_ckThetaRec_allen_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_allen_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(-0.0026, 0.0026, 100)});
    m_ckThetaRec_allen_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_allen_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(-0.002, 0.002, 100)});
  });
}

void CompareRecAllenRichPhotons::operator()(
  const Allen::parameter_vector<AllenRichPhoton>& allenRich1Photons,
  const Allen::parameter_vector<unsigned>& allenRich1PhotonsOffsets,
  const Allen::parameter_vector<Allen::Rich::HypoData<float>>& allenRich1PhotonsPixelSignals,
  const Allen::parameter_vector<AllenRichPhoton>& allenRich2Photons,
  const Allen::parameter_vector<unsigned>& allenRich2PhotonsOffsets,
  const Allen::parameter_vector<Allen::Rich::HypoData<float>>& allenRich2PhotonsPixelSignals,
  const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>& allenRich1PixelsSmartID,
  const Allen::parameter_vector<Allen::Rich::Decoding::SmartID>& allenRich2PixelsSmartID,
  const Rich::Future::Rec::SIMDCherenkovPhoton::Vector& recPhotons,
  const Rich::Future::Rec::Relations::PhotonToParents::Vector& photRels,
  const Rich::Future::Rec::SIMDPhotonSignals::Vector& recPhotonSignals) const
{
  auto allen_not_in_rec_r1 = m_allen_not_in_rec_r1.buffer();
  auto rec_not_in_allen_r1 = m_rec_not_in_allen_r1.buffer();
  auto allen_reviewed_r1 = m_allen_reviewed_r1.buffer();
  auto rec_reviewed_r1 = m_rec_reviewed_r1.buffer();
  auto matched_photons_r1 = m_matched_photons_r1.buffer();
  auto allen_not_in_rec_r2 = m_allen_not_in_rec_r2.buffer();
  auto rec_not_in_allen_r2 = m_rec_not_in_allen_r2.buffer();
  auto allen_reviewed_r2 = m_allen_reviewed_r2.buffer();
  auto rec_reviewed_r2 = m_rec_reviewed_r2.buffer();
  auto matched_photons_r2 = m_matched_photons_r2.buffer();
  auto tracks_not_in_rec_r1 = m_tracks_not_in_rec_r1.buffer();
  auto tracks_not_in_allen_r1 = m_tracks_not_in_allen_r1.buffer();
  auto tracks_used_both_r1 = m_tracks_used_both_r1.buffer();
  auto tracks_not_in_rec_r2 = m_tracks_not_in_rec_r2.buffer();
  auto tracks_not_in_allen_r2 = m_tracks_not_in_allen_r2.buffer();
  auto tracks_used_both_r2 = m_tracks_used_both_r2.buffer();
  auto n_tracks_counter = m_n_tracks.buffer();

  // Unpack Rec SIMD photons
  std::vector<RecPhotonIndividual> recPhotonsFlat;

  for (const auto&& [recPhoton, rels, sigs] : Rich::Ranges::ConstZip(recPhotons, photRels, recPhotonSignals)) {
    for (size_t i = 0; i < recPhoton.CherenkovTheta().size(); ++i) {
      // Only include valid photons
      if (recPhoton.validityMask()[i]) {
        auto& ph = recPhotonsFlat.emplace_back(
          rels.trackIndex(),
          recPhoton.smartID()[i].key(),
          recPhoton.CherenkovTheta()[i],
          recPhoton.CherenkovPhi()[i],
          recPhoton.smartID()[i].rich());
        for (const auto id : Rich::particles()) {
          ph.signals[id] = sigs[id][i];
        }
      }
    }
  }

  // Group by trackID and rich
  std::map<unsigned, std::vector<AllenRichPhoton>> allen_by_trackid_r1;
  std::map<unsigned, std::vector<Allen::Rich::HypoData<float>>> allen_signals_by_trackid_r1;
  std::map<unsigned, std::vector<AllenRichPhoton>> allen_by_trackid_r2;
  std::map<unsigned, std::vector<Allen::Rich::HypoData<float>>> allen_signals_by_trackid_r2;
  std::map<unsigned, std::vector<RecPhotonIndividual>> rec_by_trackid_r1;
  std::map<unsigned, std::vector<RecPhotonIndividual>> rec_by_trackid_r2;
  const unsigned n_tracks = allenRich1PhotonsOffsets.size() - 1;
  {
    // Allen
    for (unsigned trackID = 0; trackID < n_tracks; trackID++) {
      unsigned start = allenRich1PhotonsOffsets[trackID];
      unsigned size = allenRich1PhotonsOffsets[trackID + 1] - start;
      for (unsigned photon = 0; photon < size; photon++) {
        ++allen_reviewed_r1;
        allen_by_trackid_r1[trackID].push_back(allenRich1Photons[start + photon]);
        allen_signals_by_trackid_r1[trackID].push_back(allenRich1PhotonsPixelSignals[start + photon]);
      }
    }
    for (unsigned trackID = 0; trackID < n_tracks; trackID++) {
      unsigned start = allenRich2PhotonsOffsets[trackID];
      unsigned size = allenRich2PhotonsOffsets[trackID + 1] - start;
      for (unsigned photon = 0; photon < size; photon++) {
        ++allen_reviewed_r2;
        allen_by_trackid_r2[trackID].push_back(allenRich2Photons[start + photon]);
        allen_signals_by_trackid_r2[trackID].push_back(allenRich2PhotonsPixelSignals[start + photon]);
      }
    }

    // Rec
    for (const auto& recPhoton : recPhotonsFlat) {
      if (recPhoton.rich == Rich::Rich1) {
        ++rec_reviewed_r1;
        rec_by_trackid_r1[recPhoton.trackID].push_back(recPhoton);
      }
      else if (recPhoton.rich == Rich::Rich2) {
        ++rec_reviewed_r2;
        rec_by_trackid_r2[recPhoton.trackID].push_back(recPhoton);
      }
    }
  }

  // Stats on tracks:
  for (unsigned trackID = 0; trackID < n_tracks; trackID++) {
    if (rec_by_trackid_r1[trackID].size() == 0 && allen_by_trackid_r1[trackID].size() != 0) ++tracks_not_in_rec_r1;
    if (allen_by_trackid_r1[trackID].size() == 0 && rec_by_trackid_r1[trackID].size() != 0) ++tracks_not_in_allen_r1;
    if (rec_by_trackid_r1[trackID].size() != 0 && allen_by_trackid_r1[trackID].size() != 0) ++tracks_used_both_r1;
    ++n_tracks_counter;
  }
  for (unsigned trackID = 0; trackID < n_tracks; trackID++) {
    if (rec_by_trackid_r2[trackID].size() == 0 && allen_by_trackid_r2[trackID].size() != 0) ++tracks_not_in_rec_r2;
    if (allen_by_trackid_r2[trackID].size() == 0 && rec_by_trackid_r2[trackID].size() != 0) ++tracks_not_in_allen_r2;
    if (rec_by_trackid_r2[trackID].size() != 0 && allen_by_trackid_r2[trackID].size() != 0) ++tracks_used_both_r2;
  }

  // Match individual photons
  for (unsigned trackID = 0; trackID < n_tracks; trackID++) {
    matchPhotonsForTrack(
      trackID,
      allen_by_trackid_r1[trackID],
      rec_by_trackid_r1[trackID],
      allenRich1PixelsSmartID,
      allen_signals_by_trackid_r1[trackID],
      matched_photons_r1,
      allen_not_in_rec_r1,
      rec_not_in_allen_r1,
      m_ckThetaRec_allen_r1,
      m_ckThetaRec_rec_r1,
      m_ckThetaRec_rec_allen_r1,
      m_ckThetaRec_allen_all_r1,
      m_ckThetaRec_rec_all_r1);
    matchPhotonsForTrack(
      trackID,
      allen_by_trackid_r2[trackID],
      rec_by_trackid_r2[trackID],
      allenRich2PixelsSmartID,
      allen_signals_by_trackid_r2[trackID],
      matched_photons_r2,
      allen_not_in_rec_r2,
      rec_not_in_allen_r2,
      m_ckThetaRec_allen_r2,
      m_ckThetaRec_rec_r2,
      m_ckThetaRec_rec_allen_r2,
      m_ckThetaRec_allen_all_r2,
      m_ckThetaRec_rec_all_r2);
  }
}
