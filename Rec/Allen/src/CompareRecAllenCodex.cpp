/***************************************************************************** \
 * (c) Copyright 2000-2023 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <string>
#include <vector>
#include <ostream>
#include <map>
#include <array>
#include <algorithm>
#include <tuple>

// Gaudi
#include "GaudiAlg/Consumer.h"
#include "Gaudi/Accumulators.h"
#include <Kernel/EventLocalAllocator.h>

// Allen
#include "CodexModel.cuh"

// Rec
#include "CodexCluster.h"

class CompareRecAllenCodex final
  : public Gaudi::Functional::Consumer<
      void(std::vector<CodexHit, LHCb::Allocators::EventLocal<CodexHit>> const&, LHCb::Codex::StripPairs const&)> {

public:
  /// Standard constructor
  CompareRecAllenCodex(const std::string& name, ISvcLocator* pSvcLocator);

  /// Algorithm execution
  void operator()(std::vector<CodexHit, LHCb::Allocators::EventLocal<CodexHit>> const&, LHCb::Codex::StripPairs const&)
    const override;

private:
  void compare(
    std::vector<CodexHit, LHCb::Allocators::EventLocal<CodexHit>> const& allenDigits,
    LHCb::Codex::StripPairs const& lhcbDigits) const;

  mutable Gaudi::Accumulators::Counter<> m_matched {this, "MatchedHits"};
  mutable Gaudi::Accumulators::Counter<> m_error {this, "MismatchedHits"};
};

DECLARE_COMPONENT(CompareRecAllenCodex)

CompareRecAllenCodex::CompareRecAllenCodex(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"Codex_digits_Allen", ""}, KeyValue {"Codex_digits_Moore", ""}})
{}
void CompareRecAllenCodex::operator()(
  std::vector<CodexHit, LHCb::Allocators::EventLocal<CodexHit>> const& Codex_digits_Allen,
  LHCb::Codex::StripPairs const& Codex_digits_Moore) const
{
  for (auto const& [allenDigits, lhcbDigits] : {std::forward_as_tuple(Codex_digits_Allen, Codex_digits_Moore)}) {
    compare(allenDigits, lhcbDigits);
  }
}

void CompareRecAllenCodex::compare(
  std::vector<CodexHit, LHCb::Allocators::EventLocal<CodexHit>> const& allenDigits,
  LHCb::Codex::StripPairs const& lhcbDigits) const
{
  // Build a simple comparable representation for Allen hits: (singlet_id, strip_type, strip_id, time)
  std::vector<std::array<int, 4>> allen_list;
  allen_list.reserve(allenDigits.size());
  for (const auto& h : allenDigits) {
    allen_list.push_back({static_cast<int>(h.singlet_id),
                          static_cast<int>(h.strip_type),
                          static_cast<int>(h.strip_id),
                          static_cast<int>(h.time)});
  }

  std::vector<std::array<int, 4>> rec_list;
  rec_list.reserve(lhcbDigits.size() * 2);
  for (const auto& sp : lhcbDigits) {
    // time0
    if (sp.time0 > 0) {
      LHCb::Codex::Strip s {sp, LHCb::Codex::Strip::UseSlot1 {false}};
      int typ = (s.type == LHCb::Codex::StripType::ETA) ? 1 : 0;
      int local_singlet = s.singlet;
      int dct = s.dct;
      rec_list.push_back(
        {static_cast<int>(3 * dct + local_singlet), typ, static_cast<int>(s.strip), static_cast<int>(s.time)});
    }
    // time1
    if (sp.time1 > 0) {
      LHCb::Codex::Strip s {sp, LHCb::Codex::Strip::UseSlot1 {true}};
      int typ = (s.type == LHCb::Codex::StripType::ETA) ? 1 : 0;
      int local_singlet = s.singlet;
      int dct = s.dct;
      rec_list.push_back(
        {static_cast<int>(3 * dct + local_singlet), typ, static_cast<int>(s.strip), static_cast<int>(s.time)});
    }
  }

  // Sort both lists to allow order-insensitive comparison
  auto cmp = [](auto const& a, auto const& b) {
    for (int i = 0; i < 4; ++i)
      if (a[i] != b[i]) return a[i] < b[i];
    return false;
  };
  std::sort(allen_list.begin(), allen_list.end(), cmp);
  std::sort(rec_list.begin(), rec_list.end(), cmp);

  // Compare lists
  const std::size_t n_allen = allen_list.size();
  const std::size_t n_rec = rec_list.size();
  if (n_allen != n_rec) {
    error() << "Different number of decoded hits: Allen=" << n_allen << " Rec=" << n_rec << endmsg;
  }

  const std::size_t ncomp = std::min(n_allen, n_rec);
  for (std::size_t i = 0; i < ncomp; ++i) {
    if (allen_list[i] == rec_list[i]) {
      ++m_matched;
    }
    else {
      ++m_error;
      error() << "Decoded hit mismatch at index " << i << ": Allen {" << allen_list[i][0] << "," << allen_list[i][1]
              << "," << allen_list[i][2] << "," << allen_list[i][3] << "} vs Rec {" << rec_list[i][0] << ","
              << rec_list[i][1] << "," << rec_list[i][2] << "," << rec_list[i][3] << "}" << endmsg;
    }
  }

  // Any remaining entries count as mismatches
  if (n_allen > ncomp) {
    for (std::size_t i = ncomp; i < n_allen; ++i) {
      ++m_error;
    }
  }
  if (n_rec > ncomp) {
    for (std::size_t i = ncomp; i < n_rec; ++i) {
      ++m_error;
    }
  }
}
