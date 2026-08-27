/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/MCParticle.h"
#include "Event/MCTrackInfo.h"
#include "Event/MCVertex.h"
#include "Event/ODIN.h"
#include "Event/RawEvent.h"
#include "Event/RawBank.h"
#include "LHCbAlgs/Transformer.h"
#include "Associators/Associators.h"
#include "Associators/Associators.h"
#include <Dumpers/Utils.h>
#include <string>

namespace {

  void collectProductss(
    const LHCb::MCVertex& mcpv,
    const LHCb::MCVertex& mcvtx,
    std::vector<const LHCb::MCParticle*>& allprods)
  {
    for (const auto& idau : mcvtx.products()) {
      double dv2 = (mcpv.position() - idau->originVertex()->position()).Mag2();
      if (dv2 > static_cast<double>((100.f * Allen::Units::mm) * (100.f * Allen::Units::mm))) continue;
      allprods.emplace_back(idau);
      for (const auto& ivtx : idau->endVertices()) {
        collectProductss(mcpv, *ivtx, allprods);
      }
    }
  }

  // count number reconstructible tracks in the same way as PrimaryVertexChecker
  int count_reconstructible_mc_particles(const LHCb::MCVertex& avtx, const MCTrackInfo& trInfo)
  {
    std::vector<const LHCb::MCParticle*> allproducts;
    collectProductss(avtx, avtx, allproducts);

    return std::ranges::count_if(allproducts, [&](const auto* pmcp) {
      if (pmcp->particleID().threeCharge() == 0 || !trInfo.hasVelo(pmcp)) return false;
      double dv2 = (avtx.position() - pmcp->originVertex()->position()).Mag2();
      return dv2 < 0.0000001 && pmcp->p() > 100. * Gaudi::Units::MeV;
    });
  }
} // namespace

/** @class PVDumper PVDumper.h
 *  tool to dump the MC truth informaiton for PVs
 *  based on the PrTrackerDumper code
 *
 *  @author Florian Reiss
 *  @date   2018-12-17
 */
class PVDumper
  : public LHCb::Algorithm::MultiTransformer<
      std::tuple<LHCb::RawEvent, LHCb::RawBank::View>(const LHCb::MCVertices& MCVertices, const LHCb::MCProperty&)> {
public:
  /// Standard constructor
  PVDumper(const std::string& name, ISvcLocator* pSvcLocator);

  std::tuple<LHCb::RawEvent, LHCb::RawBank::View> operator()(
    const LHCb::MCVertices& MCVertices,
    const LHCb::MCProperty&) const override;

private:
  static const LHCb::RawBank::BankType m_bankType = LHCb::RawBank::BankType::OTError;
};

// Declaration of the Algorithm Factory
DECLARE_COMPONENT(PVDumper)

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

PVDumper::PVDumper(const std::string& name, ISvcLocator* pSvcLocator) :
  MultiTransformer {
    name,
    pSvcLocator,
    {KeyValue {"MCVerticesLocation", LHCb::MCVertexLocation::Default},
     KeyValue {"MCPropertyLocation", LHCb::MCPropertyLocation::TrackInfo}},
    {KeyValue {"OutputRawEventLocation", "Allen/MCPVRawEvent"},
     KeyValue {"OutputRawBankLocation", "Allen/MCPVRawBank"}}}
{}

std::tuple<LHCb::RawEvent, LHCb::RawBank::View> PVDumper::operator()(
  const LHCb::MCVertices& MCVertices,
  const LHCb::MCProperty& MCProp) const
{
  DumpUtils::Writer writer;

  auto goodVertex = [](const auto* v) {
    return !v->mother() && v->type() == LHCb::MCVertex::MCVertexType::ppCollision;
  };

  int n_PVs = std::ranges::count_if(MCVertices, goodVertex);
  writer.write(n_PVs);

  MCTrackInfo trInfo {MCProp};
  for (const auto& mcv : MCVertices) {
    if (!goodVertex(mcv)) continue;
    const auto& pv = mcv->position();
    writer.write(count_reconstructible_mc_particles(*mcv, trInfo), pv.X(), pv.Y(), pv.Z());
  }

  // Write PV MC information to raw event
  LHCb::RawEvent rawEvent;
  constexpr int bankSize = 64512;
  for (const auto [sourceID, data] : LHCb::range::enumerate(LHCb::range::chunk(writer.buffer(), bankSize))) {
    rawEvent.addBank(sourceID, m_bankType, 1, data);
  }

  return viewFromRawEvent(std::move(rawEvent), m_bankType);
}
