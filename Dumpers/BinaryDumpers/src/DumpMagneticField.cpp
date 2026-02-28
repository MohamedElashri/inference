/*****************************************************************************\
* (c) Copyright 2000-2025 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include <tuple>
#include <vector>

// Gaudi
#include <GaudiAlg/Transformer.h>
#include <GaudiAlg/FunctionalUtilities.h>

// LHCb
#include <Magnet/DeMagnet.h>
#include <DetDesc/GenericConditionAccessorHolder.h>

// Include files
#include <Dumpers/Utils.h>
#include <Dumpers/Identifiers.h>

#include "Dumper.h"

namespace Dumpers {
  struct MagneticField {
    MagneticField(std::vector<char>& data, const DeMagnet& deMagnet)
    {
      const auto* grid = deMagnet.fieldGrid();
      DumpUtils::Writer output {};
      output.write(grid->invDXYZ());
      output.write(grid->sizeXYZ());
      output.write(grid->minXYZ());
      output.write(grid->field());
      data = output.buffer();
    }
  };
} // namespace Dumpers

class DumpMagneticField final
  : public Allen::Dumpers::
      Dumper<void(Dumpers::MagneticField const&), LHCb::Algorithm::Traits::usesConditions<Dumpers::MagneticField>> {
public:
  DumpMagneticField(const std::string& name, ISvcLocator* svcLoc);

  void operator()(const Dumpers::MagneticField&) const override;

  StatusCode initialize() override;

private:
  std::vector<char> m_data;
};

DECLARE_COMPONENT(DumpMagneticField)

// Add the multitransformer call , which keyvalues for Magnetic Field ?

DumpMagneticField::DumpMagneticField(const std::string& name, ISvcLocator* svcLoc) :
  Dumper(name, svcLoc, {KeyValue {"MagneticFieldLocation", location(name, "Magfield")}})
{}

StatusCode DumpMagneticField::initialize()
{
  return Dumper::initialize().andThen([&] {
    register_producer(Allen::NonEventData::MagneticField::id, "magfield", m_data);
    addConditionDerivation(
      {LHCb::Det::Magnet::det_path}, inputLocation<Dumpers::MagneticField>(), [&](DeMagnet const& deMagnet) {
        auto magfield = Dumpers::MagneticField {m_data, deMagnet};
        dump();
        return magfield;
      });
  });
}

void DumpMagneticField::operator()(const Dumpers::MagneticField&) const {}
