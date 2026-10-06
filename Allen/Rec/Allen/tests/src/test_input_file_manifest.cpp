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

#define BOOST_TEST_MODULE input_file_manifest
#include <boost/test/unit_test.hpp>
#include <GaudiKernel/Bootstrap.h>
#include <GaudiKernel/IAppMgrUI.h>
#include <stdexcept>
#include "MDFProvider.h"

namespace {
  struct FileMetadata {
    std::string name;
    std::array<std::string_view, 2> locations;
    LHCb::IO::InputFileManifest manifest() const { return {name, locations}; }
  };
} // namespace

// A prefetched slice can contain events from different files; moving the batch
// into the worker must preserve both the views and their buffer-owned storage.
BOOST_AUTO_TEST_CASE(manifests_follow_events_and_buffer_lifetime)
{
  Allen::TransposeWorkers::PrefetchedEvents batch;
  auto first =
    std::make_shared<FileMetadata>(FileMetadata {"first.dst", {"/Event/pSim/MCParticles", "/Event/pSim/MCVertices"}});
  auto second = std::make_shared<FileMetadata>(
    FileMetadata {"second.dst", {"/Event/HLT2/pSim/MCParticles", "/Event/HLT2/pSim/MCVertices"}});
  std::weak_ptr<FileMetadata> first_owner = first, second_owner = second;

  for (auto const& buffer : {first, first, second}) {
    batch.events.emplace_back();
    batch.buffers.push_back(buffer);
    batch.input_file_manifests.push_back(buffer->manifest());
  }
  first.reset();
  second.reset();
  auto slice_batch = std::move(batch);
  BOOST_REQUIRE_EQUAL(slice_batch.input_file_manifests.size(), slice_batch.size());
  BOOST_CHECK_EQUAL(slice_batch.input_file_manifests[0].fileName, "first.dst");
  BOOST_CHECK_EQUAL(slice_batch.input_file_manifests[1].fileName, "first.dst");
  BOOST_CHECK_EQUAL(slice_batch.input_file_manifests[2].fileName, "second.dst");
  BOOST_CHECK_EQUAL(slice_batch.input_file_manifests[0].locations[0], "/Event/pSim/MCParticles");
  BOOST_CHECK_EQUAL(slice_batch.input_file_manifests[2].locations[0], "/Event/HLT2/pSim/MCParticles");
  BOOST_CHECK(!first_owner.expired());
  BOOST_CHECK(!second_owner.expired());

  slice_batch.reset();
  BOOST_CHECK(slice_batch.input_file_manifests.empty());
  BOOST_CHECK(slice_batch.buffers.empty());
  BOOST_CHECK(first_owner.expired());
  BOOST_CHECK(second_owner.expired());
}

BOOST_AUTO_TEST_CASE(non_root_input_rejects_manifest_requests)
{
  SmartIF<ISvcLocator> locator {Gaudi::createApplicationMgr()};
  MDFProvider provider {"ManifestTest", locator.get()};
  // No initialization or slice is needed: reject the unsupported request before
  // accessing any ROOT state, rather than returning an empty manifest.
  BOOST_CHECK_EXCEPTION(provider.getInputFileManifest(0, 0), std::logic_error, [](auto const& error) {
    return std::string_view {error.what()} == "This input provider does not support ROOT input-file manifests";
  });
}

BOOST_AUTO_TEST_CASE(input_type_property_conversion)
{
  Gaudi::Property<Allen::InputFileType> type {"InputType", Allen::InputFileType::MDF};
  for (auto input : {"ROOT", "'ROOT'", "\"ROOT\""}) {
    BOOST_REQUIRE(type.fromString(input).isSuccess());
    BOOST_CHECK(type.value() == Allen::InputFileType::ROOT);
    BOOST_CHECK_EQUAL(type.toString(), "ROOT");
  }
  for (auto input : {"MDF", "'MDF'", "\"MDF\"", "RAW", "'RAW'", "\"RAW\""}) {
    BOOST_REQUIRE(type.fromString(input).isSuccess());
    BOOST_CHECK(type.value() == Allen::InputFileType::MDF);
    BOOST_CHECK_EQUAL(type.toString(), "MDF");
  }
  for (auto input : {"MEP", "root", "", "'unknown'"}) {
    BOOST_CHECK_THROW(type.fromString(input).ignore(), GaudiException);
    BOOST_CHECK(type.value() == Allen::InputFileType::MDF);
  }
  BOOST_CHECK_EQUAL(Allen::toString(Allen::InputFileType::ROOT), "ROOT");
}
