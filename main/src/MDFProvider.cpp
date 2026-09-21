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

#include <MDFProvider.h>
#include <stdexcept>
#include <iomanip>

#ifndef ALLEN_STANDALONE
namespace Allen {
  namespace {
    using namespace std::string_view_literals;
    constexpr auto inputFileTypeNames = std::array {"MDF"sv, "ROOT"sv};
  } // namespace

  std::string toString(InputFileType type) { return std::string {inputFileTypeNames.at(static_cast<size_t>(type))}; }

  std::ostream& toStream(InputFileType type, std::ostream& stream)
  {
    return stream << std::quoted(toString(type), '\'');
  }

  StatusCode parse(InputFileType& type, std::string_view input)
  {
    std::string name;
    return Gaudi::Parsers::parse(name, input).andThen([&]() -> StatusCode {
      // PyConf's FileFormats.RAW denotes MDF input to this provider.
      if (name == "RAW") name = "MDF";
      auto found = std::ranges::find(inputFileTypeNames, name);
      if (found == inputFileTypeNames.end()) return StatusCode::FAILURE;
      type = static_cast<InputFileType>(found - inputFileTypeNames.begin());
      return StatusCode::SUCCESS;
    });
  }
} // namespace Allen
LHCb::IO::InputFileManifest IInputProviderSvc::getInputFileManifest(size_t const, unsigned const) const
{
  throw std::logic_error {"This input provider does not support ROOT input-file manifests"};
}
#endif

void MDFProvider::init()
{
  // Preallocate prefetch buffer memory
  m_buffer_pool = std::make_unique<Allen::BufferPool<Allen::ReadBuffer>>(m_nslices, [this](Allen::ReadBuffer& buffer) {
    auto epb = m_config.events_per_buffer;
    buffer.event_buffer.resize((epb < 100 ? 120 : epb + 20) * average_event_size * bank_size_fudge_factor * kB);
  });

  if (m_config.n_transpose_threads > m_nslices) {
    debug_cout << "too many transpose threads requested with respect "
                  "to the number of read buffers; reducing the number of threads to "
               << m_nslices;
    m_config.n_transpose_threads = m_nslices;
  }

  // Start the transpose threads
  m_transpose_workers = std::make_unique<Allen::TransposeWorkers>(
    Allen::TransposeWorkers::Config {
      .n_threads = m_config.n_transpose_threads, .n_slices = m_nslices, .use_retina = m_config.use_retina},
    m_config.skip_banks);

  // Start prefetch thread
  if (m_config.use_ROOT_prefetcher) {
#ifndef ALLEN_STANDALONE
    m_prefetch_thread = std::make_unique<ROOTPrefetcher>(
      m_connections,
      m_transpose_workers.get(),
      m_buffer_pool.get(),
      m_config,
      this,
      m_eventTreeName.value(),
      m_eventBranches);
#else
    throw std::runtime_error("ROOT prefetcher is not supported in standalone mode");
#endif
  }
  else {
    m_prefetch_thread =
      std::make_unique<MDFPrefetcher>(m_connections, m_transpose_workers.get(), m_buffer_pool.get(), m_config);
  }
}

EventIDs MDFProvider::event_ids(size_t slice_index, std::optional<size_t> first, std::optional<size_t> last) const
{
  auto& slice = m_transpose_workers->slice(slice_index);
  auto const& ids = slice.batch.event_ids;
  return {ids.begin() + (first ? *first : 0), ids.begin() + (last ? *last : ids.size())};
}

std::vector<char> MDFProvider::event_mask(size_t slice_index) const
{
  auto& slice = m_transpose_workers->slice(slice_index);
  return slice.batch.event_mask;
}

BanksAndOffsets MDFProvider::banks(BankTypes bank_type, size_t slice_index) const
{
  auto& slice = m_transpose_workers->slice(slice_index);
  auto ib = to_integral(bank_type);
  return slice.banks[ib].banks_and_offsets();
}

std::tuple<bool, bool, bool, size_t, size_t, std::any> MDFProvider::get_slice(std::optional<unsigned int> timeout)
{
  auto [success, done, timed_out, slice_index, n_filled, odin] = m_transpose_workers->get_slice(timeout);
  return {!m_prefetch_thread->read_error(), done, timed_out, slice_index, n_filled, odin};
}

void MDFProvider::slice_free(size_t slice_index) { m_transpose_workers->slice_free(slice_index); }

void MDFProvider::event_sizes(
  size_t const slice_index,
  std::span<unsigned int const> const selected_events,
  std::span<size_t> sizes) const
{
  auto& slice = m_transpose_workers->slice(slice_index);
  for (size_t i = 0; i < static_cast<size_t>(selected_events.size()); ++i) {
    auto& raw_event = slice.batch.events[selected_events[i]];
    size_t size = 0;
    for (auto b : raw_event.banks()) {
      size += b->totalSize();
    }
    sizes[i] = size;
  }
}

void MDFProvider::copy_banks(size_t const slice_index, unsigned int const event, std::span<char> output_buffer) const
{
  auto& slice = m_transpose_workers->slice(slice_index);
  auto& raw_event = slice.batch.events[event];
  size_t offset = 0;
  for (auto b : raw_event.banks()) {
    assert(offset + b->totalSize() <= output_buffer.size());
    std::memcpy(output_buffer.data() + offset, b, b->totalSize());
    offset += b->totalSize();
  }
}
