/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// ----------------------------------------------------------------------------
// MultiEventScheduler: Gaudi service implementing IEventProcessor. Builds the
// algorithm control-flow/dependency graph, configures the slice worker pool and
// drives Allen's event loop by pulling slices from the input provider.
// ----------------------------------------------------------------------------

#include "GaudiAlg/FunctionalDetails.h"
#include "GaudiKernel/Algorithm.h"
#include "GaudiKernel/EventContext.h"
#include "GaudiKernel/IAlgManager.h"
#include "GaudiKernel/IDataBroker.h"
#include "GaudiKernel/IEventProcessor.h"
#include "GaudiKernel/Service.h"

#include "Kernel/ISchedulerConfiguration.h"
#include "Kernel/EventContextExt.h"
#include "Kernel/ThreadLocalAllocator.h"
#include "MultiEventContextExt.h"
#include "AllenMonitoring.h"

#include "EventMask.h"
#include "Scheduler.h"
#include "SliceThreadPool.h"
#include "InputProvider.h"
#include "IOutputWriter.h"
#include <Logger.h>
#include <TCK.h>

#include <atomic>
#include <chrono>
#include <format>
#include <optional>
#include <thread>
#include <experimental/scope>

namespace {
  // Simple wrapper to use Allen's host memory manager as a polymorphic allocator
  struct allen_host_allocator : public LHCb::Allocators::memory_resource {
    allen_host_allocator(Allen::Store::host_memory_manager_t* host_allocator) : m_host_allocator(host_allocator) {}
    void* allocate(size_t bytes, size_t) override { return m_host_allocator->reserve(bytes); }
    void deallocate(void* p, size_t, size_t) override { m_host_allocator->free(reinterpret_cast<char*>(p)); }
    bool is_equal(const LHCb::Allocators::memory_resource& other) const override
    {
      if (auto* other_alloc = dynamic_cast<const allen_host_allocator*>(&other)) {
        return m_host_allocator == other_alloc->m_host_allocator;
      }
      return false;
    }

  private:
    Allen::Store::host_memory_manager_t* m_host_allocator {nullptr};
  };
} // namespace

class MultiEventScheduler final : public extends<Service, IEventProcessor, LHCb::Interfaces::ISchedulerConfiguration> {

  // Properties:
  Gaudi::Property<std::vector<NodeDefinition>> m_compositeCFProperties {
    this,
    "CompositeCFNodes",
    {},
    "Specification of composite CF nodes"};
  Gaudi::Property<std::vector<std::string>> m_producers {
    this,
    "DataProducers",
    {},
    "List of algorithms to be used to resolve data dependencies"};

  Gaudi::Property<unsigned> m_nStreams {this, "NStreams", 1, "Number of parallel independent stream (sequences)."};
  Gaudi::Property<unsigned> m_eventsPerSlice {this, "EvtsPerSlice", 1000, "Number of events per slice."};
  Gaudi::Property<unsigned> m_repetitions {this, "Repetitions", 1, "Number of repetition per slice."};
  Gaudi::Property<int> m_device_id {this, "DeviceID", 0, "CUDA device ID."};

  Gaudi::Property<unsigned> m_device_memory {
    this,
    "DeviceMemoryPool",
    500,
    "Size of the per thread device memory pool in MB."};
  Gaudi::Property<unsigned> m_host_memory {
    this,
    "HostMemoryPool",
    500,
    "Size of the per thread host memory pool in MB."};

  Gaudi::Property<bool> m_profile_ranges {
    this,
    "EnableProfileRange",
    false,
    "Enable profile ranges around algorithm execution."};

  // Ordered sequence of algorithms:
  std::vector<Allen::Scheduler::ConfiguredAlgorithm> m_sequence;
  std::vector<Allen::Scheduler::BoolExpr> m_execution_masks;

  std::vector<Allen::Scheduler::LifetimeDependencies> m_reserve_args;
  std::vector<Allen::Scheduler::LifetimeDependencies> m_free_args;

  std::unique_ptr<Allen::Scheduler::SliceThreadPool> m_workers = nullptr;

  unsigned m_nextEvt = 0;
  std::atomic<bool> m_stopRequested {false};

  // Monitoring aggregation thread
  std::thread m_monitoring_thread;
  std::atomic<bool> m_monitoring_stop {false};

  void monitoringLoop()
  {
    Allen::set_device(m_device_id.value(), 0);
    while (!m_monitoring_stop.load(std::memory_order_acquire)) {
      Allen::Monitoring::AccumulatorManager::get()->mergeAndReset();
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  // TCK-from-ODIN support
  Gaudi::Property<bool> m_tckFromODIN {this, "TCKFromODIN", false, "Enable TCK-from-ODIN run changes"};
  Gaudi::Property<std::string> m_tckRepo {this, "TCKRepo", "", "Path to the TCK repository"};
  unsigned m_currentTCK = 0;
  std::string m_tck_str {};
  ConfigurationReader m_config_reader {};

  std::unordered_map<std::string, int> m_node_names_with_indices;
  std::vector<Allen::Scheduler::BoolExpr> m_node_passed;
  std::vector<std::string> m_printableDependencyTree;
  std::deque<Gaudi::Accumulators::BinomialCounter<uint32_t>> m_NodeStateCounters;

  // Services:
  IHiveWhiteBoard* m_whiteboard = nullptr;
  IDataProviderSvc* m_EDS = nullptr;

  ServiceHandle<IInputProviderSvc> m_inputProviderSvc {this, "InputProvider", "MDFProvider"};
  ServiceHandle<IOutputWriter> m_outputWriterSvc {this, "OutputWriter", "OutputWriter"};

public:
  using extends::extends;

  StatusCode initialize() override
  {
    const auto cuda_device_max_connections = m_nStreams.value() < 32 ? m_nStreams.value() : 32;
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", std::to_string(cuda_device_max_connections).c_str(), 1);
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);

    auto [device_set, device_name, device_memory_alignment, bus_id] = Allen::set_device(m_device_id.value(), 0);
    if (!device_set) {
      error() << "Failed to set device." << endmsg;
      return StatusCode::FAILURE;
    }

    info() << "Using device: " << device_name << endmsg;

    using Clock = std::chrono::high_resolution_clock;

    info() << "Start initialization" << endmsg;

    auto start_time = Clock::now();

    StatusCode sc = Service::initialize();
    if (!sc.isSuccess()) {
      error() << "Failed to initialize Service Base class." << endmsg;
      return StatusCode::FAILURE;
    }

    // Route Allen's logger through Gaudi's message service so that messages
    // emitted from Allen (including its input/transpose threads) are formatted,
    // filtered and serialised together with the rest of the application.
    logger::setMessageSvc(msgSvc().get(), name());

    std::map<std::string, NodeDefinition> cf_nodes;
    for (auto& nodeDef : m_compositeCFProperties) {
      cf_nodes.emplace(nodeDef.name, nodeDef);
    }

    auto appMgr = service<IAlgManager>("ApplicationMgr");
    auto algs = Allen::Scheduler::configured_algorithms(*appMgr, m_producers, cf_nodes);
    auto sorted = Allen::Scheduler::topological_sort(algs);
    m_execution_masks = Allen::Scheduler::find_execution_masks(sorted, algs, cf_nodes);
    if (msgLevel(MSG::INFO)) {
      info() << "Configured sequence:" << endmsg;
      const auto names = Allen::Scheduler::algorithm_names(sorted);
      for (const auto* alg : sorted) {
        info() << "   " << alg->alg->name() << " in: " << m_execution_masks[alg->index].to_string(names) << endmsg;
      }
    }
    m_sequence = Allen::Scheduler::finalizeConfiguration(sorted);
    std::tie(m_reserve_args, m_free_args) = Allen::Scheduler::calculate_lifetime_dependencies(sorted);

    initialize_printable_node_states(cf_nodes, algs);

    // Create stores
    m_whiteboard = serviceLocator()->service<IHiveWhiteBoard>("EventDataSvc");
    if (!m_whiteboard) {
      fatal() << "Error retrieving EventDataSvc interface IHiveWhiteBoard." << endmsg;
      return StatusCode::FAILURE;
    }

    m_EDS = serviceLocator()->service<IDataProviderSvc>("EventDataSvc");
    if (!m_EDS) {
      fatal() << "Error retrieving EventDataSvc interface IDataProviderSvc." << endmsg;
      return StatusCode::FAILURE;
    }

    Allen::Monitoring::AccumulatorManager::get()->initAccumulators(m_nStreams.value());

    // Start input/output threads:
    m_inputProviderSvc->startPrefetcher();
    m_outputWriterSvc.retrieve().ignore();

    // Create thread pool:
    m_workers = std::make_unique<Allen::Scheduler::SliceThreadPool>(
      m_nStreams,
      m_inputProviderSvc->n_slices(),
      [this](const unsigned i) -> Allen::Scheduler::WorkerContext {
        Allen::set_device(m_device_id.value(), i);

        Allen::Context context {};
        context.initialize(i);
        Allen::Store::host_memory_manager_t host_allocator {"host", m_host_memory * 1024 * 1024, 64};
        Allen::Store::device_memory_manager_t device_allocator {"device", m_device_memory * 1024 * 1024, 512};
        SlabAllocator<Allen::details::shared_buffer_metadata> meta_allocator {};
        SlabAllocator<Allen::details::type_erased_dependency> dep_allocator {};

        unsigned number_of_events = m_eventsPerSlice.value();

        std::vector<EventMask> input_event_masks;
        std::vector<EventMask> event_masks;
        input_event_masks.reserve(m_sequence.size());
        event_masks.reserve(m_sequence.size());
        for (unsigned j = 0; j < m_sequence.size(); j++) {
          input_event_masks.emplace_back(number_of_events);
          event_masks.emplace_back(number_of_events);
        }

        std::vector<size_t> stores {};
        stores.reserve(number_of_events);
        for (unsigned j = 0; j < number_of_events; j++) {
          const unsigned evt_id = i * number_of_events + j;
          stores.emplace_back(m_whiteboard->allocateStore(evt_id));
        }

        return {
          i,
          std::move(context),
          std::move(host_allocator),
          std::move(device_allocator),
          std::move(meta_allocator),
          std::move(dep_allocator),
          std::move(input_event_masks),
          std::move(event_masks),
          std::move(stores),
          m_inputProviderSvc.get(),
          m_outputWriterSvc.get()};
      },
      [this](const Allen::Scheduler::EventSlice& slice, Allen::Scheduler::WorkerContext& wrkCtx) {
        for (unsigned r = 0; r < m_repetitions.value(); r++) {
          if (m_profile_ranges.value()) Allen::rangePush("Process slice");

          // Setup memory managers:
          Allen::Store::memory_managers_t memory_managers {
            &wrkCtx.host_allocator,
            &wrkCtx.device_allocator,
            &wrkCtx.meta_allocator,
            &wrkCtx.dep_allocator,
            wrkCtx.context};

          // allen_host_allocator host_alloc {&wrkCtx.host_allocator};
          LHCb::Allocators::monotonic_memory_resource host_alloc {100 * 1024 * 1024};
          LHCb::Allocators::set_tls_default_resource(&host_alloc);

          // Create event context
          EventContext evtCtx {};
          Allen::Scheduler::addContextExtensions(
            evtCtx,
            slice.slice_index,
            slice.start_event,
            slice.number_of_events,
            wrkCtx.context,
            memory_managers,
            wrkCtx.stores.data(),
            wrkCtx.input_event_masks,
            wrkCtx.event_masks);

          {
            LHCb::tla::vector<Allen::host_buffer<mask_t>> host_event_lists {};
            LHCb::tla::vector<Allen::device_buffer<mask_t>> dev_event_lists {};
            host_event_lists.reserve(m_sequence.size());
            dev_event_lists.reserve(m_sequence.size());
            for (unsigned i = 0; i < m_sequence.size(); i++) {
              host_event_lists.emplace_back(memory_managers);
              dev_event_lists.emplace_back(memory_managers);
            }

            auto store_guard = std::experimental::scope_exit([&] {
              Allen::synchronize(wrkCtx.context);
              Allen::Monitoring::AccumulatorManager::get()->streamDone(wrkCtx.stream_id);

              if (m_profile_ranges.value()) Allen::rangePush("Clear stores");
              // Clear stores
              for (unsigned i = 0; i < slice.number_of_events; i++) {
                m_whiteboard->clearStore(wrkCtx.stores[i]).ignore();
              }
              if (m_profile_ranges.value()) {
                Allen::rangePop(); // end Clear stores
                Allen::rangePop(); // end Process slice
              }
            });

            Allen::Monitoring::AccumulatorManager::get()->synchronizeStream(wrkCtx.stream_id);

            // Run sequence:
            for (auto& alg : m_sequence) {
              if (m_profile_ranges.value()) Allen::rangePush(alg.alg->name().c_str());

              bool isMultiEvent = alg.isMultiEvent;

              auto& input_mask = wrkCtx.event_masks[alg.index];
              input_mask = EventMask {slice.number_of_events}; // FIXME: better reuse
              m_execution_masks[alg.index].evaluate(input_mask, wrkCtx.event_masks);
              wrkCtx.input_event_masks[alg.index] = input_mask; // copy

              if (isMultiEvent) {
                evtCtx.set(slice.start_event, wrkCtx.stores[0]);
                Gaudi::Hive::setCurrentContext(evtCtx);
                m_whiteboard->selectStore(evtCtx.slot()).ignore();

                if (alg.inputMaskHandle) {
                  // keep the host/device masks in a cache, for reuse between algs:
                  if (m_execution_masks[alg.index].alg != -1) {
                    auto mask_id = m_execution_masks[alg.index].alg;
                    host_event_lists[alg.index] = host_event_lists[mask_id];
                    dev_event_lists[alg.index] = dev_event_lists[mask_id];
                  }
                  else {
                    input_mask.to_event_list(host_event_lists[alg.index]);
                    dev_event_lists[alg.index] = host_event_lists[alg.index].to_device();
                  }

                  m_EDS->unregisterObject(alg.inputMaskHandle->objKey()).ignore();
                  auto event_list = dev_event_lists[alg.index]; // need to copy before moving
                  alg.inputMaskHandle->put(std::move(event_list));
                }

                alg.alg->execute(evtCtx).ignore();

                if (alg.outputMaskHandle) {
                  const auto& event_list = *alg.outputMaskHandle->get();
                  Allen::host_buffer<mask_t> host_event_list {memory_managers};
                  event_list.copy_to(host_event_list);
                  input_mask.from_event_list(host_event_list);
                  host_event_lists[alg.index] = host_event_list;
                  dev_event_lists[alg.index] = event_list;
                }
              }
              else {
                EventMask output_mask {slice.number_of_events};
                for (unsigned i : input_mask) { // iterate only over valid events
                  const unsigned evt_id = slice.start_event + i;

                  evtCtx.set(evt_id, wrkCtx.stores[i]);
                  Gaudi::Hive::setCurrentContext(evtCtx);
                  m_whiteboard->selectStore(evtCtx.slot()).ignore();

                  auto ret = alg.alg->execute(evtCtx);

                  bool filterpassed = [&] {
                    if (ret == Gaudi::Functional::FilterDecision::PASSED) return true;
                    if (ret == Gaudi::Functional::FilterDecision::FAILED) return false;
                    if (ret == StatusCode::SUCCESS) return true;
                    throw GaudiException("Error in algorithm execute: " + alg.alg->name(), alg.alg->name(), ret);
                  }();

                  if (filterpassed) output_mask.set(i);
                }
                input_mask = output_mask;
              }

              // Free unused arguments
              for (DataObjID const* id : m_free_args[alg.index]) {
                m_EDS->unregisterObject(id->key()).ignore();
              }

              if (m_profile_ranges.value()) {
                Allen::synchronize(wrkCtx.context);
                Allen::rangePop(); // end Algorithm
              }
            }
          }
        }

        // Update node passed counters
        EventMask executed {slice.number_of_events};
        EventMask passed {slice.number_of_events};
        for (unsigned i = 0; i < m_node_passed.size(); ++i) {
          m_node_passed[i].evaluate(executed, wrkCtx.input_event_masks);
          m_node_passed[i].evaluate(passed, wrkCtx.event_masks);
          m_NodeStateCounters[i] += {passed.popcount(), executed.popcount()};
        }
      });

    auto end_time = Clock::now();

    // Clearly inform about the level of concurrency
    info() << "Concurrency level information:" << endmsg;
    info() << " o Number of streams: " << m_nStreams.value() << endmsg;
    info() << " o Events per slice: " << m_eventsPerSlice.value() << endmsg;
    info() << " o Number of events slots: " << m_whiteboard->getNumberOfStores() << endmsg;

    info() << "---> End of Initialization. "
           << "This took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
           << " ms" << endmsg;

    return sc;
  }

  StatusCode start() override
  {
    StatusCode sc = Service::start();
    if (sc.isFailure()) return sc;
    m_stopRequested = false;
    for (auto& alg : m_sequence) {
      sc = alg.alg->sysStart();
      if (sc.isFailure()) {
        error() << "Unable to start algorithm: " << alg.alg->name() << endmsg;
        return sc;
      }
    }

    // Start the monitoring aggregation thread
    if (!m_monitoring_thread.joinable()) {
      m_monitoring_stop = false;
      m_monitoring_thread = std::thread(&MultiEventScheduler::monitoringLoop, this);
    }

    return sc;
  }
  StatusCode stop() override
  {
    m_stopRequested = true;
    if (m_monitoring_thread.joinable()) {
      m_monitoring_stop = true;
      m_monitoring_thread.join();
    }
    for (auto& alg : m_sequence) {
      StatusCode sc = alg.alg->sysStop();
      if (sc.isFailure()) {
        error() << "Unable to stop algorithm: " << alg.alg->name() << endmsg;
        return sc;
      }
    }
    Allen::Monitoring::AccumulatorManager::get()->mergeAndReset();
    return Service::stop();
  }
  StatusCode reinitialize() override { return StatusCode::FAILURE; }
  StatusCode finalize() override
  {
    logger::clearMessageSvc();

    // print the counters
    info() << "StateTree: CFNode   #executed  #passed\n";
    auto maxTreeWidth = (*std::max_element(
                           begin(m_printableDependencyTree),
                           end(m_printableDependencyTree),
                           [](std::string_view s, std::string_view t) { return s.size() < t.size(); }))
                          .size();
    for (unsigned i = 0; i < m_printableDependencyTree.size(); ++i) {
      auto const& treeEntry = m_printableDependencyTree[i];
      info() << std::left << std::setw(maxTreeWidth + 1) << treeEntry << m_NodeStateCounters[i] << '\n';
    }
    info() << endmsg;

    for (auto& alg : m_sequence) {
      alg.alg->sysFinalize().ignore();
    }
    m_sequence.clear();
    return Service::finalize();
  }

  void initialize_printable_node_states(
    std::map<std::string, NodeDefinition>& cf_nodes,
    std::unordered_map<std::string, Allen::Scheduler::AlgEntry>& algorithms)
  {
    m_node_passed.clear();
    m_printableDependencyTree.clear();
    m_NodeStateCounters.clear();

    auto print_indented = [&](const std::string& node_name, int const currentIndent, auto& itself) -> void {
      // to recursively call this lambda, use auto& itself
      Allen::Scheduler::BoolExpr trueMask;
      trueMask.type = Allen::Scheduler::BoolExpr::NodeType::CONST_TRUE;
      m_node_passed.emplace_back(
        Allen::Scheduler::get_tree_for_node(node_name, m_execution_masks, algorithms, cf_nodes, trueMask));
      m_node_names_with_indices.emplace(node_name, m_printableDependencyTree.size());

      auto it = cf_nodes.find(node_name);
      if (it == cf_nodes.end()) {
        m_printableDependencyTree.emplace_back(fmt::format("{:{}}{} ", "", currentIndent, node_name));
      }
      else {
        auto& nodeDef = cf_nodes.at(node_name);
        const nodeType type = toNodeType(nodeDef.type);
        m_printableDependencyTree.emplace_back(
          fmt::format("{:{}}{}: {} ", "", currentIndent, toString(type), node_name));
        for (unsigned i = 0; i < nodeDef.children.size(); i++) {
          itself(nodeDef.children[i], currentIndent + 1, itself);
        }
      }
    };

    auto top_nodes = Allen::Scheduler::get_top_nodes(cf_nodes);
    for (auto& node_name : top_nodes) {
      print_indented(node_name, 0, print_indented);
    }
    m_NodeStateCounters.resize(m_node_passed.size());
  }

  // IEventProcessor interface:

  StatusCode nextEvent([[maybe_unused]] int maxevt) override
  {
    using Clock = std::chrono::high_resolution_clock;

    info() << "Called nextEvent with: " << maxevt << endmsg;

    auto start_time = Clock::now();
    while (true) {
      if (m_stopRequested) {
        info() << "Stop requested, exiting event loop" << endmsg;
        break;
      }
      auto timeout = 1000;
      auto [success, eof, timed_out, slice_index, n_filled, odin] = m_inputProviderSvc->get_slice(timeout);
      if (!timed_out && success && n_filled != 0) {
        // Check for TCK change from ODIN
        if (m_tckFromODIN.value() && odin.has_value()) {
          auto odin_data = std::any_cast<std::span<unsigned const>>(odin);
          LHCb::ODIN odin_obj {odin_data};
          auto new_tck = odin_obj.triggerConfigurationKey();

          // Load new TCK
          std::string new_tck_str = fmt::format("{:#010x}", new_tck);
          auto [new_config, new_source, new_tck_info] = Allen::load_tck(m_tckRepo, new_tck_str);
          ConfigurationReader new_config_reader {new_config};

          if (new_tck != m_currentTCK) {
            if (m_currentTCK != 0) {
              info() << "Detected tck change from " << m_tck_str << " to " << new_tck_str << endmsg;

              // Check compatibility
              if (!compatible_configurations(m_config_reader, new_config_reader)) {
                error() << "TCKs " << m_tck_str << " and " << new_tck_str << " are not compatible for fast run change."
                        << endmsg;
                break;
              }

              // Wait for all workers to finish before reconfiguring
              m_workers->wait_for_completion();
            }

            // Reconfigure algorithms
            for (auto& alg : m_sequence) {
              auto c = new_config_reader.params().find(alg.alg->name());
              if (c != new_config_reader.params().end()) {
                for (auto const& [n, r] : c->second) {
                  alg.alg->setProperty(n, r).ignore();
                }
              }
            }

            // Update state
            m_tck_str = new_tck_str;
            m_config_reader = new_config_reader;
            m_currentTCK = new_tck;
          }
        }

        if (m_repetitions == 1) {
          m_workers->submit({static_cast<unsigned>(slice_index), 0, static_cast<unsigned>(n_filled)});
          m_nextEvt += n_filled;
        }
        else {
          for (unsigned r = 0; r < m_nStreams; r++) {
            m_workers->submit({static_cast<unsigned>(slice_index), 0, static_cast<unsigned>(n_filled)});
            m_nextEvt += n_filled * m_repetitions;
          }
          break;
        }
      }
      else if (!success || eof) {
        break;
      }
    }
    m_workers->wait_for_completion();
    auto end_time = Clock::now();

    // TODO: warmup phase, cool down phase, measure only steady state, etc...
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double throughput = static_cast<double>(m_nextEvt) / totalTime * 1000.0;
    info() << "Execution time: " << totalTime << " ms. Throughput: " << std::format("{:.2f}", throughput) << " events/s"
           << endmsg;
    return StatusCode::SUCCESS;
  }
  StatusCode executeEvent(EventContext&&) override { return StatusCode::SUCCESS; }
  StatusCode executeRun(int maxevt) override
  {
    m_stopRequested = false;
    return nextEvent(maxevt);
  }
  StatusCode stopRun() override
  {
    m_stopRequested = true;
    return StatusCode::SUCCESS;
  }
  EventContext createEventContext() override { return {}; }

  // ISchedulerConfiguration interface:

  std::unordered_map<std::string, int> getNodeNamesWithIndices() const override { return m_node_names_with_indices; }

  LHCb::Interfaces::ISchedulerConfiguration::NodeState getNodeState(
    [[maybe_unused]] const EventContext& evtCtx,
    [[maybe_unused]] unsigned index) const override
  {
    const auto* ext = Allen::Scheduler::getSchedulerExtension(evtCtx);
    EventMask executed {ext->number_of_events};
    EventMask passed {ext->number_of_events};
    m_node_passed[index].evaluate(executed, ext->input_event_masks);
    m_node_passed[index].evaluate(passed, ext->event_masks);
    // executionCtr is 0 if the node was executed at least once, and 1 if it was not executed at all
    return {executed.test(evtCtx.evt()) ? 0u : 1u, passed.test(evtCtx.evt())};
  }
};

DECLARE_COMPONENT(MultiEventScheduler)
