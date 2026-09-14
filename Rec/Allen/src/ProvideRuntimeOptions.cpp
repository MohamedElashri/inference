/***************************************************************************** \
 * (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// ----------------------------------------------------------------------------
// Transformer assembling the RuntimeOptions handed to Allen algorithms for the
// current slice (input provider, slice index, event range, memory managers and
// optional checker).
// ----------------------------------------------------------------------------
#include <array>

// Gaudi
#include <GaudiAlg/Transformer.h>
#include <GaudiAlg/FunctionalUtilities.h>
#include <Gaudi/Accumulators.h>
#include "Kernel/ThreadLocalAllocator.h"

// Allen
#include <Constants.cuh>
#include <RuntimeOptions.h>
#include "AllenROOTService.h"
#include "MultiEventContextExt.h"
#include "InputProvider.h"
#include "CheckerInvoker.h"

class ProvideRuntimeOptions final : public Gaudi::Functional::Transformer<RuntimeOptions(const EventContext&)> {

public:
  /// Standard constructor
  ProvideRuntimeOptions(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override
  {
    auto status = Transformer::initialize();
    if (!status.isSuccess()) return status;
    if (m_enable_checker) m_checkerInvoker = std::make_unique<CheckerInvoker>();

    // Pre-retrieve the services single-threaded. ServiceHandle lazily caches
    // its SmartIF<ISvcLocator> on first use without synchronization, so doing
    // the retrieval here avoids a data race when this algorithm is executed
    // concurrently by the worker threads.
    if (m_inputProviderSvc.retrieve().isFailure()) {
      fatal() << "Error retrieving IInputProviderSvc." << endmsg;
      return StatusCode::FAILURE;
    }
    if (m_rootService.retrieve().isFailure()) {
      fatal() << "Error retrieving AllenROOTService." << endmsg;
      return StatusCode::FAILURE;
    }

    return StatusCode::SUCCESS;
  }

  StatusCode finalize() override
  {
    if (m_enable_checker) {
      m_checkerInvoker->report(m_n_events_processed.value());
      m_checkerInvoker.reset();
    }
    return Transformer::finalize();
  }

  /// Algorithm execution
  RuntimeOptions operator()(const EventContext&) const override;

private:
  std::unique_ptr<CheckerInvoker> m_checkerInvoker;
  mutable Gaudi::Accumulators::Counter<> m_n_events_processed {this, "Number of events processed"};

  ServiceHandle<AllenROOTService> m_rootService {this, "AllenROOTService", "AllenROOTService"};
  ServiceHandle<IInputProviderSvc> m_inputProviderSvc {this, "InputProvider", "MDFProvider"};

  Gaudi::Property<bool> m_enable_checker {this, "EnableChecker", false, "Enable the checker invoker."};
  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
};

ProvideRuntimeOptions::ProvideRuntimeOptions(const std::string& name, ISvcLocator* pSvcLocator) :
  Transformer(name, pSvcLocator, KeyValue {"RuntimeOptionsLocation", "Allen/Stream/RuntimeOptions"})
{}

RuntimeOptions ProvideRuntimeOptions::operator()(const EventContext& evtCtx) const
{
  const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(evtCtx);
  const unsigned number_of_repetitions = 1;
  const bool param_inject_mem_fail = false;
  const bool mep_layout = m_inputProviderSvc->layout() == IInputProvider::Layout::MEP;
  auto provider = std::shared_ptr<IInputProvider>(m_inputProviderSvc.get(), [](auto*) {});
  m_n_events_processed += ctxExt->number_of_events;
  return RuntimeOptions {
    provider,
    ctxExt->slice_index,
    {ctxExt->start_event, ctxExt->start_event + ctxExt->number_of_events},
    number_of_repetitions,
    mep_layout,
    param_inject_mem_fail,
    m_enable_checker ? m_checkerInvoker.get() : nullptr,
    m_rootService->rootService()};
}

DECLARE_COMPONENT(ProvideRuntimeOptions)
