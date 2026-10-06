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

#pragma once

#include "GaudiKernel/EventContext.h"
#include "Kernel/EventContextExt.h"
#include "EventMask.h"
#include "MemoryManager.cuh"
#include <vector>

namespace Allen::Scheduler {
  struct MultiEventContextExtension {
    unsigned slice_index {0};
    unsigned start_event {0};
    unsigned number_of_events {0};
    Allen::Context allen_context {};
    Allen::Store::memory_managers_t memory_managers {};
    size_t* stores {nullptr};
    std::span<EventMask> input_event_masks {};
    std::span<EventMask> event_masks {};

    MultiEventContextExtension(
      unsigned slice_index,
      unsigned start_event,
      unsigned number_of_events,
      Allen::Context allen_context,
      Allen::Store::memory_managers_t memory_managers,
      size_t* stores,
      std::span<EventMask> input_event_masks,
      std::span<EventMask> event_masks) :
      slice_index(slice_index),
      start_event(start_event), number_of_events(number_of_events), allen_context(allen_context),
      memory_managers(memory_managers), stores(stores), input_event_masks(input_event_masks), event_masks(event_masks)
    {}
  };

  inline MultiEventContextExtension& addContextExtensions(
    EventContext& evtCtx,
    unsigned slice_index,
    unsigned start_event,
    unsigned number_of_events,
    Allen::Context allen_context,
    Allen::Store::memory_managers_t& memory_managers,
    size_t* stores,
    std::span<EventMask> input_event_masks,
    std::span<EventMask> event_masks)
  {
    return evtCtx.emplaceExtension<LHCb::EventContextExtension>().emplaceSchedulerExtension<MultiEventContextExtension>(
      slice_index,
      start_event,
      number_of_events,
      allen_context,
      memory_managers,
      stores,
      input_event_masks,
      event_masks);
  }

  inline const MultiEventContextExtension* getSchedulerExtension(const EventContext& evtCtx)
  {
    if (evtCtx.hasExtension<LHCb::EventContextExtension>()) {
      const auto& ext = evtCtx.getExtension<LHCb::EventContextExtension>();
      if (ext.hasSchedulerExtension<MultiEventContextExtension>()) {
        return &ext.getSchedulerExtension<MultiEventContextExtension>();
      }
    }
    else {
      if (evtCtx.hasExtension<MultiEventContextExtension>()) {
        return &evtCtx.getExtension<MultiEventContextExtension>();
      }
    }
    return nullptr;
  }
} // namespace Allen::Scheduler
