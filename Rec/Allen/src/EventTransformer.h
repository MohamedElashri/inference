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
// Gaudi transformers bridging multi-event Allen slices and per-event TES data:
// GatherEvent::MultiTransformer gathers per-event inputs from every store of a
// slice into one multi-event output; ScatterEvent::MultiTransformer scatters a
// multi-event input back into per-event outputs.
// ----------------------------------------------------------------------------
#pragma once

#include "GaudiAlg/Transformer.h"
#include "Kernel/ThreadLocalAllocator.h"
#include "MultiEventContextExt.h"
#include "LHCbAlgs/Traits.h"

#include <type_traits>

namespace Gaudi::Functional::details {
  namespace GatherEvent {
    // Multi Event input => Single Event output
    template<typename Signature, typename Traits_>
    struct MultiTransformer;

    template<typename... Out, typename... In, typename Traits_>
    struct MultiTransformer<std::tuple<Out...>(const In&...), Traits_>
      : DataHandleMixin<std::tuple<Out...>, std::tuple<In...>, Traits_> {
      using DataHandleMixin<std::tuple<Out...>, std::tuple<In...>, Traits_>::DataHandleMixin;

      StatusCode execute(const EventContext& ctx) const override final
      {
        try {
          auto gathered_inputs = gatherInputs(ctx);

          writeOutputs(std::apply(
            [this, &ctx](auto&... vecs) { return (*this)(ctx, std::span<const In*>(vecs)...); }, gathered_inputs));

          return StatusCode::SUCCESS;
        } catch (GaudiException& e) {
          if (e.code().isFailure()) this->error() << e.tag() << " : " << e.message() << endmsg;
          return e.code();
        }
      }

      // All inputs get turned into vectors with elements comming from different stores
      virtual std::tuple<Out...> operator()(const EventContext&, const std::span<const In*>&...) const = 0;

    private:
      template<typename Handle>
      auto collectFromSlots(const Handle& handle, const EventContext& ctx) const
      {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(get(handle, *this, ctx))>>;

        const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
        const unsigned n_events = ctxExt ? ctxExt->number_of_events : 1;

        LHCb::tla::vector<const T*> collected {};
        for (unsigned i = 0; i < n_events; i++) {
          EventContext evtCtx {};
          evtCtx.set(ctx.evt() + i, ctxExt ? ctxExt->stores[i] : ctx.slot());
          this->whiteboard()->selectStore(evtCtx.slot()).ignore();
          Gaudi::Hive::setCurrentContext(evtCtx);

          collected.push_back(&get(handle, *this, evtCtx));
        }
        // Reset to current ctx:
        this->whiteboard()->selectStore(ctx.slot()).ignore();
        Gaudi::Hive::setCurrentContext(ctx);
        return collected;
      }

      auto gatherInputs(const EventContext& ctx) const
      {
        return std::apply(
          [&](const auto&... inputs) { return std::make_tuple(collectFromSlots(inputs, ctx)...); }, this->m_inputs);
      }

      void writeOutputs(std::tuple<Out...>&& results) const
      {
        std::apply(
          [this](auto&&... outputs) {
            std::apply([&outputs...](auto&&... handles) { (put(handles, std::move(outputs)), ...); }, this->m_outputs);
          },
          std::move(results));
      }

      Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
    };
  } // namespace GatherEvent

  namespace ScatterEvent {
    // Single Event input => Multi Event output
    template<typename Signature, typename Traits_>
    struct MultiTransformer;

    template<typename... Out, typename... In, typename Traits_>
    struct MultiTransformer<std::tuple<Out...>(const In&...), Traits_>
      : DataHandleMixin<std::tuple<Out...>, std::tuple<In...>, Traits_> {
      using DataHandleMixin<std::tuple<Out...>, std::tuple<In...>, Traits_>::DataHandleMixin;

      StatusCode execute(const EventContext& ctx) const override final
      {
        try {
          // Call the user's operator() which returns vectors of outputs for each event
          auto outputs_vectors =
            std::apply([this, &ctx](auto&... ins) { return (*this)(ctx, get(ins, *this, ctx)...); }, this->m_inputs);

          // Dispatch each output to its respective event store
          dispatchOutputs(std::move(outputs_vectors), ctx);

          return StatusCode::SUCCESS;
        } catch (GaudiException& e) {
          if (e.code().isFailure()) this->error() << e.tag() << " : " << e.message() << endmsg;
          return e.code();
        }
      }

      // All outputs get turned into vectors with elements going into different stores
      virtual std::tuple<std::vector<Out>...> operator()(const EventContext&, const In&...) const = 0;

    private:
      template<typename Handle, typename T>
      void dispatchToSlots(const Handle& handle, std::vector<T>&& outputs, const EventContext& ctx) const
      {
        const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
        const unsigned n_events = ctxExt ? ctxExt->number_of_events : 1;

        // Check that the number of outputs matches the number of events
        if (outputs.size() != n_events) {
          throw GaudiException(
            "ScatterEvent::MultiTransformer: Output vector size (" + std::to_string(outputs.size()) +
              ") does not match number of events (" + std::to_string(n_events) + ")",
            "SizeMismatch",
            StatusCode::FAILURE);
        }

        // Dispatch each output to its respective event context
        for (unsigned i = 0; i < n_events; i++) {
          if (ctxExt) {
            EventContext evtCtx {};
            evtCtx.set(ctx.evt() + i, ctxExt->stores[i]);
            std::ignore = this->whiteboard()->selectStore(evtCtx.slot());
            Gaudi::Hive::setCurrentContext(evtCtx);
          }

          // Move the output to its destination
          put(handle, std::move(outputs[i]));
        }

        // Reset to current ctx
        std::ignore = this->whiteboard()->selectStore(ctx.slot());
        Gaudi::Hive::setCurrentContext(ctx);
      }

      void dispatchOutputs(std::tuple<std::vector<Out>...>&& outputs_vectors, const EventContext& ctx) const
      {
        std::apply(
          [this, &ctx](auto&&... outputs) {
            std::apply(
              [&](auto&&... handles) { (dispatchToSlots(handles, std::move(outputs), ctx), ...); }, this->m_outputs);
          },
          std::move(outputs_vectors));
      }

      Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};
      Gaudi::Property<bool> m_isMultiEventOutput {this, "IsMultiEventOutput", false, ""};
    };
  } // namespace ScatterEvent
} // namespace Gaudi::Functional::details

namespace LHCb::Algorithm {
  namespace GatherEvent {
    template<typename Signature, typename Traits_ = Traits::Default>
    using MultiTransformer =
      Gaudi::Functional::details::GatherEvent::MultiTransformer<Signature, Traits::details::add_base_t<Traits_>>;
  } // namespace GatherEvent

  namespace ScatterEvent {
    template<typename Signature, typename Traits_ = Traits::Default>
    using MultiTransformer =
      Gaudi::Functional::details::ScatterEvent::MultiTransformer<Signature, Traits::details::add_base_t<Traits_>>;
  } // namespace ScatterEvent
} // namespace LHCb::Algorithm
