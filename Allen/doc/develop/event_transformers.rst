.. _event_transformers:

Scatter and gather event transformers
=====================================

Allen algorithms operate on multi-event device buffers, while the rest of the
Gaudi/LHCb stack (Moore) works with per-event objects on the transient event
store (TES).  The two worlds are bridged by the **scatter** and **gather**
multi-event transformers defined in
``Rec/Allen/src/EventTransformer.h``:

* ``LHCb::Algorithm::GatherEvent::MultiTransformer`` — multi-event input, single
  output.  It *gathers* per-event inputs from all event stores in the current
  slice and calls the user ``operator()`` once with one span per input.
* ``LHCb::Algorithm::ScatterEvent::MultiTransformer`` — single (multi-event)
  input, per-event outputs.  It calls the user ``operator()`` once and then
  *scatters* the returned per-event outputs back into the individual event
  stores of the slice.

These transformers are used by all the conversion algorithms in
``Allen/Rec/Allen/src/`` that move data between Allen buffers and LHCb event
model objects (tracks, clusters, vertices, raw banks, …).

How they fit into the Multi Event Scheduler
-------------------------------------------

The :ref:`multi_event_scheduler` executes algorithms either once per slice
(multi-event algorithms) or once per event (single-event algorithms).  The
scatter/gather transformers are multi-event Gaudi algorithms: they are
executed once per slice, but internally they visit every event store in the
slice.  They obtain the slice information from
``Allen::Scheduler::MultiEventContextExtension`` (see
``stream/gear/include/MultiEventContextExt.h``), which provides the number of
events in the slice, the per-event store handles and the Allen memory
managers.

Gather: many event stores → one multi-event output
--------------------------------------------------

``GatherEvent::MultiTransformer`` is declared as::

  template<typename Signature, typename Traits_ = Traits::Default>
  using MultiTransformer = Gaudi::Functional::details::GatherEvent::MultiTransformer<
      Signature, Traits::details::add_base_t<Traits_>>;

A concrete algorithm derives from it with a signature of the form::

  class MyGather final
    : public LHCb::Algorithm::GatherEvent::MultiTransformer<
        std::tuple<Out...>(const In&...)> { ... };

The ``operator()`` to implement receives the ``EventContext`` and one
``std::span<const In*>`` per input type.  Each span has one element per event
in the slice, collected from the individual event stores::

  std::tuple<Out...> operator()(
      const EventContext& ctx,
      const std::span<const In*>& in_span,
      ...) const override;

The transformer writes the returned ``std::tuple<Out...>`` to the output data
handles configured in the constructor.

A concrete example is ``GaudiAllenV3TracksToMEBasicParticlesRichStates`` in
``Rec/Allen/src/GaudiAllenV3TracksToTrackViews.cpp``.  It gathers per-event
``LHCb::Event::v3::Tracks`` from the slice, extracts the RICH ``SimpleKalmanState``
arrays, and produces concatenated ``Allen::device_buffer`` / ``Allen::host_buffer``
outputs for the multi-event slice::

  class GaudiAllenV3TracksToMEBasicParticlesRichStates final
    : public LHCb::Algorithm::GatherEvent::MultiTransformer<std::tuple<
        Allen::device_buffer<unsigned>,
        Allen::host_buffer<unsigned>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>,
        Allen::device_buffer<SimpleKalmanState>>(const InTracks&)> {
    ...
    std::tuple<...> operator()(
        const EventContext& ctx,
        const std::span<const InTracks*>& tracks_span) const override {
      // iterate tracks_span[e] over the events in the slice
      // fill host buffers, then copy to device buffers
    }
  };

Scatter: one multi-event input → many event stores
--------------------------------------------------

``ScatterEvent::MultiTransformer`` is declared as::

  template<typename Signature, typename Traits_ = Traits::Default>
  using MultiTransformer = Gaudi::Functional::details::ScatterEvent::MultiTransformer<
      Signature, Traits::details::add_base_t<Traits_>>;

A concrete algorithm derives from it with::

  class MyScatter final
    : public LHCb::Algorithm::ScatterEvent::MultiTransformer<
        std::tuple<Out...>(const In&...)> { ... };

The ``operator()`` to implement receives the ``EventContext`` and the (multi-event)
inputs, and must return a ``std::tuple<std::vector<Out>...>`` — one
``std::vector`` per output, with one entry per event in the slice::

  std::tuple<std::vector<Out>...> operator()(
      const EventContext& ctx,
      const In&... in) const override;

The transformer checks that each returned vector has exactly as many elements
as there are events in the slice, and then dispatches element ``i`` to the
``i``-th event store.

A concrete example is ``ConvertAllenVeloToV3Tracks`` in
``Rec/Allen/src/ConvertAllenVeloToV3Tracks.cpp``.  It receives the raw Allen
VELO device buffers for the slice, converts them to per-event
``LHCb::Event::v3::Tracks`` containers, and returns one forward and one
backward track container per event::

  std::tuple<std::vector<OutTracks>, std::vector<OutTracks>> operator()(
      const EventContext& /*ctx*/,
      const Allen::device_buffer<char>& dev_hits,
      const Allen::device_buffer<unsigned>& dev_track_offsets,
      const Allen::device_buffer<unsigned>& dev_track_hit_offsets,
      const Allen::device_buffer<char>& dev_state_data,
      const LHCb::UniqueIDGenerator& unique_id_gen) const override {
    // copy device buffers to host, loop over events, build tracks
    return {std::move(out_fwd), std::move(out_bwd)};
  }

Other scatter examples in ``Rec/Allen/src/`` include the converters for UT,
Long and Pr tracks, calo clusters, secondary vertices, PVs, raw reports and
lumi summaries, as well as the ``CompareRecAllen*`` algorithms.

The ``IsMultiEvent`` and ``IsMultiEventOutput`` properties
----------------------------------------------------------

Both transformers carry an ``IsMultiEvent`` property (``true`` by default) and
the scatter transformer additionally has ``IsMultiEventOutput``.  They tell
the scheduling machinery whether the algorithm's inputs/outputs are
multi-event buffers or per-event TES locations.  In the conversion algorithms
these are left at their defaults and are managed by the transformer base
classes.

Memory management
-----------------

Inside a transformer, use the Allen memory managers from the
``MultiEventContextExtension`` to allocate ``Allen::host_buffer`` and
``Allen::device_buffer`` objects::

  const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(ctx);
  Allen::host_buffer<unsigned> h_offsets {n_events + 1, ctxExt->memory_managers};
  Allen::device_buffer<unsigned> d_offsets {ctxExt->memory_managers};
  h_offsets.copy_to(d_offsets);

The buffers are scoped to the slice and returned to the scheduler's memory
managers at the end of the slice, so there is no per-event allocation on the
TES for the multi-event side.
