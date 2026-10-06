.. _multi_event_scheduler:

Multi Event Scheduler architecture
==================================

Allen runs natively on GPU through its **Multi Event Scheduler** (MES).
MES replaces the old ``AllenApplication`` / ``AllenEventLoop`` steering and the
standalone ``./Allen`` executable as the production way of running Allen.
It keeps the existing Allen algorithm and sequence machinery but executes it
through a Gaudi service that processes
*multiple events per GPU launch* (a "slice") and runs several slices
concurrently on independent streams.

This page describes the components that make up the MES run-time and how they
fit together.

Overview
--------

At the highest level, one Gaudi-Allen application contains:

* an **input provider** that reads events and stages them into slices,
* the **Multi Event Scheduler** itself, which owns a pool of worker threads
  (one per GPU stream) and drives the configured Allen sequence,
* an **output writer** that drains the accepted events from the per-stream
  output queues,
* a **monitoring aggregation** thread that periodically merges GPU-side
  monitoring accumulators into the Gaudi monitoring hub, and
* the usual Gaudi services (EventDataSvc / HiveWhiteBoard, conditions,
  geometry, message and monitoring services).

.. mermaid::

  graph LR
    subgraph Input
      PF[Prefetch thread<br>MDF / ROOT / MEP]
      TR[Transpose threads]
    end

    subgraph MES
      S[MultiEventScheduler]
      Q[Slice queue]
      W1[Worker stream 1]
      W2[Worker stream 2]
      WN[Worker stream N]
    end

    subgraph Output
      OW[Output writer<br>File / ZMQ / MBM]
      RB[Per-stream ring buffers]
    end

    subgraph Monitoring
      AM[AccumulatorManager]
      MT[Monitoring aggregation thread]
      GH[Gaudi monitoring hub]
    end

    PF --> TR
    TR --> S
    S --> Q
    Q --> W1
    Q --> W2
    Q --> WN
    W1 --> RB
    W2 --> RB
    WN --> RB
    RB --> OW
    W1 --> AM
    W2 --> AM
    WN --> AM
    AM --> MT
    MT --> GH

The scheduler
-------------

``MultiEventScheduler`` (``Rec/Allen/src/MultiEventScheduler.cpp``) is a Gaudi
service implementing ``IEventProcessor`` and
``LHCb::Interfaces::ISchedulerConfiguration``.  It is created by the Python
configuration layer through ``Allen.config.make_MultiEventScheduler`` and
injected into the ``ApplicationMgr`` with
``create_appMgr.global_bind(make_scheduler=make_MultiEventScheduler)``.

Its main properties are:

``CompositeCFNodes``
    The control-flow ``CompositeNode`` definitions produced by the Allen
    sequence configuration.
``DataProducers``
    The list of algorithms used to resolve data dependencies.
``NStreams``
    Number of independent worker streams (GPU streams), one per worker thread.
``EvtsPerSlice``
    Number of events processed per slice (per GPU launch).
``Repetitions``
    Number of times each slice is processed.  A value greater than one is used
    for benchmarking/throughput measurement.  When measuring the throughput,
    the first and last ``NStreams`` slice iterations are excluded to discard
    the lazy loading of non-event data (geometry, conditions, ...) triggered by
    the first iterations and the draining effects at the end of the run.
``DeviceMemoryPool`` / ``HostMemoryPool``
    Size (in MB) of the per-stream device and host memory pools.
``DeviceID``
    CUDA device to use.
``TCKFromODIN`` / ``TCKRepo``
    Enable TCK-from-ODIN fast run changes and the TCK repository to load from.

The scheduler initialises the Allen sequence exactly once, ordering the algorithms by doing
a topological sort of the dependency graph. It then creates a
``SliceThreadPool`` with ``NStreams`` workers.

Each worker owns a ``WorkerContext`` containing:

* an ``Allen::Context`` bound to a dedicated GPU stream,
* per-stream host and device memory managers (``host_memory_manager_t`` /
  ``device_memory_manager_t``),
* slab allocators for shared-buffer metadata and type-erased dependencies,
* the per-algorithm input/output ``EventMask`` vectors

Slices are submitted to a shared queue.  A worker pops a slice, runs the full
Allen sequence over it, and then declares the slice free so the input provider
can refill it.  If a slice does not fit into the reserved device memory, the
worker catches ``MemoryException``, splits the slice into two halves and
resubmits them.  A slice that can no longer be split is handled by the
single-event passthrough path.

Execution model
~~~~~~~~~~~~~~~

Within a slice, the scheduler distinguishes two kinds of algorithms:

* **multi-event algorithms** (the GPU algorithms) run once for the whole
  slice.  They receive an event list (an ``EventMask``) and execute over all
  events in the slice in a single GPU launch.
* **single-event algorithms** run once per valid event, looping over the
  events in the slice.  These are used for the small number of host-side
  Gaudi algorithms that expose a per-event interface (for example the TES
  data providers described below).

After each algorithm the scheduler frees the dependencies whose lifetime is
over, and at the end of the slice it clears the per-event HiveWhiteBoard
stores.

Event context
~~~~~~~~~~~~~

Multi-event state is passed through ``Allen::Scheduler::MultiEventContextExtension``
(``stream/gear/include/MultiEventContextExt.h``), which is attached to the
Gaudi ``EventContext``.  It carries the slice index, the first event number,
the number of events, the Allen context and memory managers, the store
handles, and the input/output event masks.

Input providers
---------------

Input providers implement the ``IInputProviderSvc`` interface
(``main/include/InputProvider.h``).  They are responsible for reading events,
transposing them into per-bank-type slices, and handing ready slices to the
scheduler.  The common interface covers:

* ``layout()`` — whether slices are in the transposed Allen layout or MEP
  layout,
* ``get_slice()`` / ``slice_free()`` — hand a filled slice to the scheduler
  and return it once processed,
* ``banks()`` / ``event_sizes()`` / ``copy_banks()`` — access to the
  transposed banks for multi-event algorithms,
* ``getRawEvent()`` / ``getODIN()`` / ``getEventBranches()`` — per-event
  access used by the single-event TES providers.

Two production implementations exist:

``MDFProvider``
    Reads MDF or ROOT files.  It is a Gaudi service (``MDFProvider``) with
    properties ``NSlices``, ``EventsPerSlice``, ``Connections``, ``EvtMax``,
    ``InputType`` (``MDF`` or ``ROOT``), ``UseRetina``, ``TransposeThreads``
    and ``EventBranches``.  Internally it runs a prefetch thread and a pool
    of transpose threads (``main/include/MDFProvider.h``,
    ``main/src/MDFProvider.cpp``).

``MEPProvider``
    Reads MEP data, either from files, from MBM buffers or through MPI
    (``MooreOnline/AllenOnline/src/MEPProvider.cpp``).  This is the provider
    used online and by the MooreOnline testbench.  Its main properties are
    ``Source`` (``Files``, ``MBM`` or ``MPI``), ``NSlices``,
    ``EventsPerSlice``, ``Connections``, ``BufferConfig``, ``TransposeMEPs``
    and the MBM connection settings.

The single-event TES providers are small Gaudi transformers that expose input
data on the transient event store for algorithms that need it:

``ProvideRawEvent``
    Publishes a ``LHCb::RawEvent`` built from the slice's banks for one event
    (``Rec/Allen/src/ProvideRawEvent.cpp``).

``ProvideEventBranches``
    Publishes ROOT event branches from the input file for one event, and also
    returns the ``RawEvent`` (``Rec/Allen/src/ProvideEventBranches.cpp``).

``ProvideODIN`` / ``AllenODINProducer``
    Expose the ODIN bank for the event/slice.

Which provider is used is decided by the Python configuration
(``Allen.config.run_allen`` and the MooreOnline ``AllenConfig.py``).  The
scheduler always talks to the provider through the ``IInputProviderSvc``
service handle, so the concrete provider is just a configuration detail.

Output writers
--------------

Output writers implement the ``IOutputWriter`` interface
(``main/include/IOutputWriter.h``).  They run a dedicated output thread that
periodically drains the per-stream ring buffers produced by the Allen
persistency algorithms, so that the worker streams are never blocked by I/O.

The base ``OutputWriter`` service owns the ring buffers
(``OutputManager``) and the single-event passthrough object.  Concrete
writers are selected by the ``OutputConnection`` string:

``OutputWriter`` (default)
    Consumes the write queue without writing anything.  Used when no output
    file is requested or for ROOT output handled by the stream writer.

``FileOutputWriter``
    Writes the selected events to an output file (``FileWriter``).

``ZMQOutputWriter``
    Sends the selected events over ZeroMQ (``tcp://…`` connection).

``Allen__MBMOutput``
    Writes into an MBM buffer online (``mbm://…`` connection).  It exposes the
    MBM partition/buffer settings documented in ``Allen.config.AllenOptions``.

The writer properties include ``NStreams``, ``RBCapacity``,
``OutputConnection``, ``TCK``, ``TaskId``, ``routingbit_map`` and
``DoChecksum``.

Monitoring
----------

Monitoring in MES is handled by ``Allen::Monitoring::AccumulatorManager``
(``main/include/AllenMonitoring.h``) together with a dedicated aggregation
thread in the scheduler.

GPU algorithms declare monitoring accumulators — ``Counter``,
``AveragingCounter``, ``Histogram``, ``Histogram2D`` and ``LogHistogram`` —
as members.  During a slice the kernels fill these directly in device memory.
The accumulator manager uses double buffering so the GPU can keep writing
while the aggregation thread copies the previous buffer back to the host.

Once per second the scheduler's monitoring thread calls
``AccumulatorManager::mergeAndReset()``, which merges the device buffers into
the host-side accumulators and resets the device side.  The host accumulators
are registered with the Gaudi monitoring hub, so they are published through
the normal Gaudi/online monitoring machinery (DIM, counters, histograms).

The same monitoring classes are also used in Gaudi mode to publish histograms
and counters per algorithm and per line; see
:ref:`monitoring_allen`.

TCK-from-ODIN and run changes
-----------------------------

When ``TCKFromODIN`` is enabled, the scheduler inspects the ODIN bank of each
incoming slice and compares the trigger configuration key (TCK) with the
currently loaded configuration.  On a change it waits for in-flight slices to
finish, reloads the configuration from the TCK repository, and reconfigures
the sequence algorithms.  This implements the fast run change used online
without restarting the process.

Relationship to Moore and MooreOnline
-------------------------------------

* **Offline / integration** (Moore): Allen runs as the HLT1 application
  inside Moore, steered through ``Allen.config.run_allen``.  This is the
  recommended way to run Allen on files and for HLT1 physics studies; see
  :ref:`run_allen_in_stack`.

* **Online / throughput** (MooreOnline): the same MES is configured by
  ``MooreOnline/AllenOnline/options/AllenConfig.py``, with an ``MEPProvider``
  reading from MBM and an ``Allen__MBMOutput`` writing back to MBM.  This is
  the only supported way to measure throughput; see
  :ref:`measuring_throughput`.
