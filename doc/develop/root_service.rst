.. _root_service:

ROOT Service
====================================================

The ``ROOTService`` utility lets an Allen algorithm write tuples to a shared
ROOT file. It owns the file and trees, serializes access from concurrent
processing threads, and writes the trees when Allen shuts down.

Use this service for general algorithms that need tuple output. Selection lines
have a higher-level monitoring interface described in
:doc:`selections`.

The implementation is in
`ROOTService.h <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/main/include/ROOTService.h>`_
and
`ROOTService.cpp <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/main/src/ROOTService.cpp>`_.

Adding tuple output to an algorithm
-----------------------------------

Declare a helper method that receives the algorithm arguments, runtime options,
and execution context. The algorithm must include ``ROOTService.h`` because the
service is accessed through ``RuntimeOptions``.

The following reduced example copies a device output to the host and stores one
value per entry:

.. code-block:: c++

  #include "ROOTService.h"

  void my_algorithm::my_algorithm_t::output_tuples(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions& runtime_options,
    const Allen::Context& context) const
  {
    auto handler = runtime_options.root_service->handle(name());
    auto* tree = handler.tree("monitor_tree");
    if (tree == nullptr) return;

    float score;
    handler.branch(tree, "score", score);

    const auto host_scores = make_host_buffer<dev_scores_t>(arguments, context);
    for (unsigned i = 0; i < host_scores.size(); ++i) {
      score = host_scores[i];
      tree->Fill();
    }
  }

``handle(name())`` creates or selects a directory named after the algorithm
instance. ``tree("monitor_tree")`` creates the tree on first use and returns
the existing tree on later calls. ``branch`` similarly creates a branch once
and reconnects it on subsequent calls.

The handler holds the service lock for its lifetime. Keep its scope limited to
the ROOT operations and do not retain the returned tree outside that scope.
The service owns the file and tree objects, so algorithms must not close or
delete them.

Copying device data
-------------------

ROOT runs on the host. Device results therefore need to be copied before they
are used to fill a tree. ``make_host_buffer`` performs the required transfer
using the supplied context:

.. code-block:: c++

  const auto host_values = make_host_buffer<dev_values_t>(arguments, context);

When several buffers are needed, create each host buffer before filling the
tree. Avoid writing ROOT objects from a device kernel.

Enabling the output
-------------------

Tuple output is normally guarded by an algorithm property so production
sequences do not pay the transfer and I/O cost unless it is requested:

.. code-block:: c++

  Allen::Property<bool> m_enable_tupling {
    this,
    "enable_tupling",
    false,
    "Enable tuple output"};

Call the helper after the device work has been scheduled:

.. code-block:: c++

  if (m_enable_tupling.value()) {
    output_tuples(arguments, runtime_options, context);
  }

Set the property when the algorithm is configured in Python. The exact builder
function depends on the algorithm and sequence.

Choosing the output file
------------------------

In standalone Allen, pass the output path with
``--monitoring-filename``:

.. code-block:: sh

  ./Allen \
    --sequence my_sequence \
    --monitoring-filename monitoring.root \
    --mdf /path/to/input.mdf

The file contains one directory per handler name and the trees requested under
that handler. If ROOT output is disabled, ``tree`` returns ``nullptr``; always
check the pointer before declaring branches or filling entries.

Examples in the repository
--------------------------

These algorithms show complete uses of the service:

* `FindMuonHits.cu <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/device/muon/match_velo_muon/src/FindMuonHits.cu>`_
  copies several device buffers and writes per-track muon information.
* `VeloKalmanFilter.cu <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/device/velo/simplified_kalman_filter/src/VeloKalmanFilter.cu>`_
  writes fitted VELO track-state values.
* `ReconstrucibleSignalCounter.cpp <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/host/validators/src/ReconstrucibleSignalCounter.cpp>`_
  shows the same service from a host algorithm.

For histograms and the monitoring infrastructure used in production, see
:doc:`../monitoring/monitoring_allen`.
