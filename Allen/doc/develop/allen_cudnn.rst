AllenCuDNN
==========

``AllenCuDNN`` is Allen's internal cuDNN utility layer for CUDA inference
algorithms. It provides reusable handle access, descriptor plans, workspace
contracts, algorithm selection, precision policy, and device-weight storage.
It is not a model framework: each Allen algorithm still owns its model
topology, input and output buffers, model parsing, validation, and any
client-specific CUDA kernels.

The core rule is that cuDNN details stay inside the CUDA implementation of the
client algorithm. Allen parser-facing ``Parameters`` and properties should use
ordinary Allen types such as booleans, integers, strings, and Allen buffers.
Do not expose cuDNN descriptors, handles, algorithms, or backend enums through
the algorithm interface.

Build Boundary
--------------

``AllenCuDNN`` is built only for CUDA configurations with cuDNN enabled:

.. code-block:: sh

   cmake -S Allen -B build-cuda-cudnn \
     -DSTANDALONE=ON \
     -DTARGET_DEVICE=CUDA \
     -DWITH_CUDNN=ON
   cmake --build build-cuda-cudnn --target AllenCuDNN

For CPU, HIP, or CUDA builds without cuDNN, leave ``WITH_CUDNN`` unset or set
it to ``OFF``. The ``AllenCuDNN`` target is then not created.

Client algorithms should link the backend only inside the CUDA/cuDNN build
guard:

.. code-block:: cmake

   if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
     target_link_libraries(MyAlgorithm PRIVATE AllenCuDNN)
     target_compile_definitions(MyAlgorithm PRIVATE ALLEN_CUDNN_BACKEND_CUDA)
   endif()

The HIP shim currently only provides a named placeholder. There is no supported
MIOpen implementation behind this layer.

Public Include
--------------

New clients should include the umbrella header:

.. code-block:: c++

   #include "AllenCuDNN.h"

The public API includes:

* ``HandleProvider`` and ``get_thread_local_handle`` for per-thread cuDNN
  handles bound to the current CUDA stream.
* ``ForwardConvPlan`` for forward convolution.
* ``BackwardDataConvPlan`` for backward-data convolution and transposed
  convolution.
* ``BiasAddPlan`` and ``ActivationPlan`` for generic tensor bias and activation
  operations.
* ``PoolingPlan`` for generic NCHW pooling operations.
* ``FusedConvPlan`` for forward convolution with optional channel-bias and
  activation post-ops.
* ``TensorShape``, ``Conv2DShape``, ``Conv1DShape``, ``Pooling2DShape``,
  ``Pooling1DShape``, and ``TensorLayout`` for explicit shape and layout
  descriptions. ``TensorLayout::NCHW`` is currently the supported layout.
* ``ConvPlanOptions`` and ``PrecisionPolicy`` for algorithm, workspace,
  precision, math-mode, Tensor Op, TF32, and cache choices.
* ``Workspace``, ``WorkspacePlanner``, ``WorkspacePlan``, and
  ``WorkspaceArena`` for typed external workspace.
* ``DeviceWeights`` for namespaced process-lifetime or externally owned device
  weights.

Compatibility overloads that accept raw ``std::array`` shapes still exist for
older call sites. New clients should use ``Conv2DShape`` or ``Conv1DShape`` so
validation and diagnostics remain explicit.

Adding A Client Algorithm
-------------------------

Use this lifecycle for a new Allen algorithm that wants cuDNN-backed inference:

1. Keep model parsing and topology in the client.
2. Add ordinary Allen properties for feature flags, model paths, dimensions,
   precision mode names, or workspace choices.
3. Declare all event-time scratch and output buffers in the client algorithm.
4. Load model weights into a namespaced ``DeviceWeights`` instance during
   initialization or a process-lifetime first-load path.
5. Build cuDNN plans once, after a live ``cudnnHandle_t`` is available. Use
   ``std::call_once`` or an equivalent process-lifetime guard if the plans are
   shared across algorithm instances.
6. In ``operator()``, get the thread-local handle for the event stream and call
   the plans with client-owned device pointers.
7. Record plan metadata in validation or benchmark logs.

Minimal execution pattern:

.. code-block:: c++

   auto handle = Allen::CuDNN::HandleProvider::get(context.stream());

   Allen::CuDNN::ForwardConvPlan conv;
   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

   conv.create(
     handle,
     Allen::CuDNN::Conv1DShape::forward(
       events, input_channels, width, output_channels, kernel_width, padding),
     options);

   conv.forward(handle, 1.0f, 0.0f, dev_input, dev_filter, dev_output);

Plans should be created outside event processing. Event-time code should only
bind the handle to the current stream, pass pointers, and launch work.

Handle Lifetime
---------------

Use ``HandleProvider`` for new code:

.. code-block:: c++

   auto handle = Allen::CuDNN::HandleProvider::get(context.stream());

The provider creates one cuDNN handle per OS thread on first use, reuses it, and
calls ``cudnnSetStream`` on every request. Do not store mutable cuDNN handles in
each algorithm instance unless you are maintaining legacy code.

Weights
-------

Use one ``DeviceWeights`` namespace per model or algorithm. The namespace is
prepended to keys in diagnostics and prevents unrelated clients from colliding:

.. code-block:: c++

   Allen::CuDNN::DeviceWeights weights {"my_algorithm"};

   weights.load_from_buffer(
     "conv1.weight",
     host_bytes,
     host_bytes_size,
     expected_bytes,
     Allen::CuDNN::DuplicateKeyPolicy::Reject);

   const float* dev_weight = weights.get<float>("conv1.weight");

When another component already owns a device allocation, register it as an
external pointer:

.. code-block:: c++

   weights.register_device_pointer(
     "conv1.external_weight",
     dev_weight,
     expected_bytes,
     expected_bytes,
     Allen::CuDNN::DuplicateKeyPolicy::Reject);

``DeviceWeights`` supports process-lifetime device allocations and caller-owned
external device pointers. It does not define a common model-file format or an
Allen constants reload policy; those remain client responsibilities.

``WeightRegistry`` is a legacy facade over ``DeviceWeights`` for old migration
call sites. New code should use ``DeviceWeights`` directly.

Shapes And Layout
-----------------

Describe tensor geometry through ``Conv1DShape`` or ``Conv2DShape``:

.. code-block:: c++

   auto shape = Allen::CuDNN::Conv2DShape::forward(
     Allen::CuDNN::TensorShape {batch, input_channels, height, width},
     Allen::CuDNN::TensorShape {output_channels, input_channels, kernel_h, kernel_w},
     {pad_h, pad_w},
     {stride_h, stride_w},
     {dilation_h, dilation_w});

The wrapper validates positive dimensions, pad/stride/dilation values, channel
compatibility, supported layout, and caller-supplied output shapes before
calling cuDNN where possible. The current implementation supports NCHW
descriptors.

Forward Convolution
-------------------

.. code-block:: c++

   Allen::CuDNN::ForwardConvPlan conv;
   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::TimedFind;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
   options.workspace_limit_bytes = 64ul * 1024 * 1024;

   conv.create(handle, shape, options);
   conv.forward(handle, 1.0f, 0.0f, dev_input, dev_filter, dev_output);

``ForwardConvPlan`` owns tensor, filter, and convolution descriptors. Depending
on the selected workspace policy, it either owns its workspace from creation
time or requires the caller to supply external workspace at execution.

Transposed Convolution
----------------------

Use ``BackwardDataConvPlan`` for cuDNN backward-data convolution, including
transposed-convolution style layers:

.. code-block:: c++

   Allen::CuDNN::BackwardDataConvPlan deconv;
   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;

   deconv.create(
     handle,
     Allen::CuDNN::Conv1DShape::backward_data(
       events,
       input_channels,
       input_width,
       output_channels,
       output_width,
       kernel_width,
       padding,
       stride),
     options);

   deconv.backward_data(handle, 1.0f, 0.0f, dev_filter, dev_input, dev_output);

Algorithm Selection
-------------------

Convolution plans use ``ConvPlanOptions``:

.. code-block:: c++

   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::TimedFind;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
   options.workspace_limit_bytes = 64ul * 1024 * 1024;

Supported algorithm policies are:

* ``ZeroWorkspace``: select the zero-workspace default.
* ``Heuristic``: ask cuDNN for algorithm candidates and choose one within the
  workspace limit.
* ``TimedFind``: run cuDNN timing-based selection and choose one within the
  workspace limit.

Plans expose metadata for diagnostics:

.. code-block:: c++

   const auto md = conv.metadata();
   std::printf(
     "algorithm=%s workspace=%zu source=%s cache=%s fallback=%s\n",
     md.algorithm_name.c_str(),
     md.workspace_bytes,
     Allen::CuDNN::to_string(md.selection_source),
     Allen::CuDNN::to_string(md.cache.status),
     md.fallback_reason.c_str());

The numeric algorithm ID is a cuDNN backend value and is not stable across
cuDNN versions, GPU models, or descriptor shapes. Prefer ``algorithm_name`` and
the full metadata when comparing logs.

Verbose plan creation logging can be enabled per plan or through the
environment:

.. code-block:: c++

   options.log_plan_creation = true;

.. code-block:: sh

   ALLEN_CUDNN_VERBOSE=1 ./Allen ...

Algorithm Cache
---------------

``ForwardConvPlan`` and ``BackwardDataConvPlan`` can use a process-local
in-memory algorithm-selection cache:

.. code-block:: c++

   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::TimedFind;
   options.cache_policy = Allen::CuDNN::AlgorithmCachePolicy::LookupAndPopulate;

Cache policies are ``Disabled``, ``LookupOnly``, ``Populate``,
``LookupAndPopulate``, and ``StrictLookup``. Strict lookup throws when an exact
descriptor/environment key is absent. Cache keys include operation type, shape,
layout, precision, math policy, workspace policy and limit, GPU model, CUDA
runtime version, and cuDNN version. Fallback selections are not cached unless
``cache_fallback_results`` is set explicitly.

Workspace
---------

Choose the workspace policy deliberately:

* ``ZeroOnly`` requires a zero-workspace algorithm and must be paired with
  ``AlgorithmSelectionPolicy::ZeroWorkspace``.
* ``OwnedInitTime`` allocates any needed workspace when the plan is created.
* ``AllenExternal`` requires the caller to pass a ``Workspace`` at execution.

Use ``AllenExternal`` when the client wants Allen-managed scratch instead of
process-lifetime private allocations:

.. code-block:: c++

   Allen::CuDNN::Workspace workspace {dev_scratch, scratch_bytes};
   conv.forward(
     handle,
     1.0f,
     0.0f,
     dev_input,
     dev_filter,
     dev_output,
     workspace);

The typed overload checks that the supplied buffer is large enough for
``conv.workspace_bytes()`` before calling cuDNN.

For several plans that execute sequentially, use ``WorkspacePlanner`` to map all
requirements onto one Allen buffer:

.. code-block:: c++

   Allen::CuDNN::WorkspacePlanner planner;
   planner.add("conv_a", conv_a.workspace_bytes());
   planner.add("conv_b", conv_b.workspace_bytes());

   const auto plan = planner.non_overlapping_plan();

   Allen::CuDNN::WorkspaceArena arena {{dev_workspace, workspace_bytes}, plan};
   conv_a.forward(handle, 1.0f, 0.0f, in_a, w_a, out_a, arena.slice("conv_a"));
   conv_b.forward(handle, 1.0f, 0.0f, in_b, w_b, out_b, arena.slice("conv_b"));

``non_overlapping_plan`` reuses offset zero for sequential work and sizes the
buffer to the maximum requirement. ``overlapping_plan`` assigns separate aligned
slices for work that may run concurrently.

Fused Convolution
-----------------

``FusedConvPlan`` describes forward convolution plus an optional generic
post-op sequence. Supported post-op sequences are:

* convolution only
* convolution plus channel bias
* convolution plus activation
* convolution plus channel bias plus activation

Residual/add post-ops are rejected until a concrete client needs that contract.
Client-specific operations, such as custom normalization, concatenation, or
output transforms, should stay in the client unless they become useful to more
than one algorithm.

Example:

.. code-block:: c++

   Allen::CuDNN::FusedConvPlanOptions options {};
   options.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::TimedFind;
   options.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
   options.backend_preference =
     Allen::CuDNN::FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp;
   options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(
     Allen::CuDNN::ActivationMode::Relu);

   Allen::CuDNN::FusedConvPlan plan;
   plan.create(handle, shape, options);

   plan.execute(handle, 1.0f, 0.0f, dev_input, dev_filter, dev_bias, dev_output);

The executable backend is intentionally narrow: FP32, NCHW, and identity/ReLU
activation for the CUDA post-op path. ``FusedConvBackendPreference`` can prefer
or force the legacy cuDNN-convolution-plus-CUDA-post-op backend or the cuDNN
frontend graph probe. If a forced backend is unsupported, creation fails with a
diagnostic reason. If fallback is allowed, metadata records the selected backend
and ``fallback_reason``.

The no-handle ``create`` overload is useful for metadata-only tests and
compile-time client-contract checks:

.. code-block:: c++

   Allen::CuDNN::FusedConvPlan metadata_only;
   metadata_only.create(shape, options);

Pooling
-------

``PoolingPlan`` describes generic NCHW pooling. It was added as the first V2.10
operation extension after the V2.9 cutover made PVFinder an accepted direct
``AllenCuDNN`` client. The first accepted test representative is max pooling
with typed ``Pooling2DShape``/``Pooling1DShape`` shapes and explicit precision
metadata:

.. code-block:: c++

   Allen::CuDNN::PoolingOptions options {};
   options.mode = Allen::CuDNN::PoolingMode::Max;

   Allen::CuDNN::PoolingPlan pool;
   pool.create(Allen::CuDNN::Pooling1DShape::forward(N, C, W, 2, 2), options);
   pool.forward(handle, 1.0f, dev_input, 0.0f, dev_output);

Pooling plans do not use cuDNN workspace. Creation validates the requested
layout, shape, precision policy, and computed output shape.
``ALLEN_CUDNN_VERBOSE=1`` logs mode, input/output shape, window, padding,
stride, workspace bytes, and precision policy.

PVFinder's FP32 maxpool hot path remains on its local kernel for now. The V2.10
trial of replacing it with ``PoolingPlan`` missed the existing FP32 PyTorch
validation tolerance, so the generic plan is available and tested but not yet an
accepted PVFinder default-path operation.

Precision
---------

Precision is a plan contract, not a validation claim. FP32 is the default:

.. code-block:: c++

   options.precision = Allen::CuDNN::fp32_precision_policy();

Non-FP32 modes must be explicit:

.. code-block:: c++

   options.precision = Allen::CuDNN::fp16_precision_policy(true);

An FP16 plan requires ``fp16_experimental=true``. A client should keep any
non-FP32 mode behind an opt-in property until it has documented:

* dtype mode, compute type, math mode, Tensor Op, and TF32 policy
* reference output source and numerical tolerances
* benchmark command and throughput gate
* supported GPU architecture class
* known physics or numerical caveats

Metadata And Validation Logs
----------------------------

For validation and benchmark runs, archive enough metadata to explain the
runtime path:

* client algorithm and model/version identifier
* input, filter, and output shapes
* precision policy
* selected algorithm name and selection source
* workspace policy, workspace limit, and selected workspace bytes
* cache policy and cache status
* fused backend preference and selected backend, when using ``FusedConvPlan``
* fallback reason, when non-empty

This makes benchmark comparisons reproducible and prevents a silent fallback
from being mistaken for an accepted performance result.

PVFinder As An Example
----------------------

PVFinder's UNet is an example client of ``AllenCuDNN``. It demonstrates several
recommended patterns:

* weights are loaded through a namespaced ``DeviceWeights`` instance
* cuDNN handles are retrieved through the thread-local provider
* FP32 CBR layers use ``FusedConvPlan`` directly for convolution plus channel
  bias and ReLU
* non-CBR forward and backward-data convolution plans are created once and
  reused
* the accepted Allen external workspace path uses a single reusable Allen
  buffer for sequential cuDNN plans by default
* FP16 remains an opt-in experimental mode with client-owned conversion kernels
  and separate validation records

PVFinder-specific pieces are not part of the generic backend contract. Its model
file format, interval feature construction, batch-normalization folding,
maxpooling, concatenation, softplus/output kernels, FP16 experimental
conversion/pooling path, and PyTorch comparison path belong to PVFinder. A new
Allen algorithm should copy the integration pattern, not the PVFinder topology.

Client Checklist
----------------

Before merging a new cuDNN-backed client, check that:

* cuDNN-only code is guarded by ``WITH_CUDNN`` and CUDA build logic.
* The client links ``AllenCuDNN`` only in CUDA/cuDNN builds.
* Parser-facing ``Parameters`` contain no cuDNN types.
* Weight keys are namespaced through ``DeviceWeights``.
* Plans are created during initialization or another guarded first-use path,
  not per event.
* Shapes use ``Conv1DShape`` or ``Conv2DShape``.
* Algorithm policy, workspace policy, workspace limit, precision policy, and
  cache policy are set intentionally.
* Event-time execution does not allocate hidden cuDNN workspace.
* Plan metadata is available in benchmark or validation output.
* Non-FP32 precision and unaccepted fused-post-op paths are opt-in until
  accepted by the client.
* Unsupported backends fail clearly or record an explicit fallback reason.

Testing
-------

Generic cuDNN tests live under ``Allen/test/unit_tests/generic/src`` and are
compiled into the normal ``unit_tests`` target when testing is enabled:

.. code-block:: sh

   cmake -S Allen -B build-cuda-cudnn-tests \
     -DSTANDALONE=ON \
     -DTARGET_DEVICE=CUDA \
     -DWITH_CUDNN=ON \
     -DBUILD_TESTING=ON
   cmake --build build-cuda-cudnn-tests --target unit_tests
   ctest --test-dir build-cuda-cudnn-tests -R cudnn

For ``WITH_CUDNN=OFF`` builds, the same test source compiles a stub test that
does not include cuDNN descriptors. Runtime convolution tests require a working
CUDA driver and a visible CUDA-capable device.
