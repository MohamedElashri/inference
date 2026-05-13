AllenCuDNN
==========

``AllenCuDNN`` is Allen's internal cuDNN utility layer for CUDA inference
algorithms. It is intentionally small: the backend owns reusable cuDNN
descriptor, workspace, algorithm-selection, and device-weight mechanics, while
client algorithms keep their own model topology, parsing, validation, and
Allen buffer declarations.

Build Matrix
------------

``AllenCuDNN`` is only built for CUDA configurations with cuDNN enabled:

.. code-block:: sh

   cmake -S Allen -B build-cuda-cudnn \
     -DSTANDALONE=ON \
     -DTARGET_DEVICE=CUDA \
     -DWITH_CUDNN=ON
   cmake --build build-cuda-cudnn --target AllenCuDNN

For ordinary CPU, HIP, or CUDA builds without cuDNN, leave ``WITH_CUDNN`` unset
or set it to ``OFF``. The ``AllenCuDNN`` target is then not created, and
algorithms must keep cuDNN-only code behind the same build guards used by
PVFinder:

.. code-block:: cmake

   if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
     target_link_libraries(MyAlgorithm PRIVATE AllenCuDNN)
     target_compile_definitions(MyAlgorithm PRIVATE ALLEN_CUDNN_BACKEND_CUDA)
   endif()

HIP currently has only a named MIOpen stub in ``CuDNNBackendShim.h``. There is
no supported MIOpen implementation and no compatibility promise for HIP-backed
neural-network inference through this layer.

Public Include
--------------

New clients should include the umbrella header and link the ``AllenCuDNN``
target:

.. code-block:: c++

   #include "AllenCuDNN.h"

The umbrella header exposes:

* ``ForwardConvPlan`` for forward convolution.
* ``BackwardDataConvPlan`` for backward-data convolution and transposed
  convolution.
* ``BiasAddPlan`` and ``ActivationPlan`` for generic tensor bias and activation
  operations.
* ``TensorShape``, ``Conv2DShape``, ``Conv1DShape``, and ``TensorLayout`` for
  explicit shape and layout descriptions. ``TensorLayout::NCHW`` is the only
  supported layout in the current implementation.
* ``PrecisionPolicy`` plus ``fp32_precision_policy`` and
  ``fp16_precision_policy`` for dtype, compute type, math mode, Tensor Op, and
  TF32 choices. FP16 remains opt-in through ``fp16_experimental``.
* ``Workspace`` for typed externally owned workspace.
* ``DeviceWeights`` for namespaced process-lifetime or externally owned device
  weights.

Compatibility overloads that accept raw ``std::array`` shapes still exist for
older call sites. New code should prefer ``Conv2DShape`` or ``Conv1DShape`` so
shape validation and diagnostics stay explicit.

Algorithm Selection And Workspace
---------------------------------

Convolution plans are created with ``ConvPlanOptions``:

.. code-block:: c++

   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
   options.workspace_limit_bytes = 64ul * 1024 * 1024;

Supported algorithm policies are ``ZeroWorkspace``, ``Heuristic``, and
``TimedFind``. Supported workspace policies are ``ZeroOnly``, ``OwnedInitTime``,
and ``AllenExternal``. ``ZeroOnly`` must be paired with ``ZeroWorkspace``.

Plans expose metadata for benchmark and debugging output:

.. code-block:: c++

   const auto md = plan.metadata();
   std::printf(
     "algorithm=%s workspace=%zu source=%s fallback=%s\n",
     md.algorithm_name.c_str(),
     md.workspace_bytes,
     Allen::CuDNN::to_string(md.selection_source),
     md.fallback_reason.c_str());

The numeric algorithm ID is a cuDNN backend value and is not stable across
cuDNN versions, GPU models, or descriptor shapes. Use ``algorithm_name`` and
the full metadata when comparing logs.

Verbose plan creation logging can be enabled either per plan or through the
environment:

.. code-block:: c++

   options.log_plan_creation = true;

.. code-block:: sh

   ALLEN_CUDNN_VERBOSE=1 ./Allen ...

The ``fallback_reason`` field is empty when the requested selector succeeds. If
cuDNN returns no successful algorithm inside the workspace limit, or an API call
fails during selection, the plan records ``AlgorithmSelectionSource::Fallback``
and stores the reason there.

External Workspace
------------------

New clients that use Allen-managed scratch should use the typed ``Workspace``
overload rather than the raw ``void*`` overload:

.. code-block:: c++

   Allen::CuDNN::Workspace workspace {dev_scratch, scratch_bytes};
   plan.forward(handle, 1.0f, dev_input, dev_filter, 0.0f, dev_output, workspace);

The typed overload checks that the supplied buffer is large enough for
``plan.workspace_bytes()`` before calling cuDNN. No convolution plan allocates
workspace during event processing; ``OwnedInitTime`` allocates at plan creation,
and ``AllenExternal`` requires the caller to provide the buffer at execution.

Tiny Forward-Convolution Client
-------------------------------

.. code-block:: c++

   auto handle = Allen::CuDNN::HandleProvider::get(stream);

   Allen::CuDNN::ForwardConvPlan conv;
   Allen::CuDNN::ConvPlanOptions options {};
   options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
   options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

   conv.create(
     handle,
     Allen::CuDNN::Conv1DShape::forward(
       events, input_channels, width, output_channels, kernel_width, padding),
     options);

   conv.forward(handle, 1.0f, dev_input, dev_filter, 0.0f, dev_output);

Tiny Transposed-Convolution Client
----------------------------------

.. code-block:: c++

   auto handle = Allen::CuDNN::HandleProvider::get(stream);

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

   deconv.backward_data(handle, 1.0f, dev_filter, dev_input, 0.0f, dev_output);

Minimal DeviceWeights Client
----------------------------

Use one namespace per model or algorithm. Keys are stored as
``namespace.local_key`` in diagnostics.

.. code-block:: c++

   Allen::CuDNN::DeviceWeights weights {"my_model"};

   weights.load_from_buffer(
     "conv1.weight",
     host_bytes,
     host_bytes_size,
     expected_bytes,
     Allen::CuDNN::DuplicateKeyPolicy::Reject);

   const float* dev_weight = weights.get<float>("conv1.weight");

When a caller already owns a device allocation, register it as external:

.. code-block:: c++

   weights.register_device_pointer("conv1.external_weight", dev_weight, expected_bytes, expected_bytes);

Testing
-------

The generic cuDNN tests live in ``Allen/test/unit_tests/generic/src`` and are
compiled into the normal ``unit_tests`` target when testing is enabled.
Configure a test build explicitly with ``BUILD_TESTING=ON``:

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
