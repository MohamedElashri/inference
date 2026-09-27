.. _allen_cudnn:

CNNs with cuDNN (AllenCuDNN)
============================
``AllenCuDNN`` (``device/cudnn_backend``) lets any Allen algorithm run a
convolutional network with cuDNN's graph API. It provides per-stream cuDNN
handles, graphs and plans, and CNN layers built from them. It is not a model
framework: the algorithm owns its network's structure, weights (see
:doc:`add_mva_model`), buffers and any kernels of its own.

It is built with ``-DWITH_CUDNN=ON`` for ``TARGET_DEVICE=CUDA``, which also
defines ``ALLEN_WITH_CUDNN`` everywhere. An algorithm links it in its
``CMakeLists.txt``::

   if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
     target_link_libraries(MyAlgorithm PRIVATE AllenCuDNN)
   endif()

and includes ``AllenCuDNN.h`` inside ``#ifdef ALLEN_WITH_CUDNN``, refusing to
run (``StrException`` in ``init()``) when it is not defined.

Handles
^^^^^^^
``Allen::CuDNN::handle(context)`` returns the cuDNN handle of the algorithm's
stream, bound to it: one per stream, shared by all algorithms, created on
first use. In ``init()`` (no stream yet) use ``Allen::CuDNN::handle(nullptr)``.

Graphs and plans
^^^^^^^^^^^^^^^^
A ``Graph`` describes a computation: tensors that are bound to memory when it
runs (inputs, and the results declared with ``output()``) and operations whose
results are virtual unless declared outputs::

   Allen::CuDNN::Graph g(DataType::BFloat16, DataType::Float);   // storage, arithmetic
   auto x = g.input({N, C, 1, W}, Layout::NHWC);
   auto w = g.input({K, C, 1, R}, Layout::NHWC);
   auto b = g.input({1, K, 1, 1}, Layout::NHWC);
   g.output(g.relu(g.add(g.convolution(x, w, {{0, R / 2}, {1, 1}, {1, 1}}), b)));
   Plan plan = g.build(Allen::CuDNN::handle(nullptr));             // init()
   plan.execute(Allen::CuDNN::handle(context), {x_ptr, w_ptr, b_ptr, y_ptr}, workspace);  // operator()

Operations: ``convolution``, ``transposed_convolution``, ``add`` and ``mul``
(the second operand broadcasts), ``scale``, ``relu``, ``leaky_relu``,
``sigmoid``, ``tanh``, ``softplus``, ``pooling`` and ``matmul``. Tensors are
``[N][C][H][W]`` (1D: ``H = 1``), channels first (``NCHW``) or last
(``NHWC``); other ranks take explicit strides.

``build()`` asks cuDNN's heuristics for engines that run the whole graph, keeps
those the ``BuildOptions`` allow and returns a ``Plan``:

* ``allow_reduced_precision`` (off): TF32 tensor cores for float32,
  down-converted inputs or reduced-precision reductions. Off, float32 graphs
  are exact float32.
* ``allow_nondeterministic`` (off): engines whose results change from run to
  run.
* ``allow_runtime_compilation`` (on): engines compiled when the plan is built.
* ``max_workspace``, and ``max_candidates``: with more than one, the
  candidates are timed on scratch memory and the fastest is kept.

Building can compile a kernel (about a second), so build in ``init()``. Plans
are cached for the process by the graph's signature, so identical layers and
algorithms share them. ``build()`` throws a ``StrException`` naming the graph
when no allowed engine runs it; ``try_build()`` returns an invalid plan
instead. ``execute()`` takes the device pointers of the bound tensors in the
order they were declared. Its workspace must hold ``plan.workspace_size()``
bytes: request it as a ``DEVICE_OUTPUT`` argument sized in
``set_arguments_size``, so that it comes from Allen's memory manager.

What cuDNN runs is up to its version and the GPU. With cuDNN 9.6 on an
RTX 3090 (compute capability 8.6): exact float32 convolutions, transposed
convolutions, max pooling (channels first) and matrix products run as
single-operation graphs, but no float32 fusion does without TF32; bfloat16
convolution + bias + activation fuses in channels last (runtime-compiled
tensor-core engines); graphs of pointwise operations alone have no engine.

Layers
^^^^^^
``ConvolutionLayer`` is a 1D or 2D convolution, or transposed convolution,
with an optional per-channel bias, activation and output scale, for a fixed
batch shape. ``create()`` builds one fused graph when an allowed engine runs
it; otherwise it builds the convolution alone and ``forward()`` applies bias,
activation and scale with one element-wise kernel, in place. ``describe()``
says which, for the log::

   Allen::CuDNN::ConvolutionLayer layer;
   layer.create(handle, {.batch = N, .in_channels = C, .out_channels = K, .input_size = {W},
                         .kernel_size = {5}, .padding = {2}, .bias = true,
                         .activation = Activation::Relu});
   layer.forward(handle, x, w, b, y, workspace);

Weights are ``[K][C][R][S]`` for a convolution and ``[C][K][R][S]`` for a
transposed one (PyTorch's ``Conv`` and ``ConvTranspose`` layouts), in the
layer's layout. ``PoolingLayer`` does max or average pooling.

Example
^^^^^^^
``pvfinder_unet`` (``device/pvfinder``) runs its float32 UNet with these
layers: five convolution + bias + ReLU layers, two max-pools, two transposed
convolutions and an output convolution with softplus and scale, all built in
``init()`` with one shared workspace argument.

Tests
^^^^^
``test/unit_tests/generic/src/TestCuDNNGraph.cu`` checks every operation and
the layers against double-precision references on the host
(``-DBUILD_TESTING=ON``, then ``unit_tests "[AllenCuDNN]"``).
