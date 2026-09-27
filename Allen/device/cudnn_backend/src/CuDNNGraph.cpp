/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "CuDNNGraph.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>

namespace Allen::CuDNN {

  namespace {
    using Handle = std::shared_ptr<std::remove_pointer_t<cudnnBackendDescriptor_t>>;

    Handle make(cudnnBackendDescriptorType_t type)
    {
      cudnnBackendDescriptor_t d = nullptr;
      ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(type, &d));
      return Handle(d, [](cudnnBackendDescriptor_t p) { cudnnBackendDestroyDescriptor(p); });
    }

    void
    set(const Handle& d, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t n, const void* v)
    {
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d.get(), name, type, n, v));
    }

    void set_desc(const Handle& d, cudnnBackendAttributeName_t name, const Handle& value)
    {
      cudnnBackendDescriptor_t raw = value.get();
      set(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw);
    }

    void finalize(const Handle& d) { ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d.get())); }

    void cuda_check(cudaError_t status, const char* what)
    {
      if (status != cudaSuccess) {
        throw StrException(std::string("Allen::CuDNN: ") + what + ": " + cudaGetErrorString(status));
      }
    }

    cudnnDataType_t cudnn_type(DataType t)
    {
      switch (t) {
      case DataType::Half: return CUDNN_DATA_HALF;
      case DataType::BFloat16: return CUDNN_DATA_BFLOAT16;
      default: return CUDNN_DATA_FLOAT;
      }
    }

    size_t element_size(DataType t) { return t == DataType::Float ? 4 : 2; }

    std::vector<int64_t> packed_strides(const std::vector<int64_t>& dims, bool channels_last)
    {
      std::vector<int64_t> strides(dims.size(), 1);
      if (channels_last && dims.size() >= 3) {
        // [N][C][spatial...] stored as [N][spatial...][C]
        int64_t s = dims[1];
        for (size_t i = dims.size() - 1; i >= 2; --i) {
          strides[i] = s;
          s *= dims[i];
        }
        strides[1] = 1;
        strides[0] = s;
        return strides;
      }
      for (size_t i = dims.size() - 1; i-- > 0;)
        strides[i] = strides[i + 1] * dims[i + 1];
      return strides;
    }

    std::string join(const std::vector<int64_t>& v)
    {
      std::ostringstream s;
      for (size_t i = 0; i < v.size(); ++i)
        s << (i ? "," : "") << v[i];
      return s.str();
    }

    std::mutex cache_mutex;
    std::map<std::string, std::shared_ptr<const Plan::Impl>> plan_cache;
  } // namespace

  // ---------------------------------------------------------------------------
  // Graph description
  // ---------------------------------------------------------------------------
  struct Graph::Impl {
    struct Tensor {
      std::vector<int64_t> dims, strides;
      DataType type = DataType::Float;
      bool is_virtual = true;
      bool by_value = false;
      float value = 0.f;
    };
    enum class Kind { Convolution, TransposedConvolution, Pointwise, Pooling, Matmul };
    struct Op {
      Kind kind;
      std::vector<int64_t> in; // uids
      int64_t out;
      ConvolutionParams conv;
      PoolingParams pool;
      cudnnPointwiseMode_t pointwise = CUDNN_POINTWISE_ADD;
      float pointwise_param = 0.f;
    };
    DataType io, compute;
    std::vector<Tensor> tensors; // uid = index + 1
    std::vector<Op> ops;
    std::vector<int64_t> bound;

    Tensor& at(TensorId t)
    {
      if (t.uid < 1 || t.uid > static_cast<int64_t>(tensors.size())) {
        throw StrException("Allen::CuDNN::Graph: unknown tensor");
      }
      return tensors[t.uid - 1];
    }
    TensorId add_tensor(Tensor t)
    {
      tensors.push_back(std::move(t));
      return TensorId {static_cast<int64_t>(tensors.size())};
    }
    bool channels_last(TensorId t) { return at(t).strides.size() >= 3 && at(t).strides[1] == 1 && at(t).dims[1] > 1; }
    TensorId virtual_like(std::vector<int64_t> dims, bool cl)
    {
      Tensor t;
      t.strides = packed_strides(dims, cl);
      t.dims = std::move(dims);
      t.type = compute;
      return add_tensor(t);
    }
    TensorId pointwise(cudnnPointwiseMode_t mode, TensorId a, TensorId b = {}, float param = 0.f)
    {
      Tensor& ta = at(a);
      TensorId y = virtual_like(ta.dims, channels_last(a));
      at(y).strides = at(a).strides;
      Op op {Kind::Pointwise, {a.uid}, y.uid, {}, {}, mode, param};
      if (b.uid > 0) op.in.push_back(b.uid);
      ops.push_back(op);
      return y;
    }
  };

  Graph::Graph(DataType io, DataType compute) : m_impl(std::make_unique<Impl>())
  {
    m_impl->io = io;
    m_impl->compute = compute;
  }
  Graph::~Graph() = default;
  Graph::Graph(Graph&&) noexcept = default;
  Graph& Graph::operator=(Graph&&) noexcept = default;

  TensorId Graph::input(std::vector<int64_t> dims, Layout layout)
  {
    return input(dims, packed_strides(dims, layout == Layout::NHWC));
  }

  TensorId Graph::input(std::vector<int64_t> dims, std::vector<int64_t> strides)
  {
    Impl::Tensor t;
    t.strides = strides.empty() ? packed_strides(dims, false) : std::move(strides);
    t.dims = std::move(dims);
    t.type = m_impl->io;
    t.is_virtual = false;
    const TensorId id = m_impl->add_tensor(t);
    m_impl->bound.push_back(id.uid);
    return id;
  }

  void Graph::output(TensorId t)
  {
    auto& tensor = m_impl->at(t);
    if (!tensor.is_virtual) throw StrException("Allen::CuDNN::Graph: output must be an operation's result");
    tensor.is_virtual = false;
    tensor.type = m_impl->io;
    m_impl->bound.push_back(t.uid);
  }

  TensorId Graph::convolution(TensorId x, TensorId w, const ConvolutionParams& p)
  {
    const auto& xd = m_impl->at(x).dims;
    const auto& wd = m_impl->at(w).dims;
    std::vector<int64_t> yd {xd[0], wd[0]};
    for (size_t i = 2; i < xd.size(); ++i) {
      const size_t s = i - 2;
      yd.push_back((xd[i] + 2 * p.padding[s] - p.dilation[s] * (wd[i] - 1) - 1) / p.stride[s] + 1);
    }
    const TensorId y = m_impl->virtual_like(yd, m_impl->channels_last(x));
    m_impl->ops.push_back({Impl::Kind::Convolution, {x.uid, w.uid}, y.uid, p, {}, CUDNN_POINTWISE_ADD, 0.f});
    return y;
  }

  TensorId
  Graph::transposed_convolution(TensorId x, TensorId w, const ConvolutionParams& p, std::vector<int64_t> y_dims)
  {
    const TensorId y = m_impl->virtual_like(y_dims, m_impl->channels_last(x));
    m_impl->ops.push_back({Impl::Kind::TransposedConvolution, {x.uid, w.uid}, y.uid, p, {}, CUDNN_POINTWISE_ADD, 0.f});
    return y;
  }

  TensorId Graph::add(TensorId a, TensorId b) { return m_impl->pointwise(CUDNN_POINTWISE_ADD, a, b); }
  TensorId Graph::mul(TensorId a, TensorId b) { return m_impl->pointwise(CUDNN_POINTWISE_MUL, a, b); }
  TensorId Graph::scale(TensorId a, float factor)
  {
    Impl::Tensor s;
    s.dims = std::vector<int64_t>(m_impl->at(a).dims.size(), 1);
    s.strides = s.dims;
    s.type = DataType::Float;
    s.is_virtual = false;
    s.by_value = true;
    s.value = factor;
    return m_impl->pointwise(CUDNN_POINTWISE_MUL, a, m_impl->add_tensor(s));
  }
  TensorId Graph::relu(TensorId a) { return m_impl->pointwise(CUDNN_POINTWISE_RELU_FWD, a); }
  TensorId Graph::leaky_relu(TensorId a, float slope)
  {
    return m_impl->pointwise(CUDNN_POINTWISE_RELU_FWD, a, {}, slope);
  }
  TensorId Graph::sigmoid(TensorId a) { return m_impl->pointwise(CUDNN_POINTWISE_SIGMOID_FWD, a); }
  TensorId Graph::tanh(TensorId a) { return m_impl->pointwise(CUDNN_POINTWISE_TANH_FWD, a); }
  TensorId Graph::softplus(TensorId a) { return m_impl->pointwise(CUDNN_POINTWISE_SOFTPLUS_FWD, a, {}, 1.f); }

  TensorId Graph::pooling(TensorId x, const PoolingParams& p)
  {
    const auto& xd = m_impl->at(x).dims;
    std::vector<int64_t> yd {xd[0], xd[1]};
    for (size_t i = 2; i < xd.size(); ++i) {
      const size_t s = i - 2;
      yd.push_back((xd[i] + 2 * p.padding[s] - p.window[s]) / p.stride[s] + 1);
    }
    const TensorId y = m_impl->virtual_like(yd, m_impl->channels_last(x));
    m_impl->ops.push_back({Impl::Kind::Pooling, {x.uid}, y.uid, {}, p, CUDNN_POINTWISE_ADD, 0.f});
    return y;
  }

  TensorId Graph::matmul(TensorId a, TensorId b)
  {
    const auto& ad = m_impl->at(a).dims;
    const auto& bd = m_impl->at(b).dims;
    std::vector<int64_t> yd {ad[0], ad[1], bd[2]};
    const TensorId y = m_impl->virtual_like(yd, false);
    m_impl->ops.push_back({Impl::Kind::Matmul, {a.uid, b.uid}, y.uid, {}, {}, CUDNN_POINTWISE_ADD, 0.f});
    return y;
  }

  std::vector<TensorId> Graph::bound_tensors() const
  {
    std::vector<TensorId> r;
    for (const auto uid : m_impl->bound)
      r.push_back(TensorId {uid});
    return r;
  }

  const std::vector<int64_t>& Graph::dims(TensorId t) const { return m_impl->at(t).dims; }

  std::string Graph::signature() const
  {
    std::ostringstream s;
    s << "io" << static_cast<int>(m_impl->io) << ",compute" << static_cast<int>(m_impl->compute) << ";";
    for (size_t i = 0; i < m_impl->tensors.size(); ++i) {
      const auto& t = m_impl->tensors[i];
      s << "t" << i + 1 << "[" << join(t.dims) << "/" << join(t.strides) << "]" << static_cast<int>(t.type)
        << (t.is_virtual ? "v" : "") << (t.by_value ? "=" + std::to_string(t.value) : "") << ";";
    }
    for (const auto& op : m_impl->ops) {
      s << "op" << static_cast<int>(op.kind) << "(" << join(op.in) << "->" << op.out << ")";
      if (op.kind == Impl::Kind::Convolution || op.kind == Impl::Kind::TransposedConvolution) {
        s << "p" << join(op.conv.padding) << "s" << join(op.conv.stride) << "d" << join(op.conv.dilation);
      }
      if (op.kind == Impl::Kind::Pooling) {
        s << "m" << static_cast<int>(op.pool.mode) << "w" << join(op.pool.window) << "s" << join(op.pool.stride) << "p"
          << join(op.pool.padding);
      }
      if (op.kind == Impl::Kind::Pointwise) s << "pw" << op.pointwise << "/" << op.pointwise_param;
      s << ";";
    }
    s << "bound" << join(m_impl->bound);
    return s.str();
  }

  // ---------------------------------------------------------------------------
  // Plans
  // ---------------------------------------------------------------------------
  // A plan holds the engine configuration the build chose. cuDNN ties an
  // execution plan to the handle it is finalized with, so each handle gets
  // its own plan from that configuration (same engine, same numerics) on first
  // use, and keeps the variant pack of its last call, rebuilt only when the
  // pointers change. As for the handles themselves, one thread at a time uses
  // a given handle (Allen: one per stream).
  struct Plan::Impl {
    struct PerHandle {
      Handle plan;
      Handle pack;
      std::vector<void*> pointers; // of the pack: bound tensors, then scalars
      void* workspace = nullptr;
      std::vector<float> values; // the by-value scalars the pack points to
    };

    Handle config;
    size_t workspace = 0;
    std::string engine;
    std::vector<int64_t> bound;                     // uids, declaration order
    std::vector<std::pair<int64_t, float>> scalars; // by-value tensors

    mutable std::mutex mutex;
    mutable std::map<cudnnHandle_t, std::shared_ptr<PerHandle>> per_handle;

    PerHandle& for_handle(cudnnHandle_t handle) const;
  };

  namespace {
    Handle make_plan(cudnnHandle_t handle, const Handle& config, size_t& workspace);
  }

  Plan::Impl::PerHandle& Plan::Impl::for_handle(cudnnHandle_t handle) const
  {
    std::lock_guard<std::mutex> lock {mutex};
    auto& entry = per_handle[handle];
    if (!entry) {
      auto h = std::make_shared<PerHandle>();
      size_t bytes = 0;
      h->plan = make_plan(handle, config, bytes);
      if (!h->plan || bytes > workspace) {
        per_handle.erase(handle);
        throw StrException("Allen::CuDNN::Plan: cannot finalize " + engine + " for another cuDNN handle");
      }
      for (const auto& scalar : scalars)
        h->values.push_back(scalar.second);
      entry = std::move(h);
    }
    return *entry;
  }

  void Plan::execute(cudnnHandle_t handle, std::initializer_list<const void*> pointers, void* workspace) const
  {
    execute(handle, std::vector<const void*>(pointers), workspace);
  }

  void Plan::execute(cudnnHandle_t handle, const std::vector<const void*>& pointers, void* workspace) const
  {
    if (!m_impl) throw StrException("Allen::CuDNN::Plan: executing an empty plan");
    if (pointers.size() != m_impl->bound.size()) {
      throw StrException(
        "Allen::CuDNN::Plan: " + std::to_string(pointers.size()) + " pointers for " +
        std::to_string(m_impl->bound.size()) + " bound tensors");
    }
    Impl::PerHandle& h = m_impl->for_handle(handle);
    const bool same =
      h.pack && h.workspace == workspace && std::equal(pointers.begin(), pointers.end(), h.pointers.begin());
    if (!same) {
      std::vector<int64_t> uids = m_impl->bound;
      h.pointers.assign(pointers.size(), nullptr);
      for (size_t i = 0; i < pointers.size(); ++i)
        h.pointers[i] = const_cast<void*>(pointers[i]);
      for (size_t i = 0; i < m_impl->scalars.size(); ++i) {
        uids.push_back(m_impl->scalars[i].first);
        h.pointers.push_back(&h.values[i]);
      }
      auto pack = make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
      set(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, static_cast<int64_t>(uids.size()), uids.data());
      set(
        pack,
        CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        CUDNN_TYPE_VOID_PTR,
        static_cast<int64_t>(h.pointers.size()),
        h.pointers.data());
      set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
      finalize(pack);
      h.pack = std::move(pack);
      h.workspace = workspace;
    }
    ALLEN_CUDNN_CHECK(cudnnBackendExecute(handle, h.plan.get(), h.pack.get()));
  }

  size_t Plan::workspace_size() const { return m_impl ? m_impl->workspace : 0; }

  const std::string& Plan::engine() const
  {
    static const std::string none = "none";
    return m_impl ? m_impl->engine : none;
  }

  size_t plan_cache_size()
  {
    std::lock_guard<std::mutex> lock {cache_mutex};
    return plan_cache.size();
  }

  namespace {
    struct Built {
      std::vector<Handle> keep; // descriptors the operation graph refers to
      Handle graph;
    };

    Built build_operation_graph(const Graph::Impl& g, cudnnHandle_t handle)
    {
      Built b;
      std::vector<Handle> tensors;
      for (size_t i = 0; i < g.tensors.size(); ++i) {
        const auto& t = g.tensors[i];
        auto d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        const cudnnDataType_t type = cudnn_type(t.type);
        const int64_t uid = static_cast<int64_t>(i) + 1, alignment = 16;
        const bool is_virtual = t.is_virtual, by_value = t.by_value;
        set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
        set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
        set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, static_cast<int64_t>(t.dims.size()), t.dims.data());
        set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, static_cast<int64_t>(t.strides.size()), t.strides.data());
        set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
        set(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
        if (by_value) set(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value);
        finalize(d);
        tensors.push_back(d);
      }
      const auto tensor = [&](int64_t uid) -> const Handle& { return tensors[uid - 1]; };
      const cudnnDataType_t compute = cudnn_type(g.compute);
      const float alpha = 1.f, beta = 0.f;
      std::vector<Handle> ops;
      for (const auto& op : g.ops) {
        switch (op.kind) {
        case Graph::Impl::Kind::Convolution:
        case Graph::Impl::Kind::TransposedConvolution: {
          auto conv = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
          const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
          const int64_t dims = static_cast<int64_t>(op.conv.padding.size());
          set(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
          set(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
          set(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &dims);
          set(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, dims, op.conv.padding.data());
          set(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, dims, op.conv.padding.data());
          set(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, dims, op.conv.stride.data());
          set(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dims, op.conv.dilation.data());
          finalize(conv);
          b.keep.push_back(conv);
          if (op.kind == Graph::Impl::Kind::Convolution) {
            auto d = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensor(op.in[0]));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, tensor(op.in[1]));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, tensor(op.out));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv);
            set(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
            set(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
            finalize(d);
            ops.push_back(d);
          }
          else {
            // y = x (*)^T w: the backward-data convolution with dy = x and dx = y.
            auto d = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, tensor(op.in[0]));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, tensor(op.in[1]));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, tensor(op.out));
            set_desc(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, conv);
            set(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
            set(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
            finalize(d);
            ops.push_back(d);
          }
          break;
        }
        case Graph::Impl::Kind::Pointwise: {
          auto pw = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
          const cudnnPointwiseMode_t mode = op.pointwise;
          set(pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
          set(pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute);
          if (mode == CUDNN_POINTWISE_RELU_FWD && op.pointwise_param != 0.f) {
            const double slope = op.pointwise_param;
            set(pw, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, CUDNN_TYPE_DOUBLE, 1, &slope);
          }
          if (mode == CUDNN_POINTWISE_SOFTPLUS_FWD) {
            const double beta_sp = op.pointwise_param;
            set(pw, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, CUDNN_TYPE_DOUBLE, 1, &beta_sp);
          }
          finalize(pw);
          b.keep.push_back(pw);
          auto d = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
          set_desc(d, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pw);
          set_desc(d, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, tensor(op.in[0]));
          if (op.in.size() > 1) set_desc(d, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, tensor(op.in[1]));
          set_desc(d, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, tensor(op.out));
          finalize(d);
          ops.push_back(d);
          break;
        }
        case Graph::Impl::Kind::Pooling: {
          auto rs = make(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
          const cudnnResampleMode_t mode =
            op.pool.mode == PoolingMode::Max ? CUDNN_RESAMPLE_MAXPOOL : CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
          const cudnnPaddingMode_t padding_mode = CUDNN_ZERO_PAD;
          const int64_t dims = static_cast<int64_t>(op.pool.window.size());
          std::vector<cudnnFraction_t> window, stride, pad;
          for (int64_t i = 0; i < dims; ++i) {
            window.push_back({op.pool.window[i], 1});
            stride.push_back({op.pool.stride[i], 1});
            pad.push_back({op.pool.padding[i], 1});
          }
          set(rs, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, 1, &mode);
          set(rs, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
          set(rs, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &dims);
          set(rs, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, dims, window.data());
          set(rs, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, dims, stride.data());
          set(rs, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, dims, pad.data());
          set(rs, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, dims, pad.data());
          set(rs, CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_TYPE_PADDING_MODE, 1, &padding_mode);
          finalize(rs);
          b.keep.push_back(rs);
          auto d = make(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
          set_desc(d, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, tensor(op.in[0]));
          set_desc(d, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, tensor(op.out));
          set_desc(d, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, rs);
          finalize(d);
          ops.push_back(d);
          break;
        }
        case Graph::Impl::Kind::Matmul: {
          auto mm = make(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
          set(mm, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
          finalize(mm);
          b.keep.push_back(mm);
          auto d = make(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
          set_desc(d, CUDNN_ATTR_OPERATION_MATMUL_ADESC, tensor(op.in[0]));
          set_desc(d, CUDNN_ATTR_OPERATION_MATMUL_BDESC, tensor(op.in[1]));
          set_desc(d, CUDNN_ATTR_OPERATION_MATMUL_CDESC, tensor(op.out));
          set_desc(d, CUDNN_ATTR_OPERATION_MATMUL_DESC, mm);
          finalize(d);
          ops.push_back(d);
          break;
        }
        }
      }
      b.graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
      std::vector<cudnnBackendDescriptor_t> raw;
      for (const auto& op : ops)
        raw.push_back(op.get());
      set(b.graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
      set(
        b.graph,
        CUDNN_ATTR_OPERATIONGRAPH_OPS,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        static_cast<int64_t>(raw.size()),
        raw.data());
      finalize(b.graph);
      b.keep.insert(b.keep.end(), tensors.begin(), tensors.end());
      b.keep.insert(b.keep.end(), ops.begin(), ops.end());
      return b;
    }

    // The engine configurations cuDNN's heuristics propose, best first.
    std::vector<Handle> engine_configs(const Handle& graph, cudnnBackendHeurMode_t mode)
    {
      auto heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
      set_desc(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
      set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
      if (cudnnBackendFinalize(heur.get()) != CUDNN_STATUS_SUCCESS) return {};
      int64_t count = 0;
      if (
        cudnnBackendGetAttribute(
          heur.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr) !=
          CUDNN_STATUS_SUCCESS ||
        count == 0) {
        return {};
      }
      std::vector<Handle> configs;
      std::vector<cudnnBackendDescriptor_t> raw;
      for (int64_t i = 0; i < count; ++i) {
        configs.push_back(make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
        raw.push_back(configs.back().get());
      }
      int64_t got = 0;
      ALLEN_CUDNN_CHECK(cudnnBackendGetAttribute(
        heur.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &got, raw.data()));
      configs.resize(got);
      return configs;
    }

    // Whether the options allow the configuration's engine; its description.
    bool allowed(const Handle& config, const BuildOptions& options, DataType io, std::string& description)
    {
      auto engine = make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
      cudnnBackendDescriptor_t raw = engine.get();
      int64_t n = 0;
      if (
        cudnnBackendGetAttribute(
          config.get(), CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n, &raw) !=
        CUDNN_STATUS_SUCCESS) {
        return false;
      }
      int64_t index = -1;
      cudnnBackendGetAttribute(raw, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &n, &index);
      cudnnBackendNumericalNote_t numerical[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
      int64_t n_numerical = 0;
      cudnnBackendGetAttribute(
        raw,
        CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
        CUDNN_TYPE_NUMERICAL_NOTE,
        CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
        &n_numerical,
        numerical);
      cudnnBackendBehaviorNote_t behavior[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
      int64_t n_behavior = 0;
      cudnnBackendGetAttribute(
        raw,
        CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
        CUDNN_TYPE_BEHAVIOR_NOTE,
        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
        &n_behavior,
        behavior);
      std::ostringstream s;
      s << "engine " << index;
      bool ok = true;
      for (int64_t i = 0; i < n_numerical; ++i) {
        switch (numerical[i]) {
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
          s << ", tensor cores";
          // float32 on tensor cores is TF32
          if (io == DataType::Float && !options.allow_reduced_precision) ok = false;
          break;
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
          s << ", down-converted inputs";
          if (!options.allow_reduced_precision) ok = false;
          break;
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
          s << ", reduced-precision reduction";
          if (!options.allow_reduced_precision) ok = false;
          break;
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
          s << ", non-deterministic";
          if (!options.allow_nondeterministic) ok = false;
          break;
        case CUDNN_NUMERICAL_NOTE_FFT: s << ", FFT"; break;
        case CUDNN_NUMERICAL_NOTE_WINOGRAD: s << ", Winograd"; break;
        default: break;
        }
      }
      for (int64_t i = 0; i < n_behavior; ++i) {
        if (behavior[i] == CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION) {
          s << ", compiled at run time";
          if (!options.allow_runtime_compilation) ok = false;
        }
      }
      description = s.str();
      return ok;
    }

    Handle make_plan(cudnnHandle_t handle, const Handle& config, size_t& workspace)
    {
      cudnnBackendDescriptor_t d = nullptr;
      if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &d) != CUDNN_STATUS_SUCCESS) return {};
      Handle plan(d, [](cudnnBackendDescriptor_t p) { cudnnBackendDestroyDescriptor(p); });
      cudnnBackendDescriptor_t raw = config.get();
      if (
        cudnnBackendSetAttribute(d, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle) !=
          CUDNN_STATUS_SUCCESS ||
        cudnnBackendSetAttribute(d, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw) !=
          CUDNN_STATUS_SUCCESS ||
        cudnnBackendFinalize(d) != CUDNN_STATUS_SUCCESS) {
        return {};
      }
      int64_t bytes = 0, n = 0;
      ALLEN_CUDNN_CHECK(
        cudnnBackendGetAttribute(d, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &n, &bytes));
      workspace = static_cast<size_t>(bytes);
      return plan;
    }

    // Median time of a plan on scratch memory (contents irrelevant), in ms.
    float time_plan(cudnnHandle_t handle, const Plan& plan, const Graph::Impl& g, const std::vector<int64_t>& bound)
    {
      cudaStream_t stream = nullptr;
      ALLEN_CUDNN_CHECK(cudnnGetStream(handle, &stream));
      std::vector<void*> buffers;
      std::vector<const void*> pointers;
      for (const auto uid : bound) {
        const auto& t = g.tensors[uid - 1];
        size_t n = 1;
        for (size_t i = 0; i < t.dims.size(); ++i)
          n += (t.dims[i] - 1) * t.strides[i];
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, n * element_size(t.type)), "cudaMalloc");
        cuda_check(cudaMemsetAsync(p, 0, n * element_size(t.type), stream), "cudaMemsetAsync");
        buffers.push_back(p);
        pointers.push_back(p);
      }
      void* workspace = nullptr;
      if (plan.workspace_size() > 0) cuda_check(cudaMalloc(&workspace, plan.workspace_size()), "cudaMalloc");
      cudaEvent_t start, stop;
      cuda_check(cudaEventCreate(&start), "cudaEventCreate");
      cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
      std::vector<float> times;
      plan.execute(handle, pointers, workspace); // warm-up
      for (int i = 0; i < 10; ++i) {
        cuda_check(cudaEventRecord(start, stream), "cudaEventRecord");
        plan.execute(handle, pointers, workspace);
        cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        times.push_back(ms);
      }
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      for (void* p : buffers)
        cudaFree(p);
      if (workspace) cudaFree(workspace);
      std::sort(times.begin(), times.end());
      return times[times.size() / 2];
    }
  } // namespace

  Plan Graph::try_build(cudnnHandle_t handle, const BuildOptions& options) const
  {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    std::ostringstream key;
    key << "device" << device << ";" << signature() << ";opt" << options.max_candidates
        << options.allow_reduced_precision << options.allow_nondeterministic << options.allow_runtime_compilation
        << options.max_workspace;
    {
      std::lock_guard<std::mutex> lock {cache_mutex};
      const auto it = plan_cache.find(key.str());
      if (it != plan_cache.end()) {
        Plan p;
        p.m_impl = it->second;
        return p;
      }
    }

    const Built built = build_operation_graph(*m_impl, handle);
    auto configs = engine_configs(built.graph, CUDNN_HEUR_MODE_A);
    for (auto& c : engine_configs(built.graph, CUDNN_HEUR_MODE_FALLBACK))
      configs.push_back(c);

    std::vector<std::pair<int64_t, float>> scalars;
    for (size_t i = 0; i < m_impl->tensors.size(); ++i) {
      if (m_impl->tensors[i].by_value) scalars.emplace_back(static_cast<int64_t>(i) + 1, m_impl->tensors[i].value);
    }

    Plan best;
    float best_time = 0.f;
    int candidates = 0;
    for (const auto& config : configs) {
      if (candidates >= options.max_candidates) break;
      std::string description;
      if (!allowed(config, options, m_impl->io, description)) continue;
      size_t workspace = 0;
      Handle plan = make_plan(handle, config, workspace);
      if (!plan || workspace > options.max_workspace) continue;
      auto impl = std::make_shared<Plan::Impl>();
      impl->config = config;
      impl->workspace = workspace;
      impl->engine = description + ", workspace " + std::to_string(workspace) + " B";
      impl->bound = m_impl->bound;
      impl->scalars = scalars;
      auto first = std::make_shared<Plan::Impl::PerHandle>();
      first->plan = plan;
      for (const auto& scalar : scalars)
        first->values.push_back(scalar.second);
      impl->per_handle.emplace(handle, std::move(first));
      Plan candidate;
      candidate.m_impl = impl;
      ++candidates;
      if (options.max_candidates == 1) {
        best = candidate;
        break;
      }
      const float t = time_plan(handle, candidate, *m_impl, m_impl->bound);
      if (!best.valid() || t < best_time) {
        best = candidate;
        best_time = t;
      }
    }
    if (best.valid()) {
      std::lock_guard<std::mutex> lock {cache_mutex};
      plan_cache.emplace(key.str(), best.m_impl);
    }
    return best;
  }

  Plan Graph::build(cudnnHandle_t handle, const BuildOptions& options) const
  {
    Plan plan = try_build(handle, options);
    if (!plan.valid()) {
      throw StrException("Allen::CuDNN::Graph: no cuDNN engine allowed by the options runs the graph " + signature());
    }
    return plan;
  }

} // namespace Allen::CuDNN
