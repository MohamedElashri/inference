/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

#include <atomic>
#include <mutex>
#include <iostream>
#include <Algorithm.cuh>

#ifndef ALLEN_STANDALONE
#include "ServiceLocator.h"
#endif

namespace Allen::Monitoring {
  struct AccumulatorBase;

  /// Guards the host-side accumulator values (bins, entries, sums). They are written by the
  /// aggregation thread in AccumulatorManager::mergeAndReset() and read (to_json) or reset
  /// (reset) by the Gaudi monitoring hub's sinks and users on other threads.
  inline std::mutex& hostDataMutex()
  {
    static std::mutex s_mutex;
    return s_mutex;
  }

  template<typename T>
  struct Counter;

  template<typename T>
  struct AveragingCounter;

  struct AccumulatorInfosAndPointers {
    std::size_t offset {0};
    std::size_t size {0};
    std::size_t element_size {0};
    std::vector<AccumulatorBase*> owners;
  };

  struct CountersHistogram {

    CountersHistogram() : m_title("CountersHistogram"), m_bins(2, 0.0f) {}

    friend void reset(CountersHistogram& c)
    {
      std::lock_guard lock {hostDataMutex()};
      std::fill(c.m_bins.begin(), c.m_bins.end(), 0.0f);
      c.m_totNEntries = 0.0;
    }

    friend void to_json(nlohmann::json& j, CountersHistogram const& h)
    {
      std::lock_guard lock {hostDataMutex()};
      j = {
        {"type", "histogram:WeightedHistogram:d"},
        {"title", h.m_title},
        {"dimension", 1},
        {"empty", h.m_totNEntries == 0},
        {"nEntries", h.m_totNEntries},
        {"axis",
         {{{"nBins", h.m_bins.size() - 2},
           {"minValue", h.m_minValue},
           {"maxValue", h.m_maxValue},
           {"title", ""},
           {"labels", h.m_labels}}}},
        {"bins", h.m_bins}};
    }

    void registerHistogram()
    {

      // Handle warning if no counters are used in the sequence
      if (m_labels.size() == 0) {
        m_labels.push_back("empty_bin");
        m_maxValue++;
        m_bins.push_back(0.f);
      }

// Register CountersHistogram for Gaudi
#ifndef ALLEN_STANDALONE
      Gaudi::svcLocator()->monitoringHub().registerEntity(
        "CountersHistogram", "CountersValues", "histogram:WeightedHistogram:d", *this);
#endif
    }

    void addCounter(std::string label)
    {
      m_labels.push_back(label);
      m_maxValue++;
      m_bins.push_back(0.f);
    }

    void addAvCounter(const std::string& label)
    {
      m_labels.insert(m_labels.end(), {label + "_sum", label + "_n_entries"});
      m_maxValue += 2;
      m_bins.insert(m_bins.end(), 2, 0.f);
    }

    void updateBin(int bin_index, float value) { m_bins[bin_index + 1] = value; }

    void resetHistogram()
    {
      m_bins = std::vector<float>(2, 0.f);
      m_labels = std::vector<std::string>();
      m_minValue = 0;
      m_maxValue = 0;
    }

    std::string m_title;
    std::vector<float> m_bins;
    int m_minValue = 0;
    int m_maxValue = 0;
    std::vector<std::string> m_labels;
    unsigned m_totNEntries = 0;
  };

  struct AccumulatorManager {
    static AccumulatorManager* get()
    {
      static AccumulatorManager instance;
      return &instance;
    }

    void registerAccumulator(AccumulatorBase* acc);
    void registerCounter(Counter<unsigned>* c) { m_counters.push_back(c); }
    void registerAveragingCounter(AveragingCounter<unsigned>* c) { m_av_counters.push_back(c); }
    void initAccumulators(unsigned number_of_streams);
    void mergeAndReset();
    char* bufferForStream(unsigned stream_id) const
    {
      return m_dev_buffer_ptr[m_stream_current_buffer[stream_id].load(std::memory_order_acquire)];
    }
    void synchronizeStream(unsigned stream_id)
    {
      m_stream_done[stream_id].store(false, std::memory_order_release);
      m_stream_current_buffer[stream_id].store(
        m_current_buffer.load(std::memory_order_acquire), std::memory_order_release);
    }
    void streamDone(unsigned stream_id) { m_stream_done[stream_id].store(true, std::memory_order_release); }

  private:
    char* m_dev_buffer_ptr[2] {nullptr, nullptr}; // double buffering
    char* m_host_buffer_ptr {nullptr};
    std::size_t m_buffer_size {0};

    // Written by the aggregation thread (mergeAndReset()) and/or each stream's own
    // worker thread (synchronizeStream()/streamDone()), read cross-thread by the
    // other side -- must be genuinely atomic, not just re-typed, to avoid a data
    // race under the C++ memory model (see lhcb/Allen#630). initAccumulators() runs
    // once, single-threaded, strictly before any worker/aggregation thread starts,
    // so it move-assigns freshly-sized vectors instead of resizing (std::atomic<T>
    // is neither copy- nor move-constructible, so resize() would not compile).
    std::atomic<unsigned> m_current_buffer {0};
    std::vector<std::atomic<unsigned>> m_stream_current_buffer;
    std::vector<std::atomic<bool>> m_stream_done;

    CountersHistogram m_counters_histogram;

    std::vector<AccumulatorBase*> m_registered_accumulators;
    std::vector<Counter<unsigned>*> m_counters;
    std::vector<AveragingCounter<unsigned>*> m_av_counters;
    std::map<std::string, AccumulatorInfosAndPointers> m_accumulators;
  };

  struct AccumulatorBase {
    AccumulatorBase(const Allen::Algorithm* owner, std::string name) : m_owner(owner), m_name(name)
    {
      AccumulatorManager::get()->registerAccumulator(this);
    }
    virtual ~AccumulatorBase();
    std::string component() const { return m_owner->name(); }
    std::string name() const { return m_name; }
    std::string uniqueName() const { return m_owner->name() + ":" + m_name; }
    virtual std::size_t size() const { return 0; }
    virtual std::size_t elementSize() const { return 0; }
    virtual void registerAccumulator() {}
    virtual void fillAccumulator(void*) {}

    void setBufferInfos(const AccumulatorInfosAndPointers* infos) { m_buffer_infos = infos; }
    char* currentDevicePtr(unsigned stream_id) const
    {
      return AccumulatorManager::get()->bufferForStream(stream_id) + m_buffer_infos->offset;
    }

  protected:
    // Set by a derived class' registerAccumulator() override once it has actually
    // registered *this with Gaudi::svcLocator()->monitoringHub(); the destructor
    // only calls removeEntity() if this is true. Needed because AccumulatorManager
    // only calls registerAccumulator() on the first owner of a given unique name
    // (see AccumulatorManager::initAccumulators), and because tools that
    // introspect algorithms without a real Gaudi ApplicationMgr (e.g.
    // configuration/src/default_properties.cpp) construct and destroy
    // AccumulatorBase-derived objects without ever calling initAccumulators() at
    // all -- Gaudi::svcLocator() lazily creates (and leaks) a whole ApplicationMgr
    // if this destructor calls it unconditionally in that context.
    bool m_registered {false};

  private:
    const Allen::Algorithm* m_owner;
    std::string m_name;
    const AccumulatorInfosAndPointers* m_buffer_infos {nullptr};
  };

  // Counters:

  template<typename T = unsigned>
  struct DeviceCounter {
    __host__ __device__ DeviceCounter(T* data) : m_data(data) {}
    __device__ T* data() const { return m_data; }

#if defined(TARGET_DEVICE_CUDA) && defined(DEVICE_COMPILER)
    __device__ void increment() const
    {
      unsigned mask = __activemask();
      unsigned peers = mask;
      unsigned count = __popc(peers);
      int rank = __popc(peers & __lanemask_lt());
      bool is_leader = rank == 0;
      if (is_leader) {
        atomicAdd(&m_data[0], count);
      }
    }
#else
    __device__ void increment() const { __atomic_add_fetch(&m_data[0], 1, __ATOMIC_RELAXED); }
#endif

  private:
    T* m_data;
  };

  template<typename T = unsigned>
  struct Counter : AccumulatorBase {
    using type = T;
    using DeviceType = DeviceCounter<T>;

    Counter(const Allen::Algorithm* owner, std::string name) : AccumulatorBase(owner, name)
    {
      if constexpr (std::is_same<T, unsigned>::value) AccumulatorManager::get()->registerCounter(this);
    }
    std::size_t size() const override { return 1; }
    std::size_t elementSize() const override { return sizeof(T); }
    DeviceType data(const Allen::Context& ctx) const { return reinterpret_cast<T*>(currentDevicePtr(ctx.stream_id)); }

    friend void reset(Counter& c)
    {
      std::lock_guard lock {hostDataMutex()};
      c.m_entries = 0.0;
    }
    friend void to_json(nlohmann::json& j, Counter const& c)
    {
      std::lock_guard lock {hostDataMutex()};
      j = {{"type", "counter:Counter:d"}, {"empty", LHCb::essentiallyZero(c.m_entries)}, {"nEntries", c.m_entries}};
    }
    void registerAccumulator() override
    {
#ifndef ALLEN_STANDALONE
      Gaudi::svcLocator()->monitoringHub().registerEntity(component(), name(), "counter:Counter:d", *this);
      m_registered = true;
#endif
    }
    void fillAccumulator(void* ptr) override { m_entries += reinterpret_cast<T*>(ptr)[0]; }
    double m_entries = 0.0;
  };

  template<typename T = unsigned>
  struct DeviceAveragingCounter {
    __host__ __device__ DeviceAveragingCounter(T* data) : m_data(data) {}
    __device__ T* data() const { return m_data; }

#if defined(TARGET_DEVICE_CUDA) && defined(DEVICE_COMPILER)
    __device__ void add(T value) const
    {
      unsigned mask = __activemask();
      unsigned peers = mask;
      unsigned count = __popc(peers);
      int rank = __popc(peers & __lanemask_lt());
      bool is_leader = rank == 0;
      peers &= __lanemask_gt();
      while (__any_sync(mask, peers)) {
        int next = __ffs(peers);
        T tmp = __shfl_sync(mask, value, next - 1);
        if (next) value += tmp;
        peers &= __ballot_sync(mask, !(rank & 1));
        rank >>= 1;
      }
      if (is_leader) {
        atomicAdd(&m_data[0], value);
        atomicAdd(&m_data[1], count);
      }
    }
#else
    __device__ void add(T value) const
    {
      __atomic_add_fetch(&m_data[0], value, __ATOMIC_RELAXED);
      __atomic_add_fetch(&m_data[1], 1, __ATOMIC_RELAXED);
    }
#endif

  private:
    T* m_data;
  };

  template<typename T = unsigned>
  struct AveragingCounter : AccumulatorBase {
    using type = T;
    using DeviceType = DeviceAveragingCounter<T>;

    AveragingCounter(const Allen::Algorithm* owner, std::string name) : AccumulatorBase(owner, name)
    {
      if constexpr (std::is_same<T, unsigned>::value) AccumulatorManager::get()->registerAveragingCounter(this);
    }
    std::size_t size() const override { return 2; }
    std::size_t elementSize() const override { return sizeof(T); }
    DeviceType data(const Allen::Context& ctx) const { return reinterpret_cast<T*>(currentDevicePtr(ctx.stream_id)); }

    friend void reset(AveragingCounter& c)
    {
      std::lock_guard lock {hostDataMutex()};
      c.m_sum = 0.0;
      c.m_entries = 0.0;
    }
    friend void to_json(nlohmann::json& j, AveragingCounter const& c)
    {
      std::lock_guard lock {hostDataMutex()};
      j = {
        {"type", "counter:AveragingCounter:d"},
        {"empty", LHCb::essentiallyZero(c.m_entries)},
        {"nEntries", c.m_entries},
        {"sum", c.m_sum},
        {"mean", c.m_sum / c.m_entries}};
    }
    void registerAccumulator() override
    {
#ifndef ALLEN_STANDALONE
      Gaudi::svcLocator()->monitoringHub().registerEntity(component(), name(), "counter:AveragingCounter:d", *this);
      m_registered = true;
#endif
    }
    void fillAccumulator(void* ptr) override
    {
      m_sum += reinterpret_cast<T*>(ptr)[0];
      m_entries += reinterpret_cast<T*>(ptr)[1];
    }
    double m_sum = 0.0;
    double m_entries = 0.0;
  };

  // Histograms:

  template<typename T>
  struct DeviceAxis {
    using InputType = T;
    DeviceAxis() = default;
    DeviceAxis(unsigned nBins, InputType minValue, InputType maxValue) :
      nBins(nBins), minValue(minValue), ratio(static_cast<float>(nBins) / (maxValue - minValue))
    {}
    // Returns the Gaudi/ROOT bin index: 0 is the underflow bin, nBins + 1 is the
    // overflow bin and the inner bins are shifted by one. The bin position is
    // clamped to [-1, nBins] before the integer conversion, so out-of-range values
    // fall in the flow bins and NaN (which fminf/fmaxf resolve to the overflow bin)
    // never reaches the undefined float-to-integer conversion.
    __host__ __device__ unsigned index(InputType value) const
    {
      const float position = (static_cast<float>(value) - static_cast<float>(minValue)) * ratio;
      const float clamped = fmaxf(fminf(position, static_cast<float>(nBins)), -1.f);
      return static_cast<unsigned>(static_cast<int>(clamped + 1.f));
    }

    unsigned nBins {0};

  private:
    T minValue;
    float ratio {0.f};
  };

  template<typename T>
  struct Axis {
    using DeviceType = DeviceAxis<T>;
    using InputType = T;

    Axis(unsigned nBins, T minValue, T maxValue, std::string title = {}, std::vector<std::string> labels = {}) :
      nBins(nBins), minValue(minValue), maxValue(maxValue), title(title), labels(labels)
    {}

    DeviceType deviceAxis() const { return {nBins, minValue, maxValue}; }

    friend void to_json(nlohmann::json& j, const Axis& axis)
    {
      j = nlohmann::json {
        {"nBins", axis.nBins}, {"minValue", axis.minValue}, {"maxValue", axis.maxValue}, {"title", axis.title}};
      if (!axis.labels.empty()) {
        j["labels"] = axis.labels;
      }
    }

    unsigned int nBins;              // number of bins for this Axis
    T minValue, maxValue;            // min and max values on this axis
    std::string title;               // title of this axis
    std::vector<std::string> labels; // labels for the bins
  };

  namespace {
    __device__ __host__ constexpr float logscale(float x, float a, float b, float c) { return log2f(a * x + c) * b; }
  } // namespace

  struct DeviceLogAxis {
    using InputType = float;
    DeviceLogAxis() = default;
    DeviceLogAxis(unsigned nBins, float _minValue, float _maxValue, float a, float b, float c) :
      nBins(nBins), a(a), b(b), c(c)
    {
      minValue = logscale(_minValue, a, b, c);
      ratio = static_cast<float>(nBins) / (logscale(_maxValue, a, b, c) - minValue);
    }
    // Returns the Gaudi/ROOT bin index: 0 is the underflow bin, nBins + 1 is the
    // overflow bin and the inner bins are shifted by one. The transform is applied
    // first, so the flow bins are defined on the transformed axis.
    __host__ __device__ unsigned index(InputType value) const
    {
      const float position = (logscale(value, a, b, c) - minValue) * ratio;
      const float clamped = fmaxf(fminf(position, static_cast<float>(nBins)), -1.f);
      return static_cast<unsigned>(static_cast<int>(clamped + 1.f));
    }

    unsigned nBins {0};

  private:
    float minValue {0.f};
    float ratio {0.f}, a {0.f}, b {0.f}, c {0.f};
  };

  // An axis with a transform of the form y = log2(a * x + c) * b
  // The default parameters makes it equivalent to y = log10(x)
  struct LogAxis {
    using DeviceType = DeviceLogAxis;
    using InputType = float;

    LogAxis(
      unsigned nBins,
      float minValue,
      float maxValue,
      float a = 1.f,
      float b = std::log10(2),
      float c = 0.f,
      std::string title = {}) :
      nBins(nBins),
      minValue(minValue), maxValue(maxValue), a(a), b(b), c(c), title(title)
    {}

    DeviceType deviceAxis() const { return {nBins, minValue, maxValue, a, b, c}; }

    friend void to_json(nlohmann::json& j, const LogAxis& axis)
    {
      std::vector<double> xbins;
      xbins.reserve(axis.nBins + 1);

      double minValue = static_cast<double>(logscale(axis.minValue, axis.a, axis.b, axis.c));
      double maxValue = static_cast<double>(logscale(axis.maxValue, axis.a, axis.b, axis.c));
      double step = (maxValue - minValue) / static_cast<double>(axis.nBins);
      double a = static_cast<double>(axis.a);
      double b = static_cast<double>(axis.b);
      double c = static_cast<double>(axis.c);

      for (unsigned i = 0; i <= axis.nBins; i++) {
        double y = minValue + i * step;
        xbins.emplace_back((-c + std::exp(y * std::log(2) / b)) / a);
      }
      j = nlohmann::json {
        {"nBins", axis.nBins},
        {"minValue", axis.minValue},
        {"maxValue", axis.maxValue},
        {"title", axis.title},
        {"xbins", xbins}};
    }

    unsigned int nBins;       // number of bins for this Axis
    float minValue, maxValue; // min and max values on this axis
    float a, b, c;            // scaling coefficients
    std::string title;        // title of this axis
  };

  namespace details {
    template<class... Args, std::size_t... Is>
    constexpr auto remove_last_helper(std::tuple<Args...> tp, std::index_sequence<Is...>)
    {
      return std::tuple {std::get<Is>(tp)...};
    }

    template<class... Args>
    constexpr auto remove_last(std::tuple<Args...> tp)
    {
      return remove_last_helper(tp, std::make_index_sequence<sizeof...(Args) - 1> {});
    }
  } // namespace details

  template<typename T = unsigned, typename... Types>
  struct DeviceNDHistogram {
    __host__ DeviceNDHistogram(T* data, std::tuple<Types...> axis_h) : m_data(data)
    {
      std::apply(
        [&](auto... axis) {
          unsigned i = 0;
          ((stride[i] = axis.nBins + 2, i++), ...);
          for (unsigned i = 0; (i + 2u) < sizeof...(Types); i++) {
            stride[i + 1] *= stride[i];
          }
        },
        details::remove_last(axis_h));
      m_axis = (std::apply(
        [&](auto... axis) { return std::tuple<decltype(axis.deviceAxis())...> {axis.deviceAxis()...}; }, axis_h));
    }

    template<typename First, typename... InputTypes>
    __host__ __device__ unsigned index(First& first, InputTypes&... values) const
    {
      unsigned sum = std::get<0>(m_axis).index(first);
      std::apply(
        [&](auto, auto... axis) {
          unsigned i = 0;
          ((sum += stride.at(i) * axis.index(values), i++), ...);
        },
        m_axis);
      return sum;
    }

    __device__ T* data() const { return m_data; }

#if defined(TARGET_DEVICE_CUDA) && defined(DEVICE_COMPILER)
    template<typename... InputTypes>
    __device__ void increment(InputTypes... values) const
    {
      // Based on https://hal.science/hal-03330414/document
      // Out-of-range values are not dropped but counted in the underflow/overflow
      // bins, as Gaudi/ROOT histograms do.
      unsigned index_ = index(values...);
      unsigned active = __activemask();
      unsigned peers = conflict_mask(active, index_);
      unsigned count = __popc(peers);
      unsigned rank = __popc(peers & __lanemask_lt());
      if (rank == 0) atomicAdd(&m_data[index_], count);
    }
#else
    template<typename... InputTypes>
    void increment(InputTypes... values) const
    {
      unsigned index_ = index(values...);
      __atomic_add_fetch(&m_data[index_], 1, __ATOMIC_RELAXED);
    }
#endif

  private:
    T* m_data;
    std::tuple<typename Types::DeviceType...> m_axis;
    std::array<unsigned, sizeof...(Types) - 1> stride;
  };

  template<typename T = unsigned, typename... Types>
  struct HistogramND : AccumulatorBase {
    using type = T;
    using DeviceType = DeviceNDHistogram<T, Types...>;

    HistogramND(const Allen::Algorithm* owner, std::string name, std::string title, Types... axis) :
      AccumulatorBase(owner, name), m_title(title), m_axis(axis...)
    {}

    std::size_t size() const override
    {
      // The device buffer mirrors the Gaudi/ROOT layout, i.e. each axis has two
      // extra bins (underflow and overflow).
      return std::apply([&](auto... axis) { return (std::size_t {1} * ... * (axis.nBins + 2)); }, m_axis);
    }

    std::size_t elementSize() const override { return sizeof(T); }

    DeviceType data(const Allen::Context& ctx) const
    {
      T* ptr = reinterpret_cast<T*>(currentDevicePtr(ctx.stream_id));
      return DeviceType(ptr, m_axis);
    }

    auto& x_axis() { return std::get<0>(m_axis); }

    auto& y_axis() { return std::get<1>(m_axis); }

    auto& z_axis() { return std::get<2>(m_axis); }

    friend void reset(HistogramND& c)
    {
      std::lock_guard lock {hostDataMutex()};
      std::fill(c.m_bins.begin(), c.m_bins.end(), 0.0);
      c.m_totNEntries = 0.0;
    }

    friend void to_json(nlohmann::json& j, HistogramND const& h)
    {
      std::lock_guard lock {hostDataMutex()};
      j = {
        {"type", "histogram:Histogram:d"},
        {"title", h.m_title},
        {"dimension", sizeof...(Types)},
        {"empty", LHCb::essentiallyZero(h.m_totNEntries)},
        {"nEntries", h.m_totNEntries},
        {"axis", h.axisArray()},
        {"bins", h.m_bins}};
    }

    void registerAccumulator() override
    {
      m_bins.assign(size(), 0.0);
      m_totNEntries = 0.0;
#ifndef ALLEN_STANDALONE
      Gaudi::svcLocator()->monitoringHub().registerEntity(component(), name(), "histogram:Histogram:d", *this);
      m_registered = true;
#endif
    }

    void fillAccumulator(void* ptr) override
    {
      // Device and host buffers now share the same Gaudi/ROOT layout, so all
      // counts -- including the underflow/overflow bins -- can be copied directly.
      for (std::size_t bin = 0; bin < m_bins.size(); bin++) {
        auto count = reinterpret_cast<T*>(ptr)[bin];
        m_bins[bin] += count;
        m_totNEntries += count;
      }
    }

    std::string m_title;
    double m_totNEntries = 0.0;
    std::vector<double> m_bins;
    std::tuple<Types...> m_axis;

  private:
    constexpr auto axisArray() const
    {
      auto axis_arrays = (std::apply([&](auto... axis) { return std::array {axis...}; }, m_axis));

      return axis_arrays;
    }
  };

  template<typename AxisT = Axis<float>, typename T = unsigned>
  using Histogram = HistogramND<T, AxisT>;

  template<typename AxisT = Axis<float>, typename AxisT2 = Axis<float>, typename T = unsigned>
  using Histogram2D = HistogramND<T, AxisT, AxisT2>;

  template<typename AxisT = LogAxis, typename T = unsigned>
  using LogHistogram = HistogramND<T, AxisT>;

  template<typename HistogramType>
  struct HistogramBinAsCounter {
    HistogramBinAsCounter() = default;

    HistogramBinAsCounter(
      [[maybe_unused]] const Allen::Algorithm* owner,
      [[maybe_unused]] std::string name,
      const HistogramType* histo,
      unsigned bin) :
      m_histo(histo),
      m_bin(bin)
    {
#ifndef ALLEN_STANDALONE
      Gaudi::svcLocator()->monitoringHub().registerEntity(owner->name(), name, "counter:Counter:d", *this);
      m_registered = true;
#endif
    }
    ~HistogramBinAsCounter()
    {
#ifndef ALLEN_STANDALONE
      if (m_registered) {
        Gaudi::svcLocator()->monitoringHub().removeEntity(*this);
      }
#endif
    }
    friend void to_json(nlohmann::json& j, HistogramBinAsCounter const& c)
    {
      std::lock_guard lock {hostDataMutex()};
      const auto entries =
        (c.m_histo != nullptr && c.m_bin + 1 < c.m_histo->m_bins.size()) ? c.m_histo->m_bins[c.m_bin + 1] : 0.0;
      j = {{"type", "counter:Counter:d"}, {"empty", LHCb::essentiallyZero(entries)}, {"nEntries", entries}};
    }
    const HistogramType* m_histo {nullptr};
    unsigned m_bin {0};
    bool m_registered {false};
  };
} // namespace Allen::Monitoring
