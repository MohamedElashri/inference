/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "States.cuh"

namespace TTracks::Constants {
  constexpr static unsigned MAX_VERTICES_PER_EVENT = 256;
  constexpr static unsigned num_reference_states = 5;
  __device__ constexpr static float z_reference_states[num_reference_states] = {2400.f, 3800.f, 4800.f, 6300.f, 8600.f};
}; // namespace TTracks::Constants

namespace TTracks::Vertex {

  enum PackedMiniVertexVar : unsigned {
    trk1 = 0,
    trk2 = 1,
    px = 2,
    py = 3,
    pz = 4,
    x = 5,
    y = 6,
    z = 7,
    doca = 8,
    COUNT = 9
  };

  template<typename T>
  concept FloatElement = std::same_as<T, float> || std::same_as<T, const float>;

  template<typename T>
  concept MutableView = not std::is_const_v<T>;

  template<FloatElement DataType>
  class PackedMiniVertexT {
  public:
    __device__ __host__ inline PackedMiniVertexT(DataType* data_ptr, unsigned num_vertices) :
      m_data_ptr(data_ptr), m_num_vertices(num_vertices)
    {}

    template<PackedMiniVertexVar var, typename T>
    __device__ __host__ inline T get(unsigned idx) const
    {
      return m_data_ptr[index<var>(idx)];
    }

    template<PackedMiniVertexVar var, typename T>
    __device__ __host__ inline void set(unsigned idx, T value)
      requires(MutableView<T>)
    {
      m_data_ptr[index<var>(idx)] = value;
    }

  private:
    template<PackedMiniVertexVar var>
    __device__ __host__ inline unsigned index(unsigned idx) const
    {
      return m_num_vertices * static_cast<unsigned>(var) + idx;
    }

    DataType* m_data_ptr;
    unsigned m_num_vertices;
  };

  using PackedMiniVertex = PackedMiniVertexT<const float>;
  using PackedMiniVertexMutable = PackedMiniVertexT<float>;

}; // namespace TTracks::Vertex

namespace TTracks::Tracks {

  enum PackedTrackStateVar : unsigned { x = 0, y = 1, z = 2, tx = 3, ty = 4, COUNT = 5 };

  class PackedTrackState {
  public:
    __device__ __host__ inline PackedTrackState(float* data_ptr, unsigned n_states) :
      m_data_ptr(data_ptr), m_n_states(n_states)
    {}

    template<PackedTrackStateVar var>
    __device__ __host__ inline float get(unsigned idx) const
    {
      return *(index<var>(idx));
    }

    template<PackedTrackStateVar var>
    __device__ __host__ inline void set(unsigned idx, float value)
    {
      *(index<var>(idx)) = value;
    }

    __device__ __host__ inline void set(unsigned idx, MiniState state)
    {
      set<PackedTrackStateVar::x>(idx, state.m_x);
      set<PackedTrackStateVar::y>(idx, state.m_y);
      set<PackedTrackStateVar::z>(idx, state.m_z);
      set<PackedTrackStateVar::tx>(idx, state.m_tx);
      set<PackedTrackStateVar::ty>(idx, state.m_ty);
    }

  private:
    template<PackedTrackStateVar var>
    __device__ __host__ inline float* index(unsigned idx) const
    {
      return m_data_ptr + m_n_states * static_cast<unsigned>(var) + idx;
    }

    float* m_data_ptr;
    unsigned m_n_states;
  };

  class PackedTrackStateConst {
  public:
    __device__ __host__ inline PackedTrackStateConst(const float* data_ptr, unsigned n_states) :
      m_data_ptr(data_ptr), m_n_states(n_states)
    {}

    template<PackedTrackStateVar var>
    __device__ __host__ inline float get(unsigned idx) const
    {
      return *(index<var>(idx));
    }

  private:
    template<PackedTrackStateVar var>
    __device__ __host__ inline const float* index(unsigned idx) const
    {
      return m_data_ptr + m_n_states * static_cast<unsigned>(var) + idx;
    }

    const float* m_data_ptr;
    unsigned m_n_states;
  };

}; // namespace TTracks::Tracks

namespace TTracks::Combinations {

  __device__ std::pair<unsigned, unsigned> inline convert_idx(unsigned combIdx, unsigned n_input_states)
  {
    const unsigned passIdx = combIdx / (n_input_states - 1);
    const unsigned elIdx = combIdx % (n_input_states - 1);
    const auto firstPack = elIdx < n_input_states - 1 - passIdx;
    return std::make_pair(
      firstPack ? passIdx : n_input_states - 1 - passIdx, firstPack ? passIdx + 1 + elIdx : elIdx + 1);
  }

  __device__ unsigned inline number_of_combinations(unsigned n_input_states)
  {
    return n_input_states * (n_input_states - 1) / 2;
  }

} // namespace TTracks::Combinations
