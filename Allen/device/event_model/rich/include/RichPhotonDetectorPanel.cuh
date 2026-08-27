/*****************************************************************************\
 * (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#pragma once
#include <RichDefinitions.cuh>
#include <RichPhotonDetector.cuh>
#include <iostream>

namespace Allen::Rich::Detector {
  // https://gitlab.cern.ch/lhcb/LHCb/-/blob/master/Rich/RichDetectors/include/RichDetectors/RichPDPanel.h?ref_type=heads#L697
  struct PDPanelLookup {
    __host__ void
    init(const DetectorType rich, const std::array<float, 12>& g2panel, const PhotonDetector* pds, unsigned n_pds)
    {
      std::vector<unsigned> non_null_pds;
      std::vector<float2> lpos;
      std::vector<short2> lpos16;
      std::vector<short2> tols;
      non_null_pds.reserve(n_pds);
      lpos.reserve(n_pds);
      lpos16.reserve(n_pds);
      tols.reserve(n_pds);

      // Cache positions and indices and find extends
      for (unsigned i = 0; i < n_pds; i++) {
        const auto& pd = pds[i];
        float2 p = pd.centrePointPanel(g2panel);
        lpos16.emplace_back(make_short2(p.x * ((1 << 15) / 750.f) + .5f, p.y * ((1 << 15) / 750.f) + .5f));
        const float tolx = 4.f * pd.getEffectivePixelXSize();
        const float toly = 4.f * pd.getEffectivePixelYSize();
        tols.emplace_back(make_short2(tolx * ((1 << 15) / 100.f) + .5f, toly * ((1 << 15) / 100.f) + .5f));
        if (pd.getIsNull()) continue;
        non_null_pds.emplace_back(i);
        lpos.emplace_back(p);
        if (m_minX > p.x) m_minX = p.x;
        if (m_maxX < p.x) m_maxX = p.x;
        if (m_minY > p.y) m_minY = p.y;
        if (m_maxY < p.y) m_maxY = p.y;
      }

      // Brute force second loop over PDs to deduce the separations between them
      float minXdiff_N {9e20}, minYdiff_N {9e20};
      float minXdiff_S {9e20}, minYdiff_S {9e20};
      for (unsigned i = 0; i < non_null_pds.size(); i++) {
        const auto& pdA = pds[non_null_pds[i]];
        if (pdA.m_isHType) continue;
        const float2 ptnA = lpos[i];
        for (unsigned j = 0; j < i; j++) {
          const auto& pdB = pds[non_null_pds[j]];
          if (pdB.m_isHType) continue;

          const float2 ptnB = lpos[j];

          const float xDiff = fabsf(ptnB.x - ptnA.x);
          const float yDiff = fabsf(ptnB.y - ptnA.y);

          // Column numbers
          const int colA = Allen::Rich::Decoding::SmartID {pdA.pdSmartID()}.panelLocalModuleColumn();
          const int colB = Allen::Rich::Decoding::SmartID {pdB.pdSmartID()}.panelLocalModuleColumn();

          // Diffs for PDs in neighbouring columns
          if (std::abs(colB - colA) == 1) {
            if (xDiff > 1 && xDiff < minXdiff_N) {
              minXdiff_N = xDiff;
            }
            if (yDiff > 1 && yDiff < minYdiff_N) {
              minYdiff_N = yDiff;
            }
          }
          // Diffs for PDs in same column
          if (colB == colA) {
            if (xDiff > 1 && xDiff < minXdiff_S) {
              minXdiff_S = xDiff;
            }
            if (yDiff > 1 && yDiff < minYdiff_S) {
              minYdiff_S = yDiff;
            }
          }
        }
      }

      // Add half neighbouring size diff to min max values
      m_minX -= 0.5f * minXdiff_N;
      m_maxX += 0.5f * minXdiff_N;
      m_minY -= 0.5f * minYdiff_N;
      m_maxY += 0.5f * minYdiff_N;
      if (rich == Rich2) {
        // Do not fully understand this but RICH2 requires a 1/2 shift in Y..
        m_minY -= 0.5f * minYdiff_N;
        m_maxY -= 0.5f * minYdiff_N;
      }

      // bin size is average of N and S diffs
      const auto minXdiff = 0.5f * (minXdiff_N + minXdiff_S);
      const auto minYdiff = 0.5f * (minYdiff_N + minYdiff_S);

      // Compute grid size
      m_incX = 1.f / minXdiff;
      m_incY = 1.f / minYdiff;

      // Compute bin size
      m_binsX = std::round((m_maxX - m_minX) / minXdiff);
      m_binsY = std::round((m_maxY - m_minY) / minYdiff);

      // initialise the look up table
      std::vector<short> indices;
      indices.reserve(m_binsX * m_binsY);

      for (auto iY = 0u; iY < m_binsY; ++iY) {
        for (auto iX = 0u; iX < m_binsX; ++iX) {
          float x = m_minX + (((float) iX + 0.5f) / m_incX);
          float y = m_minY + (((float) iY + 0.5f) / m_incY);
          short best_id = -1;
          float min_d = 9e20;
          for (unsigned i = 0; i < non_null_pds.size(); i++) {
            short pd_id = non_null_pds[i];
            const float dx = lpos[i].x - x;
            const float dy = lpos[i].y - y;
            const float d = dx * dx + dy * dy;
            if (d < min_d) {
              best_id = pd_id;
              min_d = d;
            }
          }
          indices.emplace_back(best_id);
        }
      }

      if (m_indices != nullptr) Allen::free(m_indices);
      Allen::malloc((void**) &m_indices, indices.size() * sizeof(short));
      Allen::memcpy(m_indices, indices.data(), indices.size() * sizeof(short), Allen::memcpyHostToDevice);

      if (m_centres != nullptr) Allen::free(m_centres);
      Allen::malloc((void**) &m_centres, lpos16.size() * sizeof(short2));
      Allen::memcpy(m_centres, lpos16.data(), lpos16.size() * sizeof(short2), Allen::memcpyHostToDevice);

      if (m_tols != nullptr) Allen::free(m_tols);
      Allen::malloc((void**) &m_tols, tols.size() * sizeof(short2));
      Allen::memcpy(m_tols, tols.data(), tols.size() * sizeof(short2), Allen::memcpyHostToDevice);
    }
    __device__ short find(float2 P) const
    {
      if (std::isnan(P.x) || std::isnan(P.y) || P.x < m_minX || P.x > m_maxX || P.y < m_minY || P.y > m_maxY) {
        return -1;
      }
      // Note: handle acceptance here for now as we don't have the information dumped in the panel
      unsigned x = xindex(P.x);
      unsigned y = yindex(P.y);
      const auto id = m_indices[y * m_binsX + x];
      if (id == -1) return -1;

      const float cx = m_centres[id].x * (750.f / (1 << 15));
      const float cy = m_centres[id].y * (750.f / (1 << 15));
      // TODO: We can probably do better than storing the tols per PD
      // in rich 1 they are common to all PD and in rich 2 there are only 2 types
      const float tolx = m_tols[id].x * (100.f / (1 << 15));
      const float toly = m_tols[id].y * (100.f / (1 << 15));
      if (fabsf(cx - P.x) >= tolx || fabsf(cy - P.y) >= toly) {
        return -1;
      }
      return id;
    }

    __device__ unsigned xindex(float x) const
    {
      if (x <= m_minX) return 0;
      if (x >= m_maxX) return m_binsX - 1;
      const auto index = static_cast<unsigned>((x - m_minX) * m_incX);
      return index < m_binsX ? index : m_binsX - 1;
    }

    __device__ unsigned yindex(float y) const
    {
      if (y <= m_minY) return 0;
      if (y >= m_maxY) return m_binsY - 1;
      const auto index = static_cast<unsigned>((y - m_minY) * m_incY);
      return index < m_binsY ? index : m_binsY - 1;
    }

    float m_minX {9e9};  ///< Minimum X
    float m_maxX {-9e9}; ///< Maximum X
    float m_minY {9e9};  ///< Minimum Y
    float m_maxY {-9e9}; ///< Maximum Y
    float m_incX {0};    ///< 1 / Increment in X
    float m_incY {0};    ///< 1 / Increment in Y

    unsigned m_binsX {0};
    unsigned m_binsY {0};

    short* m_indices {nullptr};
    short2* m_centres {nullptr};
    short2* m_tols {nullptr};
  };

  template<DetectorType RichID>
  struct PDPanel {
    static constexpr unsigned ModuleColumnsPerPanel = RichID == Rich1 ? 11 : 12;
    static constexpr unsigned ModulesPerPanel = 6 * ModuleColumnsPerPanel;
    static constexpr unsigned ECsPerPanel = ModulesPerPanel * Decoding::SmartID::ECsPerModule;
    static constexpr unsigned PDsPerPanel =
      Decoding::SmartID::MaPMT::MaxPDsPerModule * ModulesPerPanel; // TODO: tighter for R2
    static constexpr unsigned PDsPerRich = NPDPanelsPerRICH * PDsPerPanel;

    static constexpr unsigned modNumOffset =
      RichID == Rich1 ? 1 : (NPDPanelsPerRICH * 6 * 11 + 6); // TODO: eliminate DD4HEP module offset?
    static constexpr unsigned modNumOffsetSide = RichID == Rich1 ? ModulesPerPanel : 6 * 14;

    __host__ __device__ static inline unsigned smartID2denseID(const Decoding::SmartID smartID)
    {
      const auto side = smartID.panel();
      const auto mod = smartID.pdMod() - modNumOffset - modNumOffsetSide * side;
      const auto ec = smartID.elementaryCell();
      const auto pdInEC = smartID.pdNumInEC();
      return mod * Decoding::SmartID::MaPMT::MaxPDsPerModule + ec * Decoding::SmartID::MaxPDsPerEC + pdInEC;
    }

    __host__ __device__ inline const PhotonDetector* pds() const
    {
      return m_PDs[0].data(); // flatten array
    }

    using PDArray = std::array<PhotonDetector, Decoding::SmartID::MaPMT::MaxPDsPerModule>;
    using ModuleArray = std::array<PDArray, ModulesPerPanel>;

    /// Access the RICH detector type
    __host__ __device__ inline auto rich() const { return RichID; }

    /// Access the PD panel side
    __host__ __device__ inline auto side() const noexcept { return m_side; }

    __host__ __device__ PhotonDetector dePD(const Decoding::SmartID smartID) const
    {
      PhotonDetector result;
      result.setIsNull(true);
      if (!smartID.pdIsSet()) return result;

      const auto denseID = smartID2denseID(smartID);
      return pds()[denseID];
    }

    /// Access Panel ID
    const std::array<int8_t, 3>& getPanelID() const { return m_panelID; }

    /// Compute the global detection point for a given RichSmartID
    __host__ __device__ inline auto globalDetectionPoint(const Decoding::SmartID id) const noexcept
    {
      const auto pd = dePD(id);
      return (!pd.getIsNull() ? pd.globalDetectionPoint(id) : Point {0, 0, 0});
    }

    /// Access the global to local transform
    __host__ __device__ inline const auto& globalToPDPanel() const noexcept { return m_gloToPDPanelM; }

    /// Access all owned PD Modules
    __host__ __device__ const ModuleArray& pdModules() const noexcept { return m_PDs; }

    /// Access the detection plane
    __host__ __device__ inline const Allen::Rich::Plane& detectionPlane() const noexcept { return m_detectionPlane; }

    std::string toString() const
    {
      std::stringstream ss;
      for (size_t i = 0; i < m_PDs.size(); ++i) {
        ss << "RICH " << (int) m_rich << ", Side " << (int) m_side << ", Module No. " << i << ", PDs: ";
        for (size_t j = 0; j < m_PDs[i].size(); ++j) {
          if (!(m_PDs[i][j].getIsNull())) {
            auto pd = m_PDs[i][j];
            ss << pd.toString() << '\n';
          }
          else {
            ss << " 0\n";
          }
        }
        ss << "\n";
      }
      return ss.str();
    }

    ModuleArray m_PDs {};
    std::array<float, 12> m_gloToPDPanelM {}; // 3D Transform
    Allen::Rich::Plane m_detectionPlane {};
    uint32_t m_modNumOffset {0};
    std::array<int8_t, 3> m_panelID {DetectorType::InvalidDetector, Side::InvalidSide, -1};
    DetectorType m_rich {DetectorType::InvalidDetector};
    Side m_side {Side::InvalidSide};
  };
} // namespace Allen::Rich::Detector
