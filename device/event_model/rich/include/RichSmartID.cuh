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
#include <string>
#include <ostream>
#include <iomanip>
#include "BackendCommon.h"
#include <RichDefinitions.cuh>

namespace Allen::Rich::Decoding {
  class SmartID {
    KeyType m_key;

  public:
    // Set the RICH PD pixel row identifier
    __host__ __device__ constexpr void setPixelRow(const DataType row)
    {
      setData(row, ShiftPixelRow, MaskPixelRow, MaskPixelRowIsSet);
    }

    // Set the RICH PD pixel column identifier
    __host__ __device__ constexpr void setPixelCol(const DataType col)
    {
      setData(col, ShiftPixelCol, MaskPixelCol, MaskPixelColIsSet);
    }

    __host__ __device__ SmartID() = default;

    __host__ __device__ SmartID(KeyType key) : m_key(key) {}

    // Number of bits for each data field in the word
    static constexpr const BitPackType BitsPixelCol = 3;   //< Number of bits for MaPMT pixel column
    static constexpr const BitPackType BitsPixelRow = 3;   //< Number of bits for MaPMT pixel row
    static constexpr const BitPackType BitsPDNumInMod = 4; //< Number of bits for MaPMT 'number in module'
    static constexpr const BitPackType BitsPDMod = 9;      //< Number of bits for MaPMT module
    static constexpr const BitPackType BitsPanel = 1;      //< Number of bits for MaPMT panel
    static constexpr const BitPackType BitsRich = 1;       //< Number of bits for RICH detector
    static constexpr const BitPackType BitsPixelSubRowIsSet = 1;
    static constexpr const BitPackType BitsPixelColIsSet = 1;
    static constexpr const BitPackType BitsPixelRowIsSet = 1;
    static constexpr const BitPackType BitsPDIsSet = 1;
    static constexpr const BitPackType BitsPanelIsSet = 1;
    static constexpr const BitPackType BitsRichIsSet = 1;
    static constexpr const BitPackType BitsLargePixel = 1;

    // The shifts
    static constexpr const BitPackType ShiftPixelCol = 0;
    static constexpr const BitPackType ShiftPixelRow = ShiftPixelCol + BitsPixelCol;
    static constexpr const BitPackType ShiftPDNumInMod = ShiftPixelRow + BitsPixelRow;
    static constexpr const BitPackType ShiftPDMod = ShiftPDNumInMod + BitsPDNumInMod;
    static constexpr const BitPackType ShiftPanel = ShiftPDMod + BitsPDMod;
    static constexpr const BitPackType ShiftRich = ShiftPanel + BitsPanel;
    static constexpr const BitPackType ShiftPixelSubRowIsSet = ShiftRich + BitsRich;
    static constexpr const BitPackType ShiftPixelColIsSet = ShiftPixelSubRowIsSet + BitsPixelSubRowIsSet;
    static constexpr const BitPackType ShiftPixelRowIsSet = ShiftPixelColIsSet + BitsPixelColIsSet;
    static constexpr const BitPackType ShiftPDIsSet = ShiftPixelRowIsSet + BitsPixelRowIsSet;
    static constexpr const BitPackType ShiftPanelIsSet = ShiftPDIsSet + BitsPDIsSet;
    static constexpr const BitPackType ShiftRichIsSet = ShiftPanelIsSet + BitsPanelIsSet;
    static constexpr const BitPackType ShiftLargePixel = ShiftRichIsSet + BitsRichIsSet;

    // The masks
    static constexpr const BitPackType MaskPixelCol = (BitPackType)((BitPackType {1} << BitsPixelCol) - BitPackType {1})
                                                      << ShiftPixelCol;
    static constexpr const BitPackType MaskPixelRow = (BitPackType)((BitPackType {1} << BitsPixelRow) - BitPackType {1})
                                                      << ShiftPixelRow;
    static constexpr const BitPackType MaskPDNumInMod =
      (BitPackType)((BitPackType {1} << BitsPDNumInMod) - BitPackType {1}) << ShiftPDNumInMod;
    static constexpr const BitPackType MaskPDMod = (BitPackType)((BitPackType {1} << BitsPDMod) - BitPackType {1})
                                                   << ShiftPDMod;
    static constexpr const BitPackType MaskPanel = (BitPackType)((BitPackType {1} << BitsPanel) - BitPackType {1})
                                                   << ShiftPanel;
    static constexpr const BitPackType MaskRich = (BitPackType)((BitPackType {1} << BitsRich) - BitPackType {1})
                                                  << ShiftRich;
    static constexpr const BitPackType MaskPixelSubRowIsSet =
      (BitPackType)((BitPackType {1} << BitsPixelSubRowIsSet) - BitPackType {1}) << ShiftPixelSubRowIsSet;
    static constexpr const BitPackType MaskPixelColIsSet =
      (BitPackType)((BitPackType {1} << BitsPixelColIsSet) - BitPackType {1}) << ShiftPixelColIsSet;
    static constexpr const BitPackType MaskPixelRowIsSet =
      (BitPackType)((BitPackType {1} << BitsPixelRowIsSet) - BitPackType {1}) << ShiftPixelRowIsSet;
    static constexpr const BitPackType MaskPDIsSet = (BitPackType)((BitPackType {1} << BitsPDIsSet) - BitPackType {1})
                                                     << ShiftPDIsSet;
    static constexpr const BitPackType MaskPanelIsSet =
      (BitPackType)((BitPackType {1} << BitsPanelIsSet) - BitPackType {1}) << ShiftPanelIsSet;
    static constexpr const BitPackType MaskRichIsSet =
      (BitPackType)((BitPackType {1} << BitsRichIsSet) - BitPackType {1}) << ShiftRichIsSet;
    static constexpr const BitPackType MaskLargePixel =
      (BitPackType)((BitPackType {1} << BitsLargePixel) - BitPackType {1}) << ShiftLargePixel;

    // Max values
    static constexpr const DataType MaxPixelCol = (DataType)(BitPackType {1} << BitsPixelCol) - DataType {1};
    static constexpr const DataType MaxPixelRow = (DataType)(BitPackType {1} << BitsPixelRow) - DataType {1};
    static constexpr const DataType MaxPDNumInMod = (DataType)(BitPackType {1} << BitsPDNumInMod) - DataType {1};
    static constexpr const DataType MaxPDMod = (DataType)(BitPackType {1} << BitsPDMod) - DataType {1};
    static constexpr const DataType MaxPanel = (DataType)(BitPackType {1} << BitsPanel) - DataType {1};
    static constexpr const DataType MaxRich = (DataType)(BitPackType {1} << BitsRich) - DataType {1};

    // Number of bits for the channel identification (i.e. excluding any time info)
    // Currently use the lowest 32 bits for this.
    static constexpr const BitPackType NChannelBits = 32;

    __host__ __device__ constexpr inline void
    setData(const DataType value, const BitPackType shift, const BitPackType mask) noexcept
    {
      m_key = ((BitPackType {value} << shift) & mask) | (m_key & ~mask);
    }

    __host__ __device__ constexpr inline void
    setData(const DataType value, const BitPackType shift, const BitPackType mask, const BitPackType okMask) noexcept
    {
      m_key = ((BitPackType {value} << shift) & mask) | (m_key & ~mask) | okMask;
    }

    __host__ __device__ constexpr inline BitPackType getData(const BitPackType shift, const BitPackType mask) const
      noexcept
    {
      return (m_key & mask) >> shift;
    }

    __host__ __device__ constexpr inline BitPackType key() const noexcept { return m_key; }

    __host__ __device__ constexpr inline bool operator==(const SmartID& other) const noexcept
    {
      return m_key == other.key();
    }

    __host__ __device__ constexpr inline auto isLargePMT() const noexcept
    {
      return 0 != getData(ShiftLargePixel, MaskLargePixel);
    }

    __host__ __device__ constexpr inline auto rich() const noexcept { return getData(ShiftRich, MaskRich); }

    __host__ __device__ constexpr inline auto panel() const noexcept { return getData(ShiftPanel, MaskPanel); }

    __host__ __device__ constexpr inline auto side() const noexcept { return panel(); }

    __host__ __device__ constexpr inline DataType pdMod() const noexcept { return getData(ShiftPDMod, MaskPDMod); }

    __host__ __device__ constexpr inline DataType pdNumInMod() const noexcept
    {
      return getData(ShiftPDNumInMod, MaskPDNumInMod);
    }

    __host__ __device__ constexpr inline DataType panelLocalModuleNum() const noexcept
    {
      return pdMod() - PanelModuleOffsets()[rich()][panel()];
    }

    __host__ __device__ constexpr inline DataType columnLocalModuleNum() const noexcept
    {
      return panelLocalModuleNum() % ModulesPerColumn;
    }

    __host__ __device__ constexpr inline DataType numPMTsPerEC() const noexcept
    {
      return isLargePMT() ? HTypePMTsPerEC : RTypePMTsPerEC;
    }

    __host__ __device__ constexpr inline DataType elementaryCell() const noexcept
    {
      return pdNumInMod() / numPMTsPerEC();
    }

    __host__ __device__ constexpr inline DataType pdNumInEC() const noexcept { return pdNumInMod() % numPMTsPerEC(); }

    __host__ __device__ constexpr inline auto pixelColIsSet() const noexcept
    {
      return 0 != getData(ShiftPixelColIsSet, MaskPixelColIsSet);
    }

    __host__ __device__ constexpr inline auto pixelRowIsSet() const noexcept
    {
      return 0 != getData(ShiftPixelRowIsSet, MaskPixelRowIsSet);
    }

    [[nodiscard]] __host__ __device__ constexpr bool pdIsSet() const noexcept
    {
      return 0 != ((key() & MaskPDIsSet) >> ShiftPDIsSet);
    }

    __host__ __device__ constexpr inline DataType pixelCol() const noexcept
    {
      return getData(ShiftPixelCol, MaskPixelCol);
    }

    __host__ __device__ constexpr inline DataType pixelRow() const noexcept
    {
      return getData(ShiftPixelRow, MaskPixelRow);
    }

    __host__ __device__ constexpr inline DataType anodeIndex() const noexcept
    {
      return pixelRow() * PixelsPerCol + PixelsPerRow - 1 - pixelCol();
    }

    __host__ __device__ constexpr inline bool adcTimeIsSet() const noexcept
    {
      return (0 != (key() & MaPMT::MaskADCTimeIsSet) >> MaPMT::ShiftADCTimeIsSet);
    }

    __host__ __device__ constexpr ADCTimeType adcTime() const
    {
      return ((key() & MaPMT::MaskADCTime) >> MaPMT::ShiftADCTime);
    }

    __host__ __device__ constexpr auto time() const noexcept
    {
      return (MaPMT::MinTime + (adcTime() * MaPMT::ScaleADCToTime));
    }

    // ostream operator
    __host__ friend std::ostream& operator<<(std::ostream& str, const SmartID& id)
    {
      const auto rich = id.rich();
      const auto panel = id.panel();
      str << "{ ";
      str << "PMT";
      str << (id.isLargePMT() ? ":h " : ":r ");
      str << (rich == 0 ? "Rich1 " : "Rich2 ");
      if (rich == 0) {
        str << (panel == 0 ? "Top " : "Bot ");
      }
      else {
        str << (panel == 0 ? "L-A " : "R-C ");
      }
      str << "PD[Mod,NInMod]: ";
      str << std::setfill('0') << std::setw(3) << id.pdMod() << ',';
      str << std::setfill('0') << std::setw(2) << id.pdNumInMod() << ' ';
      str << "Mod[Col,NInCol] ";
      str << std::setfill('0') << std::setw(2) << id.panelLocalModuleNum() << ',';
      str << std::setfill('0') << std::setw(2) << id.columnLocalModuleNum() << ' ';
      str << " PD[EC,NInEC]: ";
      str << id.elementaryCell() << ',';
      str << id.pdNumInEC() << ' ';

      const auto pixColSet = id.pixelColIsSet();
      const auto pixRowSet = id.pixelRowIsSet();
      if (pixColSet || pixRowSet) {
        if (pixColSet && pixRowSet) {
          str << "Pix[Col,Row]: " << std::setfill('0') << std::setw(1) << id.pixelCol() << "," << id.pixelRow();
        }
        else {
          const auto fSPix = 2;
          if (pixColSet) {
            str << std::setfill('0') << std::setw(fSPix) << " pixCol" << id.pixelCol();
          }
          if (pixRowSet) {
            str << std::setfill('0') << std::setw(fSPix) << " pixRow" << id.pixelRow();
          }
        }
        // Include PMT derived info
        if (pixColSet && pixRowSet) {
          str << " Anode:" << std::setfill('0') << std::setw(2) << id.anodeIndex();
        }
      }
      str << " }\n";
      return str;
    }

  public:
    // Number of PMT pixels per row
    static constexpr const DataType PixelsPerRow = 8;
    // Number of PMT pixels per column
    static constexpr const DataType PixelsPerCol = 8;
    // Total number of PMT pixels
    static constexpr const DataType TotalPixels = PixelsPerRow * PixelsPerCol;
    // Number PMTs per EC for R type PMTs
    static constexpr const DataType RTypePMTsPerEC = 4;
    // Number PMTs per EC for H type PMTs
    static constexpr const DataType HTypePMTsPerEC = 1;
    // Number of ECs per module
    static constexpr const DataType ECsPerModule = 4;
    // Number of modules per column
    static constexpr const DataType ModulesPerColumn = 6;
    // Number of module columns per panel, in each RICH
    static constexpr const std::array<DataType, 2> ModuleColumnsPerPanel = {
#ifdef USE_DD4HEP
      {11, 14} // With dd4hep we have an extra column reserved at the end of each RICH2 panel
#else
      {11, 12}
#endif
    };
    // Maximum number of module columns in any panel, RICH1 or RICH2
    static constexpr const DataType MaxModuleColumnsAnyPanel =
      std::max(ModuleColumnsPerPanel[0], ModuleColumnsPerPanel[1]);
    // Number of modules per panel, in each RICH
    static constexpr const std::array<DataType, 2> ModulesPerPanel {
      {(ModulesPerColumn * ModuleColumnsPerPanel[0]), (ModulesPerColumn * ModuleColumnsPerPanel[1])}};

    /// Number of modules in RICH1
    static constexpr const DataType RICH1Modules = 2 * ModulesPerPanel[Rich::Detector::Type::Rich1];
    /// Number of modules in RICH2
    static constexpr const DataType RICH2Modules = 2 * ModulesPerPanel[Rich::Detector::Type::Rich2];
    /// Total number of modules
    static constexpr const DataType TotalModules = RICH1Modules + RICH2Modules;

    __host__ __device__ static constexpr std::array<std::array<DataType, 2>, 2> PanelModuleOffsets()
    {
      return {std::array<DataType, 2> {0, ModulesPerPanel[0]},
              std::array<DataType, 2> {2 * ModulesPerPanel[0], (2 * ModulesPerPanel[0]) + ModulesPerPanel[1]}};
    }

    class MaPMT {
    public:
      static constexpr const DataType MaxPDsPerModule = 16;
      static constexpr const DataType MaxModulesPerPanel = 92;

      // Bits for time field.
      static constexpr const BitPackType BitsADCTime = 16;
      static constexpr const BitPackType BitsADCTimeIsSet = 1;
      static constexpr const BitPackType ShiftADCTime = NChannelBits;
      static constexpr const BitPackType ShiftADCTimeIsSet = ShiftADCTime + BitsADCTime;
      static constexpr const BitPackType MaskADCTime = (BitPackType)((BitPackType {1} << BitsADCTime) - BitPackType {1})
                                                       << ShiftADCTime;
      static constexpr const BitPackType MaskADCTimeIsSet =
        (BitPackType)((BitPackType {1} << BitsADCTimeIsSet) - BitPackType {1}) << ShiftADCTimeIsSet;

      // Max ADC time that can be stored
      static constexpr const ADCTimeType MaxADCTime = static_cast<ADCTimeType>((BitPackType {1} << BitsADCTime) - 1);

      // Parameters for conversion between float and ADC time values
      static constexpr const float MinTime = -50.0f; // In nanoseconds
      static constexpr const float MaxTime = 150.0f; // In nanoseconds
      static constexpr const float ScaleTimeToADC = MaxADCTime / (MaxTime - MinTime);
      static constexpr const float ScaleADCToTime = 1.0f / ScaleTimeToADC;
    };
  }; // namespace SmartID
} // namespace Allen::Rich::Decoding
