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
#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"
#include "ParticleTypes.cuh"
#include "PVFinderConstants.cuh"
#include "PVFinderModel.h"

#include <atomic>

namespace pvfinder_fc_aggregation {

  // The latent channels (FC outputs per bin, UNet input channels): the same
  // build setting as pvfinder_unet's N_BATCH_CHANNELS, so the two always agree.
  // Default: the default model's 4 (see CMakeLists.txt).
#ifdef PVFINDER_UNET_N_BATCH_CHANNELS
  constexpr unsigned N_LATENT_CHANNELS = PVFINDER_UNET_N_BATCH_CHANNELS;
#else
  constexpr unsigned N_LATENT_CHANNELS = 4u;
#endif
  constexpr unsigned N_BINS_PER_CHANNEL = PVFinderConstants::KDE::n_bins_per_interval;
  constexpr unsigned N_INTERVALS = PVFinderConstants::KDE::n_intervals;
  // KDE bins per event (dev_pvfinder_output_histogram stride).
  constexpr unsigned KDE_BINS = PVFinderConstants::KDE::n_bins;
  // dev_pvfinder_interval_start per event: the start of each interval's entries,
  // the end of the last one, and a copy of it at CSR_TOTAL (entries in the event).
  constexpr unsigned CSR_STRIDE = N_INTERVALS + 2u;
  constexpr unsigned CSR_TOTAL = N_INTERVALS + 1u;

  // One work item of the FC kernels, from the work list the host builds each
  // slice (largest first): a slot (event, interval), or for the BF16 kernel one
  // chunk of a slot with many entries.
  // Every field is full width, so any event size fits.
  struct alignas(16) FCWorkItem {
    unsigned slot;     // event * N_INTERVALS + interval
    unsigned first;    // first CSR entry of the item, relative to the event
    unsigned entries;  // CSR entries in the item
    int row;           // feature row, -1 when the slot has none
    unsigned partial;  // split slots: the slot's first partial-sum row
    unsigned chunk;    // split slots: this item's chunk
    unsigned n_chunks; // chunks of the slot, 1 when it is not split
    unsigned unused;
  };
  constexpr unsigned FC_WORK_ITEM_WORDS = sizeof(FCWorkItem) / sizeof(unsigned);
  // L6A's physical width: one neuron per (channel, bin) pair. 800 by default.
  constexpr unsigned L6A_WIDTH = N_LATENT_CHANNELS * N_BINS_PER_CHANNEL;
  // Layer6A's weight matrix is [L6A_WIDTH x 20]; 16000 floats by default.
  constexpr unsigned L6A_WEIGHT_FLOATS = L6A_WIDTH * 20u;
  // dev_pvfinder_interval_features's per-event stride (40 intervals x
  // L6A_WIDTH); 32000 floats by default.
  constexpr unsigned INTERVAL_FEATURES_STRIDE = N_INTERVALS * L6A_WIDTH;

  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    // Events that passed the node's prefilters (the HLT1 physics prefilters in
    // the PVFinder sequences). Only these get tracks; every other event's
    // intervals are empty.
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;

    // Per-track input features [tracks x 9] (PVFinderTrackFeatures.cuh).
    DEVICE_OUTPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;
    // Track-to-interval index (CSR), per event:
    //   interval_start[event * CSR_STRIDE + i]: first entry of interval i,
    //     [N_INTERVALS] the end, [CSR_TOTAL] the number of entries;
    //   track_idx[2 * first track of the event + entry]: local track index,
    //     each interval's tracks in canonical order (a track feeds one or two
    //     intervals, hence twice the tracks).
    DEVICE_OUTPUT(dev_pvfinder_interval_start_t, int) dev_pvfinder_interval_start;
    DEVICE_OUTPUT(dev_pvfinder_track_idx_t, int) dev_pvfinder_track_idx;
    // Scatter order of the entries of events too large for the CSR build's
    // shared memory, before they are put in canonical order.
    DEVICE_OUTPUT(dev_pvfinder_track_idx_unsorted_t, int) dev_pvfinder_track_idx_unsorted;
    // Host copy of dev_pvfinder_interval_start, read back once per slice to
    // build the work list and the UNet's rows.
    HOST_OUTPUT(host_pvfinder_interval_start_t, int) host_pvfinder_interval_start;

    // The UNet's input: one row of L6A_WIDTH features (N_LATENT_CHANNELS
    // channels x 100 bins) per interval with tracks, in (event, interval)
    // order, padded with zero rows to a multiple of unet_batch_events * 40.
    // float32 [channel][bin] (precision = float32) or bfloat16 [bin][channel]
    // (precision = bfloat16). Intervals without tracks have no row: the UNet
    // writes its response to an all-zero input there.
    DEVICE_OUTPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
    // Row layout for the UNet:
    //   host_pvfinder_unet_rows[0]: 1 (host-built) or 2 (GPU-built compact rows)
    //   host_pvfinder_unet_rows[1]: rows in use, or capacity for GPU-built rows
    //   host_pvfinder_unet_rows[2]: 1 when the features are bfloat16
    //   host_pvfinder_unet_rows[3]: 1 when rows are channels last ([bin][channel])
    //   dev_pvfinder_slot_row[event * 40 + interval]: the interval's row, -1 if none
    //   GPU-built rows append three words: actual rows, work items and partial rows
    //   dev_pvfinder_row_slot[row]: its inverse, event * 40 + interval
    HOST_OUTPUT(host_pvfinder_unet_rows_t, unsigned) host_pvfinder_unet_rows;
    DEVICE_OUTPUT(dev_pvfinder_slot_row_t, int) dev_pvfinder_slot_row;
    HOST_OUTPUT(host_pvfinder_slot_row_t, int) host_pvfinder_slot_row;
    DEVICE_OUTPUT(dev_pvfinder_row_slot_t, int) dev_pvfinder_row_slot;

    // Work list of the FC kernels (FCWorkItem, FC_WORK_ITEM_WORDS words each),
    // its work counter, and for slots split over several warps (BF16 kernel)
    // their partial sums [chunks][L6A_WIDTH] and chunks done per slot.
    DEVICE_OUTPUT(dev_pvfinder_slot_order_t, unsigned) dev_pvfinder_slot_order;
    HOST_OUTPUT(host_pvfinder_slot_order_t, unsigned) host_pvfinder_slot_order;
    DEVICE_OUTPUT(dev_pvfinder_work_counter_t, unsigned) dev_pvfinder_work_counter;
    DEVICE_OUTPUT(dev_pvfinder_fc_partial_t, float) dev_pvfinder_fc_partial;
    DEVICE_OUTPUT(dev_pvfinder_fc_arrive_t, unsigned) dev_pvfinder_fc_arrive;

    // The FC's own KDE estimate [events x 4000] (softplus of the channel sum,
    // divided by the interval's tracks), read by the validation only: written
    // when dump_validation is set, empty otherwise.
    DEVICE_OUTPUT(dev_pvfinder_output_histogram_t, float) dev_pvfinder_output_histogram;
  };

  struct pvfinder_fc_aggregation_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context) const;

    // Loads the beamline into dev_beamline (the track features are in its frame).
    void update(const Constants& constants) const;

    // Copies the weights to the device and sizes the FC kernel's grid.
    void init();

  private:
    // Block of the CSR build (one per event). 512 is fastest on the RTX 3090:
    // the busiest events' canonical ranking is the kernel's tail, and more
    // threads shorten it (256: 6.0 ms, 512: 5.0, 1024: 7.3 per 100 slices).
    Allen::Property<dim3> m_block_dim {this, "block_dim", {512, 1, 1}, "block dimensions"};

    // The trained model (PVFinderModel.h): relative to the parameters
    // directory (--params), or absolute. Must be pvfinder_unet's (AllenConf
    // sets both).
    Allen::Property<std::string> m_model_file {
      this,
      "model",
      "pvfinder/unet16_lc4_scnone_asym5_final.json",
      "PVFinder model file, relative to the parameters directory or absolute; must match pvfinder_unet.model"};
    PVFinder::Model m_model {"pvfinder_fc_aggregation", [this] { return m_model_file.value(); }};

    // "float32": FP32 throughout, features stored float32 [channel][bin], for
    // pvfinder_unet with precision = float32; exact. "bfloat16": layers 2-5 and
    // L6A on BF16 tensor cores with FP32 accumulation (layer 1, bias, LeakyReLU
    // and the sums over tracks stay FP32), features stored bfloat16
    // [bin][channel], for pvfinder_unet with precision = bfloat16; needs
    // compute capability 8.0 or newer. Must match pvfinder_unet's precision
    // (AllenConf sets both).
    Allen::Property<std::string> m_precision {
      this,
      "precision",
      "float32",
      "float32 or bfloat16; must match pvfinder_unet.precision"};

    // The UNet reads whole batches of this many events' rows (unet_batch_events
    // * 40): the features are padded to a multiple of it. Must match
    // pvfinder_unet.unet_batch_events (AllenConf sets both).
    Allen::Property<unsigned> m_unet_batch_events {
      this,
      "unet_batch_events",
      20u,
      "pad the UNet rows to a multiple of this many events' rows"};

    // Fraction of the full-occupancy grid the BF16 kernel is launched with. Its
    // blocks take most of an SM's shared memory; below 1, other streams'
    // kernels keep SMs while it runs. At 16 streams on the RTX 3090 (FC + UNet,
    // both set alike): 1: 6.3% loss, 1/2: 4.9%, 1/4: 4.3%, 1/8: 4.0%, 1/16: 3.9%.
    Allen::Property<float> m_fused_grid_fraction {
      this,
      "fused_grid_fraction",
      0.125f,
      "fraction of the full-occupancy grid for the BF16 FC kernel"};

    Allen::Property<bool> m_gpu_work_list {
      this, "gpu_work_list", false, "bfloat16: build compact rows and the chunked FC work list on the GPU"};

    // Validation dump: when non-empty, the first operator() call writes the
    // FC's inputs and outputs (CSR, per-event track offsets, track features,
    // interval features, histogram, track states, beamline) to this directory.
    Allen::Property<std::string> m_dump_dir {
      this,
      "dump_validation",
      "",
      "if non-empty, dump the FC inputs and outputs of the first slice to this directory"};
    mutable std::atomic<bool> m_dump_done {false};
    void dump(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context) const;

    // Set by init(): this instance's device weights (layers 1-6A, float32),
    // whether it runs the BF16 kernel, and the FC kernel's grid.
    const float* m_dev_weights = nullptr;
    bool m_bf16 = false;
    unsigned m_grid = 0;
  };

} // namespace pvfinder_fc_aggregation
