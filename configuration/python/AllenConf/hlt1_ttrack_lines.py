###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenCore.algorithms import (ttrack_dimuon_displaced_t,
                                  ttrack_lambda2ppi_t, ttrack_ks2pipi_t)
from AllenConf.utils import initialize_number_of_events, mep_layout
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable
from AllenCore.configuration_options import is_allen_standalone


@configurable
def make_ttrack_highmass_dimuon_displaced_line(
        ttrack_vertices,
        name="Hlt1TTrackHighMassDiMuonDisplaced",
        pre_scaler=1.,
        post_scaler=1.,
        pre_scaler_hash_string=None,
        post_scaler_hash_string=None,
        enable_tupling=False,
        enable_monitoring=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_dimuon_displaced_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'],
        min_pt=3413.6,
        min_track_p=8522.3,
        min_r2=150. * 150.,
        min_m=1500.,
        max_doca=25.,
        max_ovtx_z=8500.,
        max_ip2=100. * 100.,
        min_dira=0.998,
        opposite_sign=True)


@configurable
def make_ttrack_lowmass_dimuon_displaced_line(
        ttrack_vertices,
        name="Hlt1TTrackLowMassDiMuonDisplaced",
        pre_scaler=1.,
        post_scaler=1.,
        pre_scaler_hash_string=None,
        post_scaler_hash_string=None,
        enable_tupling=False,
        enable_monitoring=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_dimuon_displaced_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'],
        min_pt=2420.,
        min_max_track_p=8520.,
        min_r2=250. * 250.,
        max_m=1500.,
        max_doca=25.,
        max_ovtx_z=8500.,
        max_ip2=100. * 100.,
        min_dira=0.998,
        opposite_sign=True)


@configurable
def make_ttrack_highmass_dimuon_displaced_samesign_line(
        ttrack_vertices,
        name="Hlt1TTrackHighMassDiMuonDisplacedSameSign",
        pre_scaler=1. / 50.,
        post_scaler=1.,
        pre_scaler_hash_string=None,
        post_scaler_hash_string=None,
        enable_tupling=False,
        enable_monitoring=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_dimuon_displaced_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'],
        min_pt=3413.6,
        min_track_p=8522.3,
        min_r2=150. * 150.,
        min_m=1500.,
        max_doca=25.,
        max_ovtx_z=8500.,
        max_ip2=100. * 100.,
        min_dira=0.998,
        opposite_sign=False)


@configurable
def make_ttrack_lowmass_dimuon_displaced_samesign_line(
        ttrack_vertices,
        name="Hlt1TTrackLowMassDiMuonDisplacedSameSign",
        pre_scaler=1. / 50.,
        post_scaler=1.,
        pre_scaler_hash_string=None,
        post_scaler_hash_string=None,
        enable_tupling=False,
        enable_monitoring=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_dimuon_displaced_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'],
        min_pt=2420.,
        min_max_track_p=8520.,
        min_r2=250. * 250.,
        max_m=1500.,
        max_doca=25.,
        max_ovtx_z=8500.,
        max_ip2=100. * 100.,
        min_dira=0.998,
        opposite_sign=False)


@configurable
def make_ttrack_lambda2ppi_line(ttrack_vertices,
                                name="Hlt1TTrackLambda2PPi",
                                pre_scaler=1. / 580.,
                                post_scaler=1.,
                                pre_scaler_hash_string=None,
                                post_scaler_hash_string=None,
                                enable_tupling=False,
                                enable_monitoring=True):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_lambda2ppi_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'])


@configurable
def make_ttrack_ks2pipi_line(ttrack_vertices,
                             name="Hlt1TTrackKs2PiPi",
                             pre_scaler=1. / 160.,
                             post_scaler=1.,
                             pre_scaler_hash_string=None,
                             post_scaler_hash_string=None,
                             enable_tupling=False,
                             enable_monitoring=True):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        ttrack_ks2pipi_t,
        name=name,
        enable_tupling=enable_tupling,
        enable_monitoring=enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        pre_scaler=pre_scaler,
        post_scaler=post_scaler,
        host_number_of_svs_t=ttrack_vertices['host_number_of_tt_vertices'],
        dev_particle_container_t=ttrack_vertices[
            'dev_multi_event_composites_view'])
