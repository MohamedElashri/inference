###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import sys
import os
import time
import signal
from subprocess import PIPE, Popen
from threading import Thread
import requests
from dateutil.parser import *
import datetime
import socket as sock
import sys
import zmq
import re

socket_expr = re.compile(R'allen_throughput_[a-z0-9]{2}')
ON_POSIX = 'posix' in sys.builtin_module_names


def send(telegraf_string):
    telegraf_url = 'http://dcinflux01:8189/telegraf'
    session = requests.session()
    session.trust_env = False
    try:
        print('Sending telegraf string: %s' % telegraf_string)
        response = session.post(telegraf_url, data=telegraf_string)
        print('http response: %s' % response.headers)
    except:
        print('Failed to submit data string %s' % telegraf_string)
        print(traceback.format_exc())


def send_to_telegraf(rate, device, app_name, rate_name):

    now = datetime.datetime.now()
    timestamp = datetime.datetime.timestamp(now) * 1000000000

    telegraf_string = "AllenIntegrationTest,host=%s,%s=%s " % (
        sock.gethostname(), app_name, device)
    telegraf_string += "%s=%.2f " % (rate_name, float(rate))
    telegraf_string += " %d" % timestamp

    send(telegraf_string)


def main():
    # if len(sys.argv) < 2:
    #     print('usage: allen_throughput.py <connect_to> <connect_to...>')
    #     sys.exit(1)

    socket_ids = [s for s in os.listdir('/tmp') if socket_expr.match(s)]

    ctx = zmq.Context()
    sockets = {}
    connections = {
        "ipc:///tmp/%s": (socket_ids, 'AllenInstance', 'allen_rate'),
    }
    for connection, (ids, app_name, rate_name) in connections.items():
        for socket_id in ids:
            con = connection % socket_id
            app_id = socket_id
            print("connecting to: " + con)
            s = ctx.socket(zmq.SUB)
            s.connect(con)
            s.setsockopt(zmq.SUBSCRIBE, b'')
            sockets[s] = (app_name, rate_name, app_id)

    poller = zmq.Poller()
    for socket in sockets.keys():
        poller.register(socket, zmq.POLLIN)

    while True:
        polled = dict(poller.poll())
        for socket, (app_name, rate_name, socket_id) in sockets.items():
            if socket in polled and polled[socket] == zmq.POLLIN:
                message = socket.recv()
                print(socket_id, message)


if __name__ == "__main__":
    main()
