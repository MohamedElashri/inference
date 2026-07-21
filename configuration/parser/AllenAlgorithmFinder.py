###############################################################################
# (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

import codecs
import os
import re

from AllenGeneratorUtils import get_namespace


class AllenAlgorithmFinder:
    """Helper class to find the definition files of Allen algorithms."""

    # File extensions considered
    __source_extensions_compiled = [
        re.compile(".*\\." + p + "$") for p in ["cpp", "cu"]
    ]

    # Folders storing device and host code
    __folders = ["device", "host"]

    @staticmethod
    def __get_filenames(folder, extensions):
        list_of_files = []
        for root, subdirs, files in os.walk(folder):
            for filename in files:
                for extension in extensions:
                    if extension.match(filename):
                        list_of_files.append(os.path.join(root, filename))
        return list_of_files

    @staticmethod
    def get_all_sources(prefix_project_folder):
        out = []
        for folder in AllenAlgorithmFinder.__folders:
            path = prefix_project_folder + folder
            out += AllenAlgorithmFinder.__get_filenames(
                path, AllenAlgorithmFinder.__source_extensions_compiled
            )
        return out

    @staticmethod
    def find_all_line_instances(prefix_project_folder):
        all_filenames = AllenAlgorithmFinder.get_all_sources(prefix_project_folder)
        lines = []
        instance_re = re.compile(r"INSTANTIATE_LINE\(([^,]+),([^\)]+)\)")
        for filename in all_filenames:
            with codecs.open(filename, "r", "utf-8") as f:
                s = f.read()
                for m in instance_re.finditer(s):
                    lines.append(get_namespace(m.group(1).strip()))
        return lines
