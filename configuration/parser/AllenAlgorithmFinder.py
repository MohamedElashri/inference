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

import re, os, sys, codecs
from AllenGeneratorUtils import get_namespace


class AllenAlgorithmFinder():
    """Helper class to find the definition files of Allen algorithms."""

    # Pattern sought in every file, prior to parsing the file for an algorithm
    __algorithm_pattern_compiled = re.compile(
        "(?P<scope>Host|Device|Selection|Validation|Provider|Barrier)Algorithm"
    )

    # File extensions considered
    __include_extensions_compiled = [
        re.compile(".*\\." + p + "$") for p in ["cuh", "h", "hpp"]
    ]
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
    def get_all_includes(prefix_project_folder):
        out = []
        for folder in AllenAlgorithmFinder.__folders:
            path = prefix_project_folder + folder
            out += AllenAlgorithmFinder.__get_filenames(
                path, AllenAlgorithmFinder.__include_extensions_compiled)
        return out

    @staticmethod
    def get_all_sources(prefix_project_folder):
        out = []
        for folder in AllenAlgorithmFinder.__folders:
            path = prefix_project_folder + folder
            out += AllenAlgorithmFinder.__get_filenames(
                path, AllenAlgorithmFinder.__source_extensions_compiled)
        return out

    @staticmethod
    def find_all_algorithm_instances(prefix_project_folder):
        all_filenames = AllenAlgorithmFinder.get_all_sources(
            prefix_project_folder)
        algorithms = []
        instance_re = re.compile(r"INSTANTIATE_ALGORITHM\(([^)]+)\)")
        instance_with_id_re = re.compile(
            r'INSTANTIATE_ALGORITHM_WITH_ID\([^,]+,\s*"([^"]+)"\)')
        for filename in all_filenames:
            with codecs.open(filename, 'r', 'utf-8') as f:
                s = f.read()
                for m in instance_re.finditer(s):
                    _, name = get_namespace(m.group(1).strip())
                    algorithms.append(name)
                for m in instance_with_id_re.finditer(s):
                    algorithms.append(m.group(1).strip())
        all_lines = AllenAlgorithmFinder.find_all_line_instances(
            prefix_project_folder)
        return algorithms + [name for namespace, name in all_lines]

    @staticmethod
    def find_all_line_instances(prefix_project_folder):
        all_filenames = AllenAlgorithmFinder.get_all_sources(
            prefix_project_folder)
        lines = []
        instance_re = re.compile(r'INSTANTIATE_LINE\(([^,]+),([^\)]+)\)')
        for filename in all_filenames:
            with codecs.open(filename, 'r', 'utf-8') as f:
                s = f.read()
                for m in instance_re.finditer(s):
                    lines.append(get_namespace(m.group(1).strip()))
        return lines

    @staticmethod
    def find_algorithm_files(prefix_project_folder):
        all_filenames = AllenAlgorithmFinder.get_all_includes(
            prefix_project_folder)
        algorithm_files = []
        for filename in all_filenames:
            with codecs.open(filename, 'r', 'utf-8') as f:
                s = f.read()
                has_algorithm = AllenAlgorithmFinder.__algorithm_pattern_compiled.search(
                    s)
                if has_algorithm:
                    algorithm_files.append(filename)
        return algorithm_files

    @staticmethod
    def find_filename_for_algorithm(alg_name, all_filenames):
        struct_re = re.compile(fr"struct\s+{alg_name}\s+:")
        for filename in all_filenames:
            with codecs.open(filename, 'r', 'utf-8') as f:
                s = f.read()
                if struct_re.search(s): return filename
        raise Exception("Could not find filename for " + alg_name)
        return ""
