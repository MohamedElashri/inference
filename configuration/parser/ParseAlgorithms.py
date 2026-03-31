#!/usr/bin/python3
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

from AllenPythonGenerator import AllenPythonGenerator
from AllenExternLinesGenerator import AllenExternLinesGenerator
from AllenGaudiWrapperGenerator import AllenGaudiWrapperGenerator
from AllenAlgorithmFinder import AllenAlgorithmFinder
from AllenGeneratorUtils import get_namespace
import argparse
import json
import sys


class AllenCore():
    @staticmethod
    def get_default_properties(default_properties_cmd, prefix_project_folder):
        from subprocess import (PIPE, run)

        # Run the default_properties executable to get a JSON
        # representation of the default values of all properties of
        # all algorithms
        p = run([default_properties_cmd], stdout=PIPE, encoding='ascii')

        default_properties = None
        if p.returncode == 0:
            default_properties = json.loads(p.stdout)
        else:
            print("Failed to obtain default property values")
            sys.exit(-1)

        # Patch filenames
        all_filenames = AllenAlgorithmFinder.get_all_includes(
            prefix_project_folder)
        for _, alg in default_properties.items():
            namespace, name = get_namespace(alg["type"])
            alg['filename'] = AllenAlgorithmFinder.find_filename_for_algorithm(
                name, all_filenames)

        return default_properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Parse the Allen codebase and generate a python representation of all algorithms.'
    )

    parser.add_argument(
        '--filename',
        nargs='?',
        type=str,
        default="algorithms.py",
        help='output filename')
    parser.add_argument(
        '--prefix_project_folder',
        nargs='?',
        type=str,
        default="..",
        help='project location')
    parser.add_argument(
        "--algorithm_wrappers_folder",
        nargs="?",
        type=str,
        default="",
        help="converted algorithms folder")
    parser.add_argument(
        "--default_properties",
        nargs="?",
        type=str,
        default="",
        help="location of default_properties executable")
    parser.add_argument(
        "--generate",
        nargs="?",
        type=str,
        default="views",
        choices=[
            "views", "wrapperlist", "wrappers", "extern_lines",
            "extern_lines_nosepcomp"
        ],
        help="action that will be performed")

    args = parser.parse_args()
    prefix_folder = args.prefix_project_folder + "/"

    if args.generate in ["views", "wrappers"]:
        default_properties = AllenCore.get_default_properties(
            args.default_properties, prefix_folder)
        with open("default_properties.json", "w") as f:
            f.write(json.dumps(default_properties, indent=2))

    if args.generate == "views":
        # Generate algorithm python views
        AllenPythonGenerator.write_algorithms_view(default_properties,
                                                   args.filename)
    elif args.generate == "wrapperlist":
        # Write algorithm list in txt format for CMake
        all_algorithms = AllenAlgorithmFinder.find_all_algorithm_instances(
            prefix_folder)
        AllenGaudiWrapperGenerator.write_algorithm_filename_list(
            all_algorithms, args.algorithm_wrappers_folder, args.filename)
    elif args.generate == "wrappers":
        # Write Gaudi wrappers on top of all algorithms
        AllenGaudiWrapperGenerator.write_gaudi_algorithms(
            default_properties, args.algorithm_wrappers_folder)
    elif args.generate == "extern_lines":
        # Write extern lines header file
        all_lines = AllenAlgorithmFinder.find_all_line_instances(prefix_folder)
        AllenExternLinesGenerator.write_extern_lines(all_lines, args.filename,
                                                     True)
    elif args.generate == "extern_lines_nosepcomp":
        all_lines = AllenAlgorithmFinder.find_all_line_instances(prefix_folder)
        # Write extern lines header file, without separable compilation
        AllenExternLinesGenerator.write_extern_lines(all_lines, args.filename,
                                                     False)
