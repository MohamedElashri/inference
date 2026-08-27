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


class AllenExternLinesGenerator:
    """Generates extern declarations and dispatch code for lines"""

    @staticmethod
    def write_extern_lines(selection_algorithms, filename, separable_compilation):
        code = "\n".join(("#pragma once", "", '#include "BackendCommon.h"', "\n"))
        for namespace, name in selection_algorithms:
            code += "\n".join(
                (
                    f"namespace {namespace} {{",
                    f"  struct {name};",
                    "  struct Parameters;",
                    "}\n",
                )
            )
        code += "\n"
        if separable_compilation:
            for namespace, name in selection_algorithms:
                code += f"extern template __device__ void process_line<{namespace}::{name}, {namespace}::Parameters>(char*, uint32_t*, unsigned*, const LineData*, const ODINData*, const unsigned*, const unsigned, const unsigned, const unsigned, const unsigned);\n"
            code += "\n"
            for namespace, name in selection_algorithms:
                code += f"extern template void line_output_monitor<{namespace}::{name}, {namespace}::Parameters>(char*, const RuntimeOptions&, const Allen::Context&);\n"
        code += "\nconstexpr auto line_strings = {\n"
        for i, (namespace, name) in enumerate(selection_algorithms):
            code += f'  "{name}"'
            if i != len(selection_algorithms) - 1:
                code += ",\n"
        code += "\n};\n\n"
        code += "__device__ inline void invoke_line_functions(unsigned index, char* a, uint32_t* b, unsigned* c, const LineData* d, const ODINData* e, const unsigned* f, const unsigned g, const unsigned h, const unsigned i, const unsigned j) {\n"
        code += f"  assert(index < {len(selection_algorithms)});\n"
        code += "  switch (index) {\n"
        for i, (namespace, name) in enumerate(selection_algorithms):
            code += f"    case {i}: process_line<{namespace}::{name}, {namespace}::Parameters>(a, b, c, d, e, f, g, h, i, j); break;\n"
        code += "  }\n}\n\n"
        code += f"constexpr std::array<void(*)(char*, const RuntimeOptions&, const Allen::Context&), {len(selection_algorithms)}> line_output_monitor_functions = {{\n"
        for i, (namespace, name) in enumerate(selection_algorithms):
            code += (
                f"  line_output_monitor<{namespace}::{name}, {namespace}::Parameters>"
            )
            if i != len(selection_algorithms) - 1:
                code += ",\n"
        code += "\n};\n"
        # void inline invoke_output_monitor(const char* arg_ref, const RuntimeOptions& runtime_options, const Allen::Context& context) {
        with open(filename, "w") as f:
            f.write(code)
