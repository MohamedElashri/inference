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

from AllenGeneratorUtils import get_namespace

class AllenGaudiWrapperGenerator:
    """Generates Gaudi wrapper C++ code"""
    
    @staticmethod
    def generate_gaudi_wrapper(algorithm):
        namespace, _ = get_namespace(algorithm["type"])
        _, name = get_namespace(algorithm["name"])

        properties = make_properties(algorithm["properties"])
        inputs, outputs, aggregates, parameters_non_aggregate = make_parameters(algorithm["parameters"])
        param_namespace = parameters_non_aggregate[0]['namespace'] # get the type of the Parameter struct (might be templated)

        input_types = make_param_vectors(inputs)
        output_types = make_param_vectors(outputs)
        aggregate_types = make_aggregate_types(aggregates)

        aggregate_handles = [
            f"std::vector<DataObjectReadHandle<Allen::parameter_vector<{typ}>>> m_{agg['name']};"
            for agg, typ in zip(aggregates, aggregate_types)
        ]
        aggregate_input_vectors = [
            "\n".join([
                f"Gaudi::Property<std::vector<DataObjID>> m_{agg['name']}_locations",
                f"{{this, \"{agg['name']}\", {{}},",
                f"  [=,this]( Gaudi::Details::PropertyBase& ) {{",
                f"    this->m_{agg['name']} =",
                f"      Gaudi::Functional::details::make_vector_of_handles<decltype( this->m_{agg['name']} )>( this, m_{agg['name']}_locations );",
                f"}},",
                f"Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}}};",
            ]) for agg in aggregates
        ]

        get_aggregates = [
            f"std::vector<{typ}, LHCb::Allocators::EventLocal<{typ}>> empty_vector_tes_wrappers_{agg['name']} {{ LHCb::getMemResource( evtCtx ) }};\n"
            +
            f"std::vector<Allen::TESWrapperInput<{typ}>> tes_wrappers_{agg['name']};\n"
            +
            f"tes_wrappers_{agg['name']}.reserve(m_{agg['name']}.size());\n"
            + f"for (auto const& h : m_{agg['name']}) {{\n" +
            f"  auto* inp = h.getIfExists(); \n" +
            f"  tes_wrappers_{agg['name']}.emplace_back(inp ? *inp : empty_vector_tes_wrappers_{agg['name']}, \"{agg['name']}\");\n"
            + f"}}\n" +
            f"std::vector<std::reference_wrapper<Allen::Store::BaseArgument>> arg_data_{agg['name']};\n"
            + f"arg_data_{agg['name']}.reserve(m_{agg['name']}.size());\n" +
            f"for (auto& w : tes_wrappers_{agg['name']}) {{\n" +
            f"  arg_data_{agg['name']}.emplace_back(w);\n" + f"}}\n"
            for agg, typ in zip(aggregates, aggregate_types)]

        aggregate_types = make_parameter_types(aggregates)

        arg_data_agg_typenames = [
            f"arg_data_{agg['name']}" for agg in aggregates
        ]

        input_handles = [
            f"DataObjectReadHandle<{typ}> m_{inp['name']} {{this, \"{inp['name']}\", \"\"}};"
            for inp, typ in zip(inputs, input_types)
        ] + [
            "DataObjectReadHandle<LHCb::ODIN> m_odin {this, \"ODIN\", \"\"};",
            "DataObjectReadHandle<RuntimeOptions> m_runtime_options {this, \"runtime_options_t\", \"\"};",
            "DataObjectReadHandle<Constants const*> m_constants {this, \"constants_t\", \"\"};",
        ]

        output_handles = [
            f"DataObjectWriteHandle<{typ}> m_{out['name']} {{this, \"{out['name']}\", \"\"}};"
            for out, typ in zip(outputs, output_types)
        ]

        tes_wrappers_list = []
        tes_wrappers_reference_initialization_list = []
        output_container_element = 0

        for i, p in enumerate(parameters_non_aggregate):
            # Fetch the type of the TES wrapper for parameter p
            # Produce the initialization line for parameter p into a TESWrapper
            if p in inputs:
                tes_wrapper_type_text = "TESWrapperInput"
                parameter_variable_name = p['name']
            else:
                tes_wrapper_type_text = "TESWrapperOutput"
                parameter_variable_name = f"std::get<{output_container_element}>(output_container)"
                output_container_element += 1

            tes_wrapper_initialization = f"{{{parameter_variable_name},\"{p['name']}\"}}"
            tes_wrapper_variable_name = f"{p['name']}_wrapper"

            tes_wrappers_list.append(
                f"Allen::{tes_wrapper_type_text}<{p['typename']}::type> {tes_wrapper_variable_name} {tes_wrapper_initialization};"
            )
            tes_wrappers_reference_initialization_list.append(
                f"{tes_wrapper_variable_name}")

        tes_wrappers = "\n".join(tes_wrappers_list)
        tes_wrappers_reference_initialization = ",".join(
            tes_wrappers_reference_initialization_list)
        tes_wrappers_reference = f"std::array<std::reference_wrapper<Allen::Store::BaseArgument>, {len(parameters_non_aggregate)}> tes_wrappers_references {{{tes_wrappers_reference_initialization}}};"

        output_alloc = make_output_alloc(outputs)

        is_filter = "mask_t" in [out['type'] for out in outputs]
        filter_decision = "FilterDecision::PASSED"
        if is_filter:
            index_of_mask_t = [out['type'] for out in outputs].index("mask_t")
            filter_decision = f"std::get<{index_of_mask_t}>(output_container).size() ? FilterDecision::PASSED : FilterDecision::FAILED"

        code = f"""#include "AlgorithmConversionTools.h"
#include <{algorithm['filename']}>
#include <Gaudi/Algorithm.h>
#include <GaudiAlg/FunctionalDetails.h>
#include <GaudiKernel/FunctionalFilterDecision.h>
#include <GaudiKernel/Environment.h>
#include <Event/ODIN.h>
#include <mutex>
#include <vector>
#include "AllenMonitoring.h"
#include "MVAModelsManager.h"

using namespace Gaudi::Functional;

class {name} final : public Gaudi::Algorithm {{
public:
  using Gaudi::Algorithm::Algorithm;

  StatusCode initialize() override {{
    const StatusCode sc = Algorithm::initialize();
    if ( sc.isFailure() ) return sc;
    Allen::initialize_algorithm(m_algorithm);
    m_algorithm.set_name(this->name());
    return sc;
  }}

  StatusCode start() override {{
    const StatusCode sc = Algorithm::start();
    if ( sc.isFailure() ) return sc;
    Allen::Monitoring::AccumulatorManager::get()->initAccumulators(1);
    Allen::MVAModels::MVAModelsManager::get()->loadData((m_cached_root + "/data").c_str());
    return sc;
  }}

  StatusCode stop() override {{
    Allen::Monitoring::AccumulatorManager::get()->mergeAndReset(true);
    const StatusCode sc = Algorithm::stop();
    if ( sc.isFailure() ) return sc;
    return sc;
  }}

private:
  {algorithm['type']} m_algorithm{{}};
  std::string                  m_cached_root;
  Gaudi::Property<std::string> m_root{{
    this, "Root", "${{PARAMFILESROOT}}",
    [this]( auto const& ) {{
      System::resolveEnv( m_root, m_cached_root ).orThrow( "ParamFileSvc", "Cannot resolve  " + m_root );
    }}, Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}}};
  mutable std::optional<unsigned> m_runNumber;
  mutable std::mutex m_mut;

  // Data handles:
  {"\n".join(input_handles + output_handles + aggregate_handles + aggregate_input_vectors)}

  // Properties:
  {"\n".join(properties)}

public:
  StatusCode execute( [[maybe_unused]] const EventContext& evtCtx ) const override {{
    // loop over inputs to get them:
    {"\n".join((f"auto const& {inp['name']} = *m_{inp['name']}.get();" for inp in inputs if 'optional' not in inp))}
    
    // optional inputs:
    // we need decltype(*{{inp.typename}}_ptr){{{{}},{{}}}} to initialize a vector with 2 elements (most often ints), such that
    // the typical access pattern of [event_number], [event_number + 1] for offsets works. Initializing explicitly with 0s fails
    // if more complicated types are to be initialized.
    // keep in mind that we only have single events in mind here, so that event_number == 0 is always true.
    // for gaudi this is a reasonable assumption as it only runs single events at a time.
    {"\n".join((f"auto const* {inp['name']}_ptr = m_{inp['name']}.getIfExists();\nauto const& {inp['name']} = {inp['name']}_ptr ? *{inp['name']}_ptr : decltype(*{inp['name']}_ptr){{{{}},{{}}}};" for inp in inputs if "optional" in inp))}

    auto const& runtime_options = *m_runtime_options.get();
    auto const& constants = *m_constants.get();
    auto const& odin = *m_odin.get();

    // Call algorithm update method on first event or if run number changes.
    {{
      std::scoped_lock lock{{m_mut}};
      if ( !m_runNumber || *m_runNumber != odin.runNumber() ) {{
        m_algorithm.update(*constants); m_runNumber = odin.runNumber();
      }}
    }}

    // Aggregates
    {"\n".join(get_aggregates)}
    std::tuple<{",".join(aggregate_types)}> input_aggregates_tuple {{{",".join(arg_data_agg_typenames)}}};

    // Output container
    [[maybe_unused]] std::tuple<{",".join(output_types)}> output_container{{{','.join(output_alloc)}}};
    
    // TES wrappers
    {tes_wrappers}

    // Inputs to set_arguments_size and operator()
    {tes_wrappers_reference}
    Allen::Context context{{}};
    const auto argument_references = ArgumentReferences<{param_namespace}>{{tes_wrappers_references, input_aggregates_tuple}};
    
    // set arguments size invocation
    m_algorithm.set_arguments_size(argument_references, runtime_options, *constants);
    
    // algorithm operator() invocation
    m_algorithm(argument_references, runtime_options, *constants, context);

    auto filter_decision = {filter_decision};

    {"\n".join([f"m_{out['name']}.put(std::move(std::get<{i}>(output_container)));" for i, out in enumerate(outputs)])}

    return filter_decision;
  }}
}};
DECLARE_COMPONENT({name})
"""
        return code

    @staticmethod
    def write_gaudi_algorithms(algorithms, algorithm_wrappers_folder):
        algorithms_generated_filenames = []
        for _, alg in algorithms.items():
            code = AllenGaudiWrapperGenerator.generate_gaudi_wrapper(alg)

            _, name = get_namespace(alg["name"])
            output_filename = f"{algorithm_wrappers_folder}/{name}_gaudi.cpp"
            with open(output_filename, "w") as f:
                f.write(code)

            algorithms_generated_filenames.append(output_filename)
        return algorithms_generated_filenames

    @staticmethod
    def write_algorithm_filename_list(algorithms,
                                      algorithm_wrappers_folder,
                                      output_filename,
                                      separator=";"):
        filenames = []
        for name in algorithms:
            filename = f"{algorithm_wrappers_folder}/{name}_gaudi.cpp"
            filenames.append(filename)
        s = separator.join([a for a in filenames])
        with open(output_filename, "w") as f:
            f.write(s)

def make_properties(properties_json):
    properties = []
    for [name, [value, typedef, description]] in properties_json.items():
        init = f"m_algorithm.set_property_value<{typedef}>(m_{name}.name(), m_{name}.value());"
        prop = f"Gaudi::Property<{typedef}> m_{name}{{this, \"{name}\", m_algorithm.get_property<{typedef}>(\"{name}\"), [=, this](auto&) {{ {init} }}, Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}, \"{description}\" }};"
        properties.append(prop)
    return properties

def make_parameters(parameters):
    for p in parameters:
        namespace, name = get_namespace(p["typename"])
        p["namespace"] = namespace
        p["name"] = name
    inputs = [p for p in parameters if p["kind"] == "input" and "aggregate" not in p]
    aggregates = [p for p in parameters if p["kind"] == "input" and "aggregate" in p]
    outputs = [p for p in parameters if p["kind"] == "output"]
    non_aggregates = [p for p in parameters if "aggregate" not in p] # this is not just inputs+outputs, it has to keep the order
    return inputs, outputs, aggregates, non_aggregates

def make_parameter_types(parameters):
    return [
        f"{p["typename"]}::type"
        for p in parameters
    ]

def make_param_vectors(parameters):
    return [
        f"Allen::parameter_vector<{t}>"
        for t in make_parameter_types(parameters)
    ]

def make_aggregate_types(parameters):
    return [
        f"typename {t}::type"
        for t in make_parameter_types(parameters)
    ]

def make_output_alloc(parameters):
    return [
        f"Allen::parameter_vector<{t}>{{Allen::param_vector_alloc<{t}>{{ LHCb::getMemResource( evtCtx ) }}}}"
        for t in make_parameter_types(parameters)
    ]