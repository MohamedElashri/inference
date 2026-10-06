/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// ----------------------------------------------------------------------------
// Scheduling/control-flow helpers used by MultiEventScheduler: boolean CF
// expressions, algorithm configuration, topological sort, execution-mask
// evaluation and lifetime-dependency calculation.
// ----------------------------------------------------------------------------

#pragma once

#include <Gaudi/Algorithm.h>

#include "HLTScheduler/CFNodePropertiesParse.h"
#include "HLTScheduler/ControlFlowNode.h"

#include <algorithm>
#include <ranges>
#include <set>
#include <vector>

#include "EventMask.h"

namespace Allen::Scheduler {
  struct BoolExpr {
    enum NodeType {
      ALGORITHM, // Leaf node representing an algorithm
      AND,
      OR,
      NOT,
      CONST_TRUE,
      CONST_FALSE
    };

    NodeType type {ALGORITHM};
    std::vector<BoolExpr> children {};
    int alg {-1}; // Valid only if type == ALGORITHM

    friend bool operator==(const BoolExpr& a, const BoolExpr& b)
    {
      if (a.type != b.type || a.alg != b.alg || a.children.size() != b.children.size()) return false;
      for (unsigned i = 0; i < a.children.size(); i++) {
        if (a.children[i] != b.children[i]) return false;
      }
      return true;
    }

    int maxOrder() const
    {
      if (type == ALGORITHM) return alg;
      int max = -1;
      for (auto& c : children) {
        int co = c.maxOrder();
        if (max < co) max = co;
      }
      return max;
    }

    void evaluate(EventMask& out, const std::span<EventMask>& event_masks) const
    {
      if (type == ALGORITHM) {
        out = event_masks[alg];
      }
      else if (type == CONST_TRUE) {
        out.fill();
      }
      else if (type == CONST_FALSE) {
        out.reset();
      }
      else if (type == OR) {
        EventMask tmp {out.max_events()};
        out.reset();
        for (auto& c : children) {
          c.evaluate(tmp, event_masks);
          out |= tmp;
        }
      }
      else if (type == AND) {
        EventMask tmp {out.max_events()};
        out.fill();
        for (auto& c : children) {
          c.evaluate(tmp, event_masks);
          out &= tmp;
        }
      }
      else if (type == NOT) {
        EventMask tmp {out.max_events()};
        children[0].evaluate(tmp, event_masks);
        out = ~tmp;
      }
    }

    void addChild(BoolExpr c)
    {
      if (type == OR) {
        if ((children.size() == 1 && children[0].type == CONST_FALSE) || c.type == CONST_TRUE) {
          children.clear();
        }
        if (
          (children.size() == 1 && children[0].type == CONST_TRUE) || (children.size() != 0 && c.type == CONST_FALSE)) {
          return;
        }
        if (c.type == OR || (c.children.size() == 1 && c.type != NOT)) {
          for (auto& n : c.children) {
            addChild(n);
          }
          return;
        }
      }
      else if (type == AND) {
        if ((children.size() == 1 && children[0].type == CONST_TRUE) || c.type == CONST_FALSE) {
          children.clear();
        }
        if (
          (children.size() == 1 && children[0].type == CONST_FALSE) || (children.size() != 0 && c.type == CONST_TRUE)) {
          return;
        }
        if (c.type == AND || (c.children.size() == 1 && c.type != NOT)) {
          for (auto& n : c.children) {
            addChild(n);
          }
          return;
        }
      }
      if (std::find_if(children.begin(), children.end(), [&c](const auto& value) {
            return value == c;
          }) != children.end()) {
        return;
      }
      children.emplace_back(c);
    }

    BoolExpr simplify()
    {
      if (children.size() == 1 && type != NOT) {
        return children[0];
      }
      return *this;
    }

    std::string to_string(const std::vector<std::string>& names) const
    {
      if (type == ALGORITHM) {
        return names[alg];
      }
      else if (type == CONST_TRUE) {
        return "true";
      }
      else if (type == CONST_FALSE) {
        return "false";
      }
      else if (type == NOT) {
        return "~" + children[0].to_string(names);
      }
      else {
        char sep = type == AND ? '&' : '|';
        std::string result = "(";
        for (unsigned i = 0; i < children.size(); i++) {
          if (i > 0) result += sep;
          result += children[i].to_string(names);
        }
        return result + ")";
      }
    }

    static BoolExpr make_alg(int alg)
    {
      BoolExpr out;
      out.type = ALGORITHM;
      out.alg = alg;
      return out;
    }

    static BoolExpr make_not(BoolExpr e)
    {
      BoolExpr out;
      out.type = NOT;
      out.children.emplace_back(e);
      return out;
    }

    static BoolExpr make_or()
    {
      BoolExpr out;
      out.type = OR;
      BoolExpr& falseMask = out.children.emplace_back();
      falseMask.type = CONST_FALSE;
      return out;
    }

    static BoolExpr make_and()
    {
      BoolExpr out;
      out.type = AND;
      BoolExpr& trueMask = out.children.emplace_back();
      trueMask.type = CONST_TRUE;
      return out;
    }
  };

  struct AlgEntry {
    std::unique_ptr<Gaudi::Algorithm> alg;
    std::vector<DataObjID const*> inputs;
    std::vector<DataObjID const*> outputs;
    // Keys coming from aggregate (list-style) inputs. Like the old standalone
    // scheduler's all_producers(False), these are conditional inputs and must
    // not be traversed when propagating execution masks: e.g. gather_selections
    // aggregates every line output but does not force the lines to run on the
    // masks of the algorithms that consume the gathered result (lumi, persistency).
    std::unordered_set<std::string> aggregate_input_keys;
    std::vector<AlgEntry*> df_dependencies;
    // Data dependencies excluding event-list (mask) handles, which the
    // scheduler produces itself. Only these represent real producer/consumer
    // relations for execution-mask propagation.
    std::vector<AlgEntry*> data_df_dependencies;
    std::vector<AlgEntry*> all_df_dependencies;
    std::vector<AlgEntry*> cf_dependencies;
    bool isMultiEvent {false};
    bool isMultiEventOutput {false};
    DataObjectReadHandle<mask_vec_t>* inputMaskHandle {nullptr};
    DataObjectWriteHandle<mask_vec_t>* outputMaskHandle {nullptr};
    int index {-1}; // the index in the final sorted sequence
    AlgEntry(std::unique_ptr<Gaudi::Algorithm>&& _alg) : alg {std::move(_alg)}
    {
      constexpr auto gather = [](auto& c, auto const& in1, auto const& in2) {
        c.reserve(in1.size() + in2.size());
        for (const DataObjID& id : in1)
          c.push_back(&id);
        for (const DataObjID& id : in2)
          c.push_back(&id);
        constexpr auto by_key = [](const DataObjID* id) { return id->fullKey(); };
        std::ranges::sort(c, std::less {}, by_key);
        auto od = std::ranges::unique(c, std::equal_to {}, by_key);
        c.erase(od.begin(), od.end());
      };
      gather(inputs, alg->inputDataObjs(), alg->extraInputDeps());
      gather(outputs, alg->outputDataObjs(), alg->extraOutputDeps());

      isMultiEvent = alg->hasProperty("IsMultiEvent") ?
                       static_cast<Gaudi::Property<bool>*>(alg->property("IsMultiEvent"))->value() :
                       false;

      isMultiEventOutput =
        isMultiEvent && (alg->hasProperty("IsMultiEventOutput") ?
                           static_cast<Gaudi::Property<bool>*>(alg->property("IsMultiEventOutput"))->value() :
                           true);

      // Search for the input/output masks:
      for (Gaudi::DataHandle* handle : alg->inputHandles()) {
        inputMaskHandle = dynamic_cast<DataObjectReadHandle<mask_vec_t>*>(handle);
        if (inputMaskHandle) break; // found it
      }
      for (Gaudi::DataHandle* handle : alg->outputHandles()) {
        outputMaskHandle = dynamic_cast<DataObjectWriteHandle<mask_vec_t>*>(handle);
        if (outputMaskHandle) break; // found it
      }

      // Aggregate handles are stored as Gaudi::Property<std::vector<DataObjID>>
      // (e.g. host_input_line_data_t), which is distinct from the
      // Gaudi::Property<DataObjIDColl> used for ExtraInputs/ExtraOutputs.
      for (auto* prop : alg->getProperties()) {
        if (auto* aggregate = dynamic_cast<Gaudi::Property<std::vector<DataObjID>>*>(prop)) {
          for (const auto& id : aggregate->value()) {
            aggregate_input_keys.insert(id.key());
          }
        }
      }
    }
    std::string to_string() const { return alg->name(); }
  };

  struct ConfiguredAlgorithm {
    std::unique_ptr<Gaudi::Algorithm> alg;
    bool isMultiEvent {false};
    DataObjectReadHandle<mask_vec_t>* inputMaskHandle {nullptr};
    DataObjectWriteHandle<mask_vec_t>* outputMaskHandle {nullptr};
    int index {-1};
    ConfiguredAlgorithm(AlgEntry* e) :
      alg(std::move(e->alg)), isMultiEvent {e->isMultiEvent}, inputMaskHandle {e->inputMaskHandle},
      outputMaskHandle {e->outputMaskHandle}, index {e->index}
    {}
  };

  std::vector<ConfiguredAlgorithm> finalizeConfiguration(std::vector<AlgEntry*>& sorted)
  {
    std::vector<ConfiguredAlgorithm> configured;
    configured.reserve(sorted.size());
    for (auto& alg : sorted) {
      configured.emplace_back(alg);
    }
    return configured;
  }

  std::vector<std::string> algorithm_names(std::vector<AlgEntry*>& sorted)
  {
    std::vector<std::string> names;
    names.reserve(sorted.size());
    for (auto& alg : sorted) {
      names.emplace_back(alg->alg->name());
    }
    return names;
  }

  AlgEntry createAlgorithm(IAlgManager& am, const std::string& alg_name)
  {
    const Gaudi::Utils::TypeNameString tn(alg_name);
    IAlgorithm* tmp = nullptr;
    StatusCode sc = am.createAlgorithm(tn.type(), tn.name(), tmp);
    if (sc.isFailure()) {
      throw GaudiException {"Failed to create " + alg_name, __func__, StatusCode::FAILURE};
    }
    sc = tmp->sysInitialize();
    if (sc.isFailure()) {
      throw GaudiException {"Failed to initialize " + alg_name, __func__, StatusCode::FAILURE};
    }
    return {std::unique_ptr<Gaudi::Algorithm>(dynamic_cast<Gaudi::Algorithm*>(tmp))};
  }

  std::vector<std::string> collect_leaf_algorithms(
    const std::map<std::string, NodeDefinition>& cf_nodes,
    const std::string& node_name)
  {

    std::unordered_set<std::string> result;
    std::function<void(const std::string&)> collect = [&](const std::string& current) {
      auto it = cf_nodes.find(current);

      if (it == cf_nodes.end()) {
        result.insert(current);
      }
      else {
        for (const auto& child : it->second.children) {
          collect(child);
        }
      }
    };

    collect(node_name);
    return std::vector<std::string>(result.begin(), result.end());
  }

  std::unordered_map<std::string, AlgEntry> configured_algorithms(
    IAlgManager& am,
    std::span<std::string> names,
    std::map<std::string, NodeDefinition> const& cf_nodes)
  {

    std::unordered_map<std::string, AlgEntry> algs; // alg_name => alg map
    std::map<DataObjID, AlgEntry*> producers;       // data => producer map
    algs.reserve(names.size());

    // instanciate algorithms
    for (const auto& alg_name : names) {
      if (algs.contains(alg_name)) continue;
      algs.emplace(alg_name, createAlgorithm(am, alg_name));
    }

    // build producers map (alg pointers are stable at this point)
    for (auto& [name, alg] : algs) {
      for (auto& out : alg.outputs) {
        producers.emplace(out->key(), &alg);
      }
    }

    // resolve dataflow dependencies
    for (auto& [name, alg] : algs) {
      alg.df_dependencies.reserve(alg.inputs.size());
      alg.data_df_dependencies.reserve(alg.inputs.size());
      const std::string mask_key = alg.inputMaskHandle ? alg.inputMaskHandle->objKey() : std::string {};
      for (auto& in : alg.inputs) {
        alg.df_dependencies.emplace_back(producers[in->key()]);
        if (alg.aggregate_input_keys.contains(in->key())) continue;
        if (!mask_key.empty() && in->key() == mask_key) continue;
        alg.data_df_dependencies.emplace_back(producers[in->key()]);
      }
    }

    // add controlflow dependencies
    for (auto& [node_name, nodeDef] : cf_nodes) {
      if (nodeDef.ordered) { // Force order, add dependencies between children
        for (unsigned i = 1; i < nodeDef.children.size(); i++) {
          for (auto& pre : collect_leaf_algorithms(cf_nodes, nodeDef.children[i - 1])) {
            for (auto& cur : collect_leaf_algorithms(cf_nodes, nodeDef.children[i])) {
              algs.at(cur).cf_dependencies.emplace_back(&algs.at(pre));
            }
          }
        }
      }
    }

    // deduplicate dependencies:
    for (auto& [name, alg] : algs) {
      std::ranges::sort(alg.df_dependencies, std::less {});
      auto od = std::ranges::unique(alg.df_dependencies, std::equal_to {});
      alg.df_dependencies.erase(od.begin(), od.end());
      std::ranges::sort(alg.data_df_dependencies, std::less {});
      od = std::ranges::unique(alg.data_df_dependencies, std::equal_to {});
      alg.data_df_dependencies.erase(od.begin(), od.end());
      std::ranges::sort(alg.cf_dependencies, std::less {});
      od = std::ranges::unique(alg.cf_dependencies, std::equal_to {});
      alg.cf_dependencies.erase(od.begin(), od.end());
    }

    return algs;
  }

  std::vector<AlgEntry*> topological_sort(std::unordered_map<std::string, AlgEntry>& algorithms)
  {
    std::vector<AlgEntry*> sorted;
    sorted.reserve(algorithms.size());

    // Build reverse adjacency lists
    std::unordered_map<AlgEntry*, std::vector<AlgEntry*>> df_reverse_deps;
    std::unordered_map<AlgEntry*, std::vector<AlgEntry*>> cf_reverse_deps;

    std::unordered_map<AlgEntry*, size_t> remaining_df_deps;
    std::unordered_map<AlgEntry*, size_t> remaining_cf_deps;

    // Track which set each algorithm is in (for moving between sets)
    std::unordered_map<AlgEntry*, int> ready_status; // 0=not ready, 1=df_only, 2=both

    // Initialize
    for (auto& [name, alg] : algorithms) {
      AlgEntry* alg_ptr = &alg;
      remaining_df_deps[alg_ptr] = alg.df_dependencies.size();
      remaining_cf_deps[alg_ptr] = alg.cf_dependencies.size();

      for (auto* dep : alg.df_dependencies) {
        df_reverse_deps[dep].push_back(alg_ptr);
      }
      for (auto* dep : alg.cf_dependencies) {
        cf_reverse_deps[dep].push_back(alg_ptr);
      }
    }

    // Initial ready sets
    std::vector<AlgEntry*> both_ready;
    std::vector<AlgEntry*> df_only_ready;

    for (auto& [name, alg] : algorithms) {
      AlgEntry* alg_ptr = &alg;
      if (remaining_df_deps[alg_ptr] == 0) {
        if (remaining_cf_deps[alg_ptr] == 0) {
          both_ready.push_back(alg_ptr);
          ready_status[alg_ptr] = 2;
        }
        else {
          df_only_ready.push_back(alg_ptr);
          ready_status[alg_ptr] = 1;
        }
      }
    }

    size_t scheduled_count = 0;
    size_t total = algorithms.size();

    while (scheduled_count < total) {
      // Choose candidates
      std::vector<AlgEntry*>& candidates = !both_ready.empty() ? both_ready : df_only_ready;

      if (candidates.empty()) {
        throw GaudiException {"Topological sort failure - circular dependency", __func__, StatusCode::FAILURE};
      }

      AlgEntry* selected = candidates.back();
      candidates.pop_back();
      ready_status.erase(selected); // Remove from tracking

      sorted.push_back(selected);
      scheduled_count++;

      // Update dataflow dependents
      for (auto* dependent : df_reverse_deps[selected]) {
        if (--remaining_df_deps[dependent] == 0) {
          if (remaining_cf_deps[dependent] == 0) {
            both_ready.push_back(dependent);
            ready_status[dependent] = 2;
          }
          else {
            df_only_ready.push_back(dependent);
            ready_status[dependent] = 1;
          }
        }
      }

      // Update controlflow dependents
      for (auto* dependent : cf_reverse_deps[selected]) {
        if (--remaining_cf_deps[dependent] == 0) {
          // Check if it's already dataflow ready
          if (remaining_df_deps[dependent] == 0) {
            // Only promote if not already scheduled (still tracked in ready_status).
            // Use find() instead of operator[] to avoid re-inserting already-erased entries.
            auto rs_it = ready_status.find(dependent);
            if (rs_it != ready_status.end()) {
              // If it was in df_only_ready, remove it and add to both_ready
              if (rs_it->second == 1) {
                auto it = std::find(df_only_ready.begin(), df_only_ready.end(), dependent);
                if (it != df_only_ready.end()) {
                  *it = df_only_ready.back();
                  df_only_ready.pop_back();
                }
              }
              both_ready.push_back(dependent);
              ready_status[dependent] = 2;
            }
          }
          // If not dataflow ready yet, it will be handled when dataflow becomes ready
        }
      }
    }

    // Detect circular dependencies that the sort cannot deadlock on:
    //  - DF-CF: A needs B's data, but B must run after A (or vice versa)
    //  - CF-CF: mutual CF dependencies (both have df=0, so they slip through df_only_ready)
    // (DF-DF cycles already cause a deadlock caught by the sort above.)
    {
      std::set<std::pair<AlgEntry*, AlgEntry*>> cycles;
      for (auto& [name, alg] : algorithms) {
        AlgEntry* a = &alg;

        // DF-CF: a depends on b via DF, b depends on a via CF
        for (auto* b : alg.df_dependencies) {
          if (!b) continue;
          for (auto* back : b->cf_dependencies) {
            if (back == a) cycles.insert(std::minmax(a, b));
          }
        }
        // CF-CF: a depends on b via CF, b depends on a via CF
        for (auto* b : alg.cf_dependencies) {
          if (!b) continue;
          for (auto* back : b->cf_dependencies) {
            if (back == a) cycles.insert(std::minmax(a, b));
          }
        }
      }
      if (!cycles.empty()) {
        std::string msg = "Circular dependency detected:\n";
        for (auto& [x, y] : cycles) {
          msg += "  " + x->to_string() + " <--> " + y->to_string() + "\n";
        }
        throw GaudiException {msg, __func__, StatusCode::FAILURE};
      }
    }

    return sorted;
  }

  BoolExpr get_tree_for_node(
    const std::string& node,
    std::vector<BoolExpr>& execution_masks,
    std::unordered_map<std::string, AlgEntry>& algorithms,
    std::map<std::string, NodeDefinition>& cf_nodes,
    BoolExpr parent_mask)
  {
    BoolExpr e;
    auto it = cf_nodes.find(node);
    if (it == cf_nodes.end()) {
      auto& alg = algorithms.at(node);
      e.alg = alg.index;
      execution_masks[alg.index].addChild(parent_mask);

      // Propagate parent_mask through data dependencies so that producers that
      // are only reachable through this algorithm's inputs run in the same mask
      // as their consumer (HLT2/Moore "on demand" semantics).
      auto mask_order = parent_mask.maxOrder();
      for (auto* dep : alg.all_df_dependencies) {
        if (mask_order >= dep->index) continue;
        execution_masks[dep->index].addChild(parent_mask);
      }
    }
    else {
      auto& nodeDef = cf_nodes.at(node);
      const nodeType type = toNodeType(nodeDef.type);
      if (type == nodeType::NOT) {
        e.type = BoolExpr::NodeType::NOT;
        e.addChild(get_tree_for_node(nodeDef.children[0], execution_masks, algorithms, cf_nodes, parent_mask));
      }
      else if (type == nodeType::LAZY_AND || type == nodeType::NONLAZY_AND) {
        e.type = BoolExpr::NodeType::AND;
        for (unsigned i = 0; i < nodeDef.children.size(); i++) {
          auto ce = get_tree_for_node(nodeDef.children[i], execution_masks, algorithms, cf_nodes, parent_mask);
          e.addChild(ce);
          if (type == nodeType::LAZY_AND) parent_mask = ce;
        }
      }
      else if (type == nodeType::LAZY_OR || type == nodeType::NONLAZY_OR) {
        e.type = BoolExpr::NodeType::OR;
        for (unsigned i = 0; i < nodeDef.children.size(); i++) {
          auto ce = get_tree_for_node(nodeDef.children[i], execution_masks, algorithms, cf_nodes, parent_mask);
          e.addChild(ce);
          if (type == nodeType::LAZY_OR) parent_mask = BoolExpr::make_not(ce);
        }
      }
    }
    return e;
  }

  BoolExpr get_tree_for_node2(
    const std::string& node,
    const std::vector<BoolExpr>& execution_masks,
    std::unordered_map<std::string, AlgEntry>& algorithms,
    std::map<std::string, NodeDefinition>& cf_nodes)
  {
    BoolExpr e;
    auto it = cf_nodes.find(node);
    if (it == cf_nodes.end()) {
      auto& alg = algorithms.at(node);
      e.alg = alg.index;
    }
    else {
      auto& nodeDef = cf_nodes.at(node);
      const nodeType type = toNodeType(nodeDef.type);
      if (type == nodeType::NOT) {
        e.type = BoolExpr::NodeType::NOT;
        e.addChild(get_tree_for_node2(nodeDef.children[0], execution_masks, algorithms, cf_nodes));
      }
      else if (type == nodeType::LAZY_AND || type == nodeType::NONLAZY_AND) {
        e.type = BoolExpr::NodeType::AND;
        for (unsigned i = 0; i < nodeDef.children.size(); i++) {
          auto ce = get_tree_for_node2(nodeDef.children[i], execution_masks, algorithms, cf_nodes);
          e.addChild(ce);
        }
      }
      else if (type == nodeType::LAZY_OR || type == nodeType::NONLAZY_OR) {
        e.type = BoolExpr::NodeType::OR;
        for (unsigned i = 0; i < nodeDef.children.size(); i++) {
          auto ce = get_tree_for_node2(nodeDef.children[i], execution_masks, algorithms, cf_nodes);
          e.addChild(ce);
        }
      }
    }
    return e;
  }

  std::vector<std::string> get_top_nodes(const std::map<std::string, NodeDefinition>& nodes)
  {
    std::unordered_set<std::string> all_children;
    for (const auto& [name, node] : nodes) {
      for (const auto& child : node.children) {
        all_children.insert(child);
      }
    }
    std::vector<std::string> top_nodes;
    for (const auto& [name, node] : nodes) {
      if (all_children.find(name) == all_children.end()) {
        top_nodes.push_back(name);
      }
    }
    return top_nodes;
  }

  std::vector<BoolExpr> find_execution_masks(
    std::vector<AlgEntry*>& sorted,
    std::unordered_map<std::string, AlgEntry>& algorithms,
    std::map<std::string, NodeDefinition>& cf_nodes)
  {
    std::vector<BoolExpr> execution_masks;
    execution_masks.reserve(sorted.size());

    // Init CF_MASK:
    int i = 0;
    for (auto& alg : sorted) {
      alg->index = i++;
      BoolExpr& mask = execution_masks.emplace_back();
      mask.type = BoolExpr::NodeType::OR;
      BoolExpr& falseMask = mask.children.emplace_back();
      falseMask.type = BoolExpr::NodeType::CONST_FALSE;

      // Compute flattened and deduplicated all_df_dependencies, in linear time using dynamic programming:
      alg->all_df_dependencies = alg->data_df_dependencies;
      for (auto* dep : alg->data_df_dependencies) {
        auto& deps_of_dep = dep->all_df_dependencies;
        alg->all_df_dependencies.insert(alg->all_df_dependencies.end(), deps_of_dep.begin(), deps_of_dep.end());
      }
      std::ranges::sort(alg->all_df_dependencies, [](AlgEntry* a, AlgEntry* b) { return a->index < b->index; });
      auto [first, last] = std::ranges::unique(alg->all_df_dependencies);
      alg->all_df_dependencies.erase(first, last);
    }

    // Sort control flow and aggregate execution masks
    auto top_nodes = get_top_nodes(cf_nodes);
    for (auto& node_name : top_nodes) {
      BoolExpr trueMask;
      trueMask.type = BoolExpr::NodeType::CONST_TRUE;
      get_tree_for_node(node_name, execution_masks, algorithms, cf_nodes, trueMask);
    }

    for (auto& alg : sorted) {
      BoolExpr& cf_mask = execution_masks[alg->index];
      if (cf_mask.children.size() == 1 && cf_mask.children[0].type == BoolExpr::NodeType::CONST_FALSE) {
        cf_mask.children[0].type = BoolExpr::NodeType::CONST_TRUE; // if no execution mask, always execute
      }
    }

    for (const auto& alg : sorted) {
      execution_masks[alg->index] = execution_masks[alg->index].simplify();

      for (int i = 0; i < alg->index; i++) {
        const auto& alg2 = sorted[i];
        if (!alg2->outputMaskHandle && alg2->inputMaskHandle && execution_masks[i] == execution_masks[alg->index]) {
          execution_masks[alg->index] = BoolExpr::make_alg(i);
          break;
        }
      }
    }

    return execution_masks;
  }

  using LifetimeDependencies = std::vector<DataObjID const*>;

  std::pair<std::vector<LifetimeDependencies>, std::vector<LifetimeDependencies>> calculate_lifetime_dependencies(
    const std::vector<AlgEntry*>& sorted)
  {
    std::vector<LifetimeDependencies> in_deps; // arguments to reserve at alg i
    in_deps.reserve(sorted.size());
    std::vector<LifetimeDependencies> out_deps; // arguments to free at alg i
    out_deps.reserve(sorted.size());

    // Pre-compute the last usage index for each argument
    std::unordered_map<std::string, int> last_usage_index;
    for (auto& alg : sorted) {
      for (const auto& input : alg->inputs) {
        last_usage_index[input->key()] = alg->index; // Keep overwriting - we want the max index
      }
    }

    std::unordered_set<const DataObjID*> live_args {};

    for (auto& alg : sorted) {
      auto& in = in_deps.emplace_back();
      auto& out = out_deps.emplace_back();

      for (const auto& live : live_args) {
        auto it = last_usage_index.find(live->key());
        // Argument is done if:
        // - It's never used again (not in map), OR
        // - Its last usage is before or at current algorithm
        if (it == last_usage_index.end() || it->second <= alg->index) {
          out.push_back(live);
        }
      }

      // Remove dead arguments
      for (const auto& arg : out) {
        live_args.erase(arg);
      }

      // Add new arguments
      if (alg->isMultiEventOutput) {
        in.reserve(alg->inputs.size());
        for (auto& output : alg->outputs) {
          in.push_back(output);
          live_args.insert(output);
        }
      }
    }

    return {in_deps, out_deps};
  }
} // namespace Allen::Scheduler
