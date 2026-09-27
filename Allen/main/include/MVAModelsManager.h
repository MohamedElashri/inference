/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

#include "BackendCommon.h"
#include "InputReader.h"
#include <functional>
#include <string>

namespace Allen::MVAModels {

  struct MVAModelBase;

  struct MVAModelsManager {
    static MVAModelsManager* get()
    {
      static MVAModelsManager instance;
      return &instance;
    }

    void registerNN(MVAModelBase* nn) { m_neural_networks.push_back(nn); }

    void loadData(std::string parameters_path);

    std::vector<MVAModelBase*> m_neural_networks;
  };

  // A model of an algorithm, a member of it. It registers itself on
  // construction; MVAModelsManager::loadData reads every model once, after the
  // algorithms' properties are set and before their init(), so init() can use
  // the model's data.
  struct MVAModelBase {
    // path: the model file, relative to the parameters directory (--params).
    MVAModelBase(std::string name, std::string path) : m_name(name), m_path(path)
    {
      MVAModelsManager::get()->registerNN(this);
    }

    // path_source: gives the model file when the model is read, so that it can
    // come from a property of the algorithm (the properties are set by then).
    MVAModelBase(std::string name, std::function<std::string()> path_source) :
      m_name(name), m_path_source(std::move(path_source))
    {
      MVAModelsManager::get()->registerNN(this);
    }

    virtual void readData(std::string) {}

    virtual ~MVAModelBase() = default;

    // The model file. With a fixed path, parameters_path + path (the fixed
    // paths start with "/"); with a path source, an absolute path as it is and
    // a relative one in the parameters directory.
    std::string file_path(const std::string& parameters_path) const
    {
      if (!m_path_source) return parameters_path + m_path;
      const std::string path = m_path_source();
      if (!path.empty() && path.front() == '/') return path;
      const bool separated = !parameters_path.empty() && parameters_path.back() == '/';
      return parameters_path + (separated ? "" : "/") + path;
    }

    bool data_was_read_before = false;
    std::string m_name;
    std::string m_path;
    std::function<std::string()> m_path_source;
  };

} // namespace Allen::MVAModels