#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Allen::CuDNN {

  struct Workspace {
    void* ptr = nullptr;
    size_t bytes = 0;

    bool empty() const { return ptr == nullptr || bytes == 0; }

    void require(size_t required_bytes, const char* owner) const
    {
      if (required_bytes == 0) return;
      if (ptr == nullptr || bytes < required_bytes) {
        throw std::runtime_error(
          std::string(owner) + " requires " + std::to_string(required_bytes) +
          " workspace bytes, but external workspace has " + std::to_string(bytes));
      }
    }
  };

  struct WorkspaceRequirement {
    std::string name;
    size_t bytes = 0;
    size_t alignment = 256;
  };

  struct WorkspaceSlice {
    std::string name;
    size_t offset = 0;
    size_t bytes = 0;
    size_t alignment = 256;
  };

  enum class WorkspacePlanningMode {
    NonOverlapping,
    Overlapping
  };

  inline size_t align_workspace_offset(size_t offset, size_t alignment)
  {
    if (alignment <= 1) return offset;
    const size_t remainder = offset % alignment;
    return remainder == 0 ? offset : offset + (alignment - remainder);
  }

  struct WorkspacePlan {
    WorkspacePlanningMode mode = WorkspacePlanningMode::NonOverlapping;
    size_t total_bytes = 0;
    size_t max_required_bytes = 0;
    std::vector<WorkspaceSlice> slices {};

    bool empty() const { return slices.empty() || total_bytes == 0; }

    const WorkspaceSlice& slice(const std::string& name) const
    {
      const auto found = std::find_if(slices.begin(), slices.end(), [&name](const WorkspaceSlice& s) {
        return s.name == name;
      });
      if (found == slices.end()) {
        throw std::invalid_argument("AllenCuDNN: workspace plan has no slice named " + name);
      }
      return *found;
    }
  };

  class WorkspacePlanner {
  public:
    void add(std::string name, size_t bytes, size_t alignment = 256)
    {
      if (alignment == 0) {
        throw std::invalid_argument("AllenCuDNN: workspace alignment must be non-zero");
      }
      m_requirements.push_back({std::move(name), bytes, alignment});
    }

    const std::vector<WorkspaceRequirement>& requirements() const { return m_requirements; }

    WorkspacePlan plan(WorkspacePlanningMode mode) const
    {
      WorkspacePlan result {};
      result.mode = mode;
      for (const auto& requirement : m_requirements) {
        result.max_required_bytes = std::max(result.max_required_bytes, requirement.bytes);
        if (mode == WorkspacePlanningMode::NonOverlapping) {
          result.slices.push_back({requirement.name, 0, requirement.bytes, requirement.alignment});
          result.total_bytes = std::max(result.total_bytes, requirement.bytes);
        }
        else {
          if (requirement.bytes == 0) {
            result.slices.push_back({requirement.name, result.total_bytes, 0, requirement.alignment});
            continue;
          }
          const size_t offset = align_workspace_offset(result.total_bytes, requirement.alignment);
          result.slices.push_back({requirement.name, offset, requirement.bytes, requirement.alignment});
          result.total_bytes = offset + requirement.bytes;
        }
      }
      return result;
    }

    WorkspacePlan non_overlapping_plan() const { return plan(WorkspacePlanningMode::NonOverlapping); }
    WorkspacePlan overlapping_plan() const { return plan(WorkspacePlanningMode::Overlapping); }

  private:
    std::vector<WorkspaceRequirement> m_requirements {};
  };

  class WorkspaceArena {
  public:
    WorkspaceArena() = default;
    WorkspaceArena(Workspace workspace, WorkspacePlan plan) : m_workspace(workspace), m_plan(std::move(plan))
    {
      m_workspace.require(m_plan.total_bytes, "WorkspaceArena");
    }

    Workspace whole() const
    {
      m_workspace.require(m_plan.total_bytes, "WorkspaceArena");
      return m_workspace;
    }

    Workspace slice(const std::string& name) const
    {
      const auto& s = m_plan.slice(name);
      m_workspace.require(s.offset + s.bytes, "WorkspaceArena");
      auto* base = static_cast<unsigned char*>(m_workspace.ptr);
      return {base + s.offset, s.bytes};
    }

    const WorkspacePlan& plan() const { return m_plan; }
    size_t total_bytes() const { return m_plan.total_bytes; }

  private:
    Workspace m_workspace {};
    WorkspacePlan m_plan {};
  };

} // namespace Allen::CuDNN
