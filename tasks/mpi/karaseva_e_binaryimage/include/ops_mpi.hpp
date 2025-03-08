#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_mpi {

// Function declarations
int GetRootLabel(std::map<int, std::set<int>>& label_parent_map, int label);
void PropagateLabelEquivalences(std::map<int, std::set<int>>& label_parent_map);
void UpdateLabels(std::vector<int>& labeled_image, int rows, int cols);
void UnionLabels(std::map<int, std::set<int>>& label_parent_map, int new_label, int neighbour_label);
void Labeling(std::vector<int>& input_image, std::vector<int>& labeled_image, int rows, int cols, int min_label,
              std::map<int, std::set<int>>& label_parent_map);
void SaveLabelMapToStream(std::ostringstream& oss, const std::map<int, std::set<int>>& label_map);
void LoadLabelMapFromStream(std::istringstream& iss, std::map<int, std::set<int>>& label_map);
void SaveLabelSetToStream(std::ostringstream& oss, const std::set<int>& label_set);
void LoadLabelSetFromStream(std::istringstream& iss, std::set<int>& label_set);
// std::vector<int> FindNeighbors(const std::vector<int>& labeled_image, int x, int y, int rows, int cols);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> image_;
  std::vector<int> labeled_image_;
  int rows_;
  int columns_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> image_, local_image_;
  std::vector<int> labeled_image_;
  int rows_;
  int columns_;
  boost::mpi::communicator world_;
};

}  // namespace karaseva_e_binaryimage_mpi