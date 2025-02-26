#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static int FindRootLabel(std::unordered_map<int, int>& label_parent, int label);
  static void MergeLabels(std::unordered_map<int, int>& label_parent, int label1, int label2);
  static void LabelingImage(std::vector<int>& image, std::vector<int>& labeled_image, int rows, int cols, int min_label,
                            std::unordered_map<int, int>& label_parent, int start_row, int end_row);
  static void HandleNeighbors(int x, int y, int rows, int cols, const std::vector<int>& labeled_image,
                              std::vector<int>& neighbors);
  static void AssignLabel(int pos, std::vector<int>& labeled_image, std::unordered_map<int, int>& label_parent,
                          int& label_counter, const std::vector<int>& neighbors);

  std::vector<int> input_, output_;
  std::vector<int> local_labeled_image_;
  boost::mpi::communicator world_;
};

}  // namespace karaseva_e_binaryimage_mpi