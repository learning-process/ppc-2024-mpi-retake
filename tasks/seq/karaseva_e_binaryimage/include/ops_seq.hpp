#pragma once

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int GetRootLabel(std::unordered_map<int, int>& label_parent, int label);
  void UnionLabels(std::unordered_map<int, int>& label_parent, int label1, int label2);
  void Labeling(std::vector<int>& image, std::vector<int>& labeled_image, int rows, int cols, int min_label,
                std::unordered_map<int, int>& label_parent, int x, int y);

  std::vector<int> input_;
  std::vector<int> output_;
  int rc_size_{};
  std::map<int, int> label_equivalence_;
  std::vector<int> image_;
  std::vector<int> labeled_image_;
  int rows_;
  int columns_;
};

}  // namespace karaseva_e_binaryimage_seq