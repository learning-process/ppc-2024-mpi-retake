// Anikin Maksim 2025
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_counting_characters_seq {

void create_data_vector(std::vector<char>* invec, std::string str);
void create_randdata_vector(std::vector<char>* invec, int count);


class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_1, input_2;
  int res;
};

}  // namespace anikin_m_counting_characters_seq