#pragma once

#include <string>
#include <utility>

#include "core/task/include/task.hpp"


namespace vasenkov_a_word_count_seq {

class WordCountSequential : public ppc::core::Task {
 public:
  explicit WordCountSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string inputString_;
  int stringSize_, wordCount_;
};

}  // namespace vasenkov_a_word_count_seq