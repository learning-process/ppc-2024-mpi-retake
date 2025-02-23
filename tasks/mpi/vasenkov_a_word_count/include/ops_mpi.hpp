#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_word_count_mpi {

class WordCountMPI : public ppc::core::Task {
 public:
  explicit WordCountMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int stringSize_, wordCount_, wordLoaclCount_;
  std::string inputString_;
  boost::mpi::communicator world_;
};

}  // namespace vasenkov_a_word_count_mpi