// Anikin Maksim
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_counting_characters_mpi {

void CreateDataVector(std::vector<char>* invec, const std::string& str);
void CreateRanddataVector(std::vector<char>* invec, int count);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_1_, input_2_;
  int res_;
  boost::mpi::communicator world_;
};

}  // namespace anikin_m_counting_characters_mpi