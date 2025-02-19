#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_contrast {

class ContrastTaskMPI : public ppc::core::Task {
 public:
  explicit ContrastTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_, output_;
  int rc_size_{};
  boost::mpi::communicator world_;

  void IncreaseContrast();
};

}  // namespace shuravina_o_contrast