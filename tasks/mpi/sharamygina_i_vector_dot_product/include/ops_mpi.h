#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_vector_dot_product_mpi {
class vector_dot_product_mpi : public ppc::core::Task {
 public:
  explicit vector_dot_product_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> v1;
  std::vector<int> v2;
  std::vector<int> local_v1;
  std::vector<int> local_v2;
  int res{};
  boost::mpi::communicator world;
  unsigned int delta;
};
}  // namespace sharamygina_i_vector_dot_product_mpi