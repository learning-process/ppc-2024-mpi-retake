#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;
  double stronginAlgorithm();
};

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;
  boost::mpi::communicator world;

  double stronginAlgorithm();
  double stronginAlgorithmParallel();
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi