#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
// #include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_shtrassen_algorithm_mpi {

struct Parametre {
  std::vector<double> a;
  std::vector<double> b;
};

class StrassenAlgoriphmSequential : public ppc::core::Task {
 public:
  explicit StrassenAlgoriphmSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;
};

class StrassenAlgoriphmParallel : public ppc::core::Task {
 public:
  explicit StrassenAlgoriphmParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> StrassenMpi(const Parametre& param, size_t n);

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;

  boost::mpi::communicator world_;
};

std::vector<double> Add(const std::vector<double>& a, const std::vector<double>& b, size_t n);
std::vector<double> Subtract(const std::vector<double>& a, const std::vector<double>& b, size_t n);
std::vector<double> StrassenSeq(const std::vector<double>& a, const std::vector<double>& b, size_t n);
}  // namespace dudchenko_o_shtrassen_algorithm_mpi