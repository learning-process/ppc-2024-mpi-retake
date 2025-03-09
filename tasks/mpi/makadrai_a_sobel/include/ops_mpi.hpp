#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makadrai_a_sobel_mpi {

class Sobel : public ppc::core::Task {
 public:
  explicit Sobel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int height_img_;
  int width_img_;
  int peding_ = 2;

  std::vector<int> img_;
  std::vector<int> simg_;
  boost::mpi::communicator world_;
};

class SobelSeq : public ppc::core::Task {
public:
  explicit SobelSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  int height_img_;
  int width_img_;
  int peding_ = 2;

  std::vector<int> img_;
  std::vector<int> simg_;
  boost::mpi::communicator world_;
};

}  // namespace makadrai_a_sobel_mpi