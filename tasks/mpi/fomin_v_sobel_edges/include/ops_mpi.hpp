#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_sobel_edges {

class SobelEdgeDetectionMPI : public ppc::core::Task {
 public:
  explicit SobelEdgeDetectionMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int height_;
  int width_;
  std::vector<unsigned char> input_image_;
  std::vector<unsigned char> local_data;
  std::vector<unsigned char> ghost_upper;
  std::vector<unsigned char> ghost_lower;
  std::vector<unsigned char> output_image_;
  boost::mpi::communicator world;
};

class SobelEdgeDetection : public ppc::core::Task {
 public:
  explicit SobelEdgeDetection(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<unsigned char> input_image_;
  std::vector<unsigned char> output_image_;
  int height_;
  int width_;
};
}  // namespace fomin_v_sobel_edges