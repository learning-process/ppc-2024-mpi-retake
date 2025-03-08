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

class SobelEdgeDetection : public ppc::core::Task {
 public:
  explicit SobelEdgeDetection(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<unsigned char> input_image_;
  std::vector<unsigned char> output_image_;
  int height_;
  int width_;
};

class SobelEdgeDetectionMPI : public SobelEdgeDetection {
 public:
  explicit SobelEdgeDetectionMPI(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // MPI variables
  int rank;
  int size;

  // Data distribution
  std::vector<int> pixel_y;
  std::vector<int> pixel_x;
  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<int> sections_sizes;
  std::vector<int> sections_displs;

  // Local data buffers
  std::vector<int> local_y;
  std::vector<int> local_x;
  std::vector<unsigned char> local_section;
  int local_count;
  int local_section_size;

  // Results handling
  std::vector<unsigned char> results;
  std::vector<int> indices;
  std::vector<int> global_indices;

  // Initialization and distribution methods
  void LoadImageData();
  void GenerateProcessingGrid();
  void CalculateImageSections();
  void DistributeComputation();
  void ComputeEdgePixels();
  void CollectResults();
  void ExportProcessedImage();
};

}  // namespace fomin_v_sobel_edges