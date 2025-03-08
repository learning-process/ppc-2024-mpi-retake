#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

#define MPI_CHECK(call)                                                                                         \
  do {                                                                                                          \
    int mpi_errno = (call);                                                                                     \
    if (mpi_errno != MPI_SUCCESS) {                                                                             \
      char error_string[MPI_MAX_ERROR_STRING];                                                                  \
      int error_len;                                                                                            \
      MPI_Error_string(mpi_errno, error_string, &error_len);                                                    \
      std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << " [Rank " << rank << "]: " << error_string \
                << std::endl;                                                                                   \
      MPI_Abort(MPI_COMM_WORLD, mpi_errno);                                                                     \
    }                                                                                                           \
  } while (0)

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    LoadImageData();
    GenerateProcessingGrid();
  }
  DistributeComputation();
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::ValidationImpl() {
  int valid = 1;
  if (rank == 0) {
    valid = fomin_v_sobel_edges::SobelEdgeDetection::ValidationImpl() ? 1 : 0;
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid == 1;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::RunImpl() {
  ComputeEdgePixels();
  CollectResults();
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PostProcessingImpl() {
  if (rank == 0) {
    ExportProcessedImage();
  }
  return true;
}

// Private methods
void fomin_v_sobel_edges::SobelEdgeDetectionMPI::LoadImageData() {
  fomin_v_sobel_edges::SobelEdgeDetection::PreProcessingImpl();
  pixel_y.clear();
  pixel_x.clear();
  for (int y = 1; y < height_ - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      pixel_y.push_back(y);
      pixel_x.push_back(x);
    }
  }
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::GenerateProcessingGrid() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_pixels = pixel_y.size();
  counts.resize(size, total_pixels / size);
  displs.resize(size);

  int remainder = total_pixels % size;
  for (int i = 0, offset = 0; i < size; ++i) {
    if (i < remainder) counts[i]++;
    displs[i] = offset;
    offset += counts[i];
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI::CalculateImageSections();
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::CalculateImageSections() {
  sections_sizes.resize(size);
  sections_displs.resize(size);

  for (int i = 0; i < size; ++i) {
    if (counts[i] == 0) {
      sections_sizes[i] = 0;
      continue;
    }

    int start = displs[i];
    int end = start + counts[i];
    auto minmax = std::minmax_element(pixel_y.begin() + start, pixel_y.begin() + end);

    int y_start = std::max(*minmax.first - 1, 0);
    int y_end = std::min(*minmax.second + 1, height_ - 1);
    sections_sizes[i] = (y_end - y_start + 1) * width_;
    sections_displs[i] = y_start * width_;
  }
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::DistributeComputation() {
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Scatter pixel coordinates
  MPI_Scatterv(pixel_y.data(), counts.data(), displs.data(), MPI_INT, local_y.data(), local_count, MPI_INT, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(pixel_x.data(), counts.data(), displs.data(), MPI_INT, local_x.data(), local_count, MPI_INT, 0,
               MPI_COMM_WORLD);

  // Scatter image sections
  MPI_Scatterv(input_image_.data(), sections_sizes.data(), sections_displs.data(), MPI_UNSIGNED_CHAR,
               local_section.data(), local_section_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::ComputeEdgePixels() {
  constexpr int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  constexpr int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  results.resize(local_count);
  indices.resize(local_count);

  for (int i = 0; i < local_count; ++i) {
    int y = local_y[i];
    int x = local_x[i];
    int sumX = 0, sumY = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int pos = (y + dy) * width_ + (x + dx) - sections_displs[rank];
        sumX += local_section[pos] * Gx[dy + 1][dx + 1];
        sumY += local_section[pos] * Gy[dy + 1][dx + 1];
      }
    }

    results[i] = static_cast<unsigned char>(std::min((int)std::hypot(sumX, sumY), 255));
    indices[i] = y * width_ + x;
  }
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::CollectResults() {
  MPI_Gatherv(results.data(), local_count, MPI_UNSIGNED_CHAR, output_image_.data(), counts.data(), displs.data(),
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  MPI_Gatherv(indices.data(), local_count, MPI_INT, global_indices.data(), counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    for (size_t i = 0; i < global_indices.size(); ++i) {
      output_image_[global_indices[i]] = results[i];
    }
  }
}

void fomin_v_sobel_edges::SobelEdgeDetectionMPI::ExportProcessedImage() {
  unsigned char* output_buffer = reinterpret_cast<unsigned char*>(task_data->outputs[0]);
  std::copy(output_image_.begin(), output_image_.end(), output_buffer);
}

bool fomin_v_sobel_edges::SobelEdgeDetection::PreProcessingImpl() {
  input_image_ = *reinterpret_cast<std::vector<unsigned char>*>(task_data->inputs[0]);
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  output_image_.resize(width_ * height_, 0);
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 2 &&
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::RunImpl() {
  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int y = 1; y < height_ - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = input_image_[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      output_image_[y * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::PostProcessingImpl() {
  unsigned char* output_buffer = reinterpret_cast<unsigned char*>(task_data->outputs[0]);
  std::copy(output_image_.begin(), output_image_.end(), output_buffer);
  return true;
}