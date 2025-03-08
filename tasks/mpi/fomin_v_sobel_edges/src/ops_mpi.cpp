#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    if (task_data->inputs.empty() || !task_data->inputs[0]) {
      std::cerr << "Error [Rank 0]: No input data provided." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    width_ = task_data->inputs_count[0];
    height_ = task_data->inputs_count[1];
    unsigned char *input_data = reinterpret_cast<unsigned char *>(task_data->inputs[0]);
    if (!input_data) {
      std::cerr << "Error [Rank 0]: Input data pointer is null." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    input_image_.assign(input_data, input_data + width_ * height_);

    if (input_image_.size() != static_cast<size_t>(width_ * height_)) {
      std::cerr << "Error [Rank 0]: Input image size does not match dimensions." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_CHECK(MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD));

  if (width_ <= 0 || height_ <= 0) {
    if (rank == 0) {
      std::cerr << "Error [Rank 0]: Invalid image dimensions: " << width_ << "x" << height_ << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int chunk_size = height_ / size;
  int remainder = height_ % size;

  if (rank == 0) {
    int root_rows = chunk_size + (remainder > 0 ? 1 : 0);
    local_data.resize(root_rows * width_);
    if (root_rows * width_ > static_cast<int>(input_image_.size())) {
      std::cerr << "Error [Rank 0]: Invalid root rows calculation." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::copy(input_image_.begin(), input_image_.begin() + root_rows * width_, local_data.begin());

    for (int i = 1; i < size; ++i) {
      int start_row = chunk_size * i + std::min(i, remainder);
      int send_rows = chunk_size + (i < remainder ? 1 : 0);
      int send_size = send_rows * width_;
      if (start_row + send_rows > height_) {
        std::cerr << "Error [Rank 0]: Invalid send range for rank " << i << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      MPI_CHECK(MPI_Send(input_image_.data() + start_row * width_, send_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD));
    }
  } else {
    int recv_rows = chunk_size + (rank < remainder ? 1 : 0);
    local_data.resize(recv_rows * width_);
    MPI_Status status;
    MPI_CHECK(MPI_Recv(local_data.data(), recv_rows * width_, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status));

    int received_count;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &received_count);
    if (received_count != recv_rows * width_) {
      std::cerr << "Error [Rank " << rank << "]: Received " << received_count << " elements, expected "
                << recv_rows * width_ << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // New checks for local data
  if (local_data.empty()) {
    std::cerr << "Error [Rank " << rank << "]: Local data is empty after distribution" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (size > 1) {
    ghost_upper.resize(width_);
    ghost_lower.resize(width_);

    if ((rank > 0 && ghost_upper.empty()) || (rank < size - 1 && ghost_lower.empty())) {
      std::cerr << "Error [Rank " << rank << "]: Ghost buffers not initialized" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank > 0) {
      MPI_CHECK(MPI_Sendrecv(local_data.data(), width_, MPI_UNSIGNED_CHAR, rank - 1, 0, ghost_upper.data(), width_,
                             MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    if (rank < size - 1) {
      MPI_CHECK(MPI_Sendrecv(local_data.data() + (local_data.size() - width_), width_, MPI_UNSIGNED_CHAR, rank + 1, 0,
                             ghost_lower.data(), width_, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE));
    }

    // Ghost buffers checks
    if (rank > 0 && ghost_upper.size() != static_cast<size_t>(width_)) {
      std::cerr << "Error [Rank " << rank << "]: Invalid ghost_upper size" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank < size - 1 && ghost_lower.size() != static_cast<size_t>(width_)) {
      std::cerr << "Error [Rank " << rank << "]: Invalid ghost_lower size" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  output_image_.resize(local_data.size(), 0);
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::ValidationImpl() {
  bool valid = true;
  if (world.rank() == 0) {
    valid = task_data->inputs_count.size() >= 2 && task_data->outputs_count.size() >= 2 &&
            task_data->inputs_count[0] > 0 && task_data->outputs_count[0] > 0 && task_data->inputs_count[1] > 0 &&
            task_data->outputs_count[1] > 0;

    // Additional consistency check
    if (valid && task_data->inputs_count[0] != task_data->outputs_count[0]) {
      std::cerr << "Error: Input and output widths mismatch" << std::endl;
      valid = false;
    }
    if (valid && task_data->inputs_count[1] != task_data->outputs_count[1]) {
      std::cerr << "Error: Input and output heights mismatch" << std::endl;
      valid = false;
    }
  }
  boost::mpi::broadcast(world, valid, 0);
  return valid;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int local_height = local_data.size() / width_;
  int start_y = (rank == 0) ? 1 : 0;
  int end_y = (rank == size - 1) ? local_height - 1 : local_height;

  if (local_data.empty() || output_image_.size() != local_data.size()) {
    std::cerr << "Error [Rank " << rank << "]: Invalid local data buffers" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Validate local dimensions
  if (local_height <= 0) {
    std::cerr << "Error [Rank " << rank << "]: Invalid local height: " << local_height << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (width_ <= 2) {
    std::cerr << "Error [Rank " << rank << "]: Image width too small for processing" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int y = start_y; y < end_y; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;

      if (y < 0 || y >= local_height || x < 0 || x >= width_) {
        std::cerr << "Error [Rank " << rank << "]: Invalid processing coordinates (" << x << "," << y << ")"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      // Boundary checks for output image
      if (y * width_ + x >= static_cast<int>(output_image_.size())) {
        std::cerr << "Error [Rank " << rank << "]: Output index out of bounds: " << y * width_ + x << " vs "
                  << output_image_.size() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int y_offset = y + i;
          int x_offset = x + j;
          unsigned char pixel = 0;

          // Validate x offset
          if (x_offset < 0 || x_offset >= width_) {
            std::cerr << "Error [Rank " << rank << "]: Invalid x offset: " << x_offset << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }

          if (y_offset < 0) {
            if (rank > 0) {
              if (x_offset >= static_cast<int>(ghost_upper.size())) {
                std::cerr << "Error [Rank " << rank << "]: ghost_upper access out of bounds" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
              pixel = ghost_upper[x_offset];
            }
          } else if (y_offset >= local_height) {
            if (rank < size - 1) {
              if (x_offset >= static_cast<int>(ghost_lower.size())) {
                std::cerr << "Error [Rank " << rank << "]: ghost_lower access out of bounds" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
              pixel = ghost_lower[x_offset];
            }
          } else {
            if (y_offset * width_ + x_offset >= static_cast<int>(local_data.size())) {
              std::cerr << "Error [Rank " << rank << "]: Local data access out of bounds" << std::endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
            pixel = local_data[y_offset * width_ + x_offset];
          }

          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::hypot(sumX, sumY));
      output_image_[y * width_ + x] = std::min(gradient, 255);
    }
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PostProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Проверка выходного буфера
  if (rank == 0 && !task_data->outputs[0]) {
    std::cerr << "Error [Rank 0]: Output buffer is null" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int send_count = output_image_.size();

  // Проверка размера данных для отправки
  if (send_count <= 0) {
    std::cerr << "Error [Rank " << rank << "]: Invalid send count: " << send_count << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::vector<int> recv_counts;
  std::vector<int> displs;

  if (rank == 0) {
    // Проверка согласованности размеров
    if (width_ * height_ != static_cast<int>(output_image_.size())) {
      std::cerr << "Error [Rank 0]: Output image size mismatch. Expected: " << width_ * height_
                << " Actual: " << output_image_.size() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    output_image_.resize(width_ * height_);
    recv_counts.resize(size);
    displs.resize(size);

    int chunk_size = height_ / size;
    int remainder = height_ % size;
    int current_displ = 0;

    for (int i = 0; i < size; ++i) {
      int rows = chunk_size + (i < remainder ? 1 : 0);
      recv_counts[i] = rows * width_;

      // Проверка валидности смещений
      if (current_displ + recv_counts[i] > width_ * height_) {
        std::cerr << "Error [Rank 0]: Displacement overflow. Current: " << current_displ
                  << " Adding: " << recv_counts[i] << " Total size: " << width_ * height_ << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      displs[i] = current_displ;
      current_displ += recv_counts[i];
    }
  }

  // Проверка буфера перед Gatherv
  if (rank == 0 && output_image_.size() < static_cast<size_t>(width_ * height_)) {
    std::cerr << "Error [Rank 0]: Output buffer too small for Gatherv" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int total = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  if (total != width_ * height_) {
    std::cerr << "Error [Rank 0]: Gatherv total mismatch. Expected: " << width_ * height_ << " Actual: " << total
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Gatherv(output_image_.data(), send_count, MPI_UNSIGNED_CHAR, (rank == 0) ? output_image_.data() : nullptr,
              recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Проверка финального размера
    if (output_image_.size() != static_cast<size_t>(width_ * height_)) {
      std::cerr << "Error [Rank 0]: Final output size mismatch. Expected: " << width_ * height_
                << " Actual: " << output_image_.size() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Проверка выходного буфера перед копированием
    if (!task_data->outputs[0]) {
      std::cerr << "Error [Rank 0]: Output buffer became null after processing" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned char *output_buffer = reinterpret_cast<unsigned char *>(task_data->outputs[0]);
    std::copy(output_image_.begin(), output_image_.end(), output_buffer);
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::PreProcessingImpl() {
  input_image_ = *reinterpret_cast<std::vector<unsigned char> *>(task_data->inputs[0]);
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
  unsigned char *output_buffer = reinterpret_cast<unsigned char *>(task_data->outputs[0]);
  std::copy(output_image_.begin(), output_image_.end(), output_buffer);
  return true;
}