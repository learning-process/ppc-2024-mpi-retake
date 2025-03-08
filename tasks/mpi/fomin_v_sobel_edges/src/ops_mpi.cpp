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

    // Проверка минимального размера изображения
    if (width_ < 3 || height_ < 3) {
      std::cerr << "Error [Rank 0]: Image must be at least 3x3 pixels" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned char *input_data = reinterpret_cast<unsigned char *>(task_data->inputs[0]);
    input_image_.assign(input_data, input_data + width_ * height_);
  }

  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Проверка размеров после широковещательной рассылки
  if (width_ < 3 || height_ < 3) {
    if (rank == 0) std::cerr << "Error: Invalid image dimensions" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int chunk_size = height_ / size;
  int remainder = height_ % size;

  // Обработка случая, когда процессов больше чем строк
  if (chunk_size == 0 && size > height_) {
    if (rank == 0) std::cerr << "Error: Too many MPI processes for image height" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0) {
    int current_row = 0;
    for (int i = 0; i < size; ++i) {
      int rows = chunk_size + (i < remainder ? 1 : 0);
      if (current_row + rows > height_) {
        std::cerr << "Error: Invalid row distribution" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      if (i == 0) {
        local_data.assign(input_image_.begin() + current_row * width_,
                          input_image_.begin() + (current_row + rows) * width_);
      } else {
        MPI_Send(input_image_.data() + current_row * width_, rows * width_, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
      }
      current_row += rows;
    }
  } else {
    int recv_rows = chunk_size + (rank < remainder ? 1 : 0);
    local_data.resize(recv_rows * width_);
    MPI_Status status;
    MPI_Recv(local_data.data(), recv_rows * width_, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);

    // Проверка полученных данных
    int received;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &received);
    if (received != recv_rows * width_) {
      std::cerr << "Error [Rank " << rank << "]: Data mismatch. Received: " << received
                << " Expected: " << recv_rows * width_ << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Инициализация ghost-зон с проверками
  if (size > 1) {
    ghost_upper.resize(width_);
    ghost_lower.resize(width_);

    if (rank > 0) {
      MPI_Sendrecv(local_data.data(), width_, MPI_UNSIGNED_CHAR, rank - 1, 0, ghost_upper.data(), width_,
                   MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
      MPI_Sendrecv(local_data.data() + (local_data.size() - width_), width_, MPI_UNSIGNED_CHAR, rank + 1, 0,
                   ghost_lower.data(), width_, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  output_image_.resize(local_data.size());
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

  const int local_height = local_data.size() / width_;
  const int start_y = (rank == 0) ? 1 : 0;
  const int end_y = (rank == size - 1) ? local_height - 1 : local_height;

  // Проверка валидности локальных данных
  if (local_height <= 0 || output_image_.size() != local_data.size()) {
    std::cerr << "Error [Rank " << rank << "]: Invalid local data" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int y = start_y; y < end_y; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;

      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int ny = y + dy;
          const int nx = x + dx;
          unsigned char pixel = 0;

          // Проверка границ
          if (ny < 0) {
            if (rank > 0 && nx >= 0 && nx < width_) {
              pixel = ghost_upper[nx];
            }
          } else if (ny >= local_height) {
            if (rank < size - 1 && nx >= 0 && nx < width_) {
              pixel = ghost_lower[nx];
            }
          } else {
            if (nx >= 0 && nx < width_) {
              const int idx = ny * width_ + nx;
              if (idx < 0 || idx >= static_cast<int>(local_data.size())) {
                std::cerr << "Error [Rank " << rank << "]: Invalid local index" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
              pixel = local_data[idx];
            }
          }

          sumX += pixel * Gx[dy + 1][dx + 1];
          sumY += pixel * Gy[dy + 1][dx + 1];
        }
      }

      const int idx = y * width_ + x;
      if (idx < 0 || idx >= static_cast<int>(output_image_.size())) {
        std::cerr << "Error [Rank " << rank << "]: Output index out of range" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      output_image_[idx] = static_cast<unsigned char>(std::min(static_cast<int>(std::hypot(sumX, sumY)), 255));
    }
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PostProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  const int send_count = output_image_.size();

  if (rank == 0) {
    output_image_.resize(width_ * height_);
    int chunk_size = height_ / size;
    int remainder = height_ % size;
    int current_displ = 0;

    for (int i = 0; i < size; ++i) {
      const int rows = chunk_size + (i < remainder ? 1 : 0);
      recv_counts[i] = rows * width_;
      displs[i] = current_displ;
      current_displ += recv_counts[i];
    }

    if (current_displ != width_ * height_) {
      std::cerr << "Error: Invalid displacement calculation" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Gatherv(output_image_.data(), send_count, MPI_UNSIGNED_CHAR, (rank == 0) ? output_image_.data() : nullptr,
              recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    if (output_image_.size() != static_cast<size_t>(width_ * height_)) {
      std::cerr << "Error: Final output size mismatch" << std::endl;
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