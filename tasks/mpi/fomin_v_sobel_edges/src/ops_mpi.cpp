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

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    width_ = task_data->inputs_count[0];
    height_ = task_data->inputs_count[1];
    if (width_ <= 0 || height_ <= 0) {
      throw std::runtime_error("Invalid image dimensions");
    }
    input_image_.resize(width_ * height_);
    std::copy(static_cast<unsigned char *>(task_data->inputs[0]),
              static_cast<unsigned char *>(task_data->inputs[0]) + width_ * height_, input_image_.begin());
  }

  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);

  const int num_procs = world.size();
  const int delta_height = height_ / num_procs;
  local_height_ = (world.rank() == num_procs - 1) ? height_ - delta_height * (num_procs - 1) : delta_height;
  local_height_ = std::max(0, local_height_);

  if (local_height_ > 0) {
    local_input_image_.resize((local_height_ + 2) * width_, 0);
    local_output_image_.resize(local_height_ * width_, 0);
  }

  if (world.rank() == 0) {
    std::vector<int> send_counts(num_procs, delta_height * width_);
    std::vector<int> displacements(num_procs, 0);
    send_counts.back() = local_height_ * width_;

    for (int i = 1; i < num_procs; ++i) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    boost::mpi::scatterv(world, input_image_.data(), send_counts, displacements,
                         local_height_ > 0 ? local_input_image_.data() + width_ : nullptr, local_height_ * width_, 0);
  } else if (local_height_ > 0) {
    boost::mpi::scatterv(world, local_input_image_.data() + width_, local_height_ * width_, 0);
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::ValidationImpl() {
  bool valid = true;
  if (world.rank() == 0) {
    valid = task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 2 &&
            task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs[0] != nullptr &&
            task_data->outputs[0] != nullptr;
  }
  boost::mpi::broadcast(world, valid, 0);
  return valid;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::RunImpl() {
  if (local_height_ <= 0) return true;

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  if (world.size() > 1) {
    if (world.rank() > 0) {
      world.recv(world.rank() - 1, 0, local_input_image_.data(), width_);
      world.send(world.rank() - 1, 0, local_input_image_.data() + width_, width_);
    }
    if (world.rank() < world.size() - 1) {
      world.send(world.rank() + 1, 0, local_input_image_.data() + local_height_ * width_, width_);
      world.recv(world.rank() + 1, 0, local_input_image_.data() + (local_height_ + 1) * width_, width_);
    }
  }

  for (int y = 1; y <= local_height_; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          sumX += local_input_image_[(y + i) * width_ + (x + j)] * Gx[i + 1][j + 1];
          sumY += local_input_image_[(y + i) * width_ + (x + j)] * Gy[i + 1][j + 1];
        }
      }
      local_output_image_[(y - 1) * width_ + x] = std::min(255, static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY)));
    }
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PostProcessingImpl() {
  if (height_ == 0 || width_ == 0) return true;

  std::vector<int> recv_counts(world.size(), 0);
  const int delta = height_ / world.size();
  for (int i = 0; i < world.size(); ++i) {
    recv_counts[i] = (i == world.size() - 1) ? (height_ - delta * i) * width_ : delta * width_;
  }

  std::vector<int> displs(world.size(), 0);
  for (int i = 1; i < world.size(); ++i) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

  if (world.rank() == 0) {
    output_image_.resize(width_ * height_);
  }

  unsigned char dummy;
  unsigned char *send_buffer = local_output_image_.empty() ? &dummy : local_output_image_.data();
  size_t send_size = local_output_image_.size();

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, send_buffer, send_size, output_image_.data(), recv_counts, displs, 0);
  } else {
    boost::mpi::gatherv(world, send_buffer, send_size, 0);
  }

  if (world.rank() == 0) {
    std::copy(output_image_.begin(), output_image_.end(), static_cast<unsigned char *>(task_data->outputs[0]));
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
  *reinterpret_cast<std::vector<unsigned char> *>(task_data->outputs[0]) = output_image_;
  return true;
}