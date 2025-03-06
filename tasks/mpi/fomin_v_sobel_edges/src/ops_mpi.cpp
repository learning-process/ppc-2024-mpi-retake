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
    input_image_ = *reinterpret_cast<std::vector<unsigned char> *>(task_data->inputs[0]);
    width_ = task_data->inputs_count[0];
    height_ = task_data->inputs_count[1];
  }

  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);

  int delta_height = height_ / world.size();
  local_height_ = delta_height;
  if (world.rank() == world.size() - 1) {
    local_height_ = height_ - delta_height * (world.size() - 1);
  }

  local_input_image_.resize((local_height_ + 2) * width_, 0);
  local_output_image_.resize(local_height_ * width_, 0);

  std::vector<int> send_counts(world.size(), delta_height * width_);
  std::vector<int> displacements(world.size(), 0);
  for (int proc = 1; proc < world.size(); ++proc) {
    displacements[proc] = proc * delta_height * width_;
  }

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, input_image_.data(), send_counts, displacements, local_input_image_.data() + width_,
                         delta_height * width_, 0);
  } else {
    boost::mpi::scatterv(world, 0, local_input_image_.data() + width_, delta_height * width_);
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::ValidationImpl() {
  bool is_valid = true;
  if (world.rank() == 0) {
    is_valid = task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 2;
    if (is_valid) {
      int width = task_data->inputs_count[0];
      int height = task_data->inputs_count[1];
      is_valid = (width > 0) && (height > 0);
    }
  }
  boost::mpi::broadcast(world, is_valid, 0);
  return is_valid;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::RunImpl() {
  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  if (world.rank() > 0) {
    if (world.rank() % 2 == 0) {
      world.send(world.rank() - 1, 0, local_input_image_.data() + width_, width_);
      world.recv(world.rank() - 1, 0, local_input_image_.data(), width_);
    } else {
      world.recv(world.rank() - 1, 0, local_input_image_.data(), width_);
      world.send(world.rank() - 1, 0, local_input_image_.data() + width_, width_);
    }
  }

  if (world.rank() < world.size() - 1) {
    if (world.rank() % 2 == 0) {
      world.send(world.rank() + 1, 0, local_input_image_.data() + local_height_ * width_, width_);
      world.recv(world.rank() + 1, 0, local_input_image_.data() + (local_height_ + 1) * width_, width_);
    } else {
      world.recv(world.rank() + 1, 0, local_input_image_.data() + (local_height_ + 1) * width_, width_);
      world.send(world.rank() + 1, 0, local_input_image_.data() + local_height_ * width_, width_);
    }
  }

  for (int y = 1; y < local_height_ + 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0;
      int sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = local_input_image_[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      local_output_image_[(y - 1) * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::PostProcessingImpl() {
  std::vector<int> recv_counts(world.size(), 0);
  int delta_height = height_ / world.size();
  for (int i = 0; i < world.size(); ++i) {
    recv_counts[i] =
        (i == world.size() - 1) ? (height_ - delta_height * (world.size() - 1)) * width_ : delta_height * width_;
  }

  std::vector<int> displacements(world.size(), 0);
  for (int proc = 1; proc < world.size(); ++proc) {
    displacements[proc] = displacements[proc - 1] + recv_counts[proc - 1];
  }

  if (world.rank() == 0) {
    output_image_.resize(width_ * height_);
  }

  boost::mpi::gatherv(world, local_output_image_.data(), local_output_image_.size(), output_image_.data(), recv_counts,
                      displacements, 0);

  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<unsigned char> *>(task_data->outputs[0]) = output_image_;
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