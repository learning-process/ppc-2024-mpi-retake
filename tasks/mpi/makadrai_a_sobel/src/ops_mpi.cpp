#include "mpi/makadrai_a_sobel/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/operations.hpp>
#include <cmath>
#include <vector>

bool makadrai_a_sobel_mpi::Sobel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    height_img_ = (int)task_data->inputs_count[1];
    width_img_ = (int)task_data->inputs_count[0];
  }

  boost::mpi::broadcast(world_, height_img_, 0);
  boost::mpi::broadcast(world_, width_img_, 0);

  if (world_.rank() == 0) {
    img_.resize((width_img_ + peding_) * (height_img_ + peding_));
    simg_.resize(width_img_ * height_img_, 0);
    const auto* in = reinterpret_cast<int*>(task_data->inputs[0]);

    for (int i = 0; i < height_img_; i++) {
      std::copy(in + (i * width_img_), in + ((i + 1) * width_img_),
                img_.begin() + (((i + 1) * (width_img_ + peding_)) + 1));
    }
  }

  return true;
}

bool makadrai_a_sobel_mpi::Sobel::ValidationImpl() {
  bool status = false;
  if (world_.rank() == 0) {
    status = task_data->outputs_count[0] == task_data->inputs_count[0] &&
             task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[0] > 0 &&
             task_data->inputs_count[1] > 0;
  }
  boost::mpi::broadcast(world_, status, 0);
  return status;
}

bool makadrai_a_sobel_mpi::Sobel::RunImpl() {
  int del = height_img_;
  int ost = height_img_;

  if (world_.size() != 1) {
    del = height_img_ / (world_.size() - 1);
    ost = height_img_ % (world_.size() - 1);
  }

  std::vector<int> send_counts(world_.size(), (del + peding_) * (width_img_ + peding_));
  std::vector<int> send_counts_res(world_.size(), del * width_img_);
  if (ost != 0) {
    send_counts[0] = (ost + peding_) * (width_img_ + peding_);
    send_counts_res[0] = ost * width_img_;
  } else {
    send_counts[0] = 0;
    send_counts_res[0] = 0;
  }

  std::vector<int> displacements_res(world_.size());
  std::vector<int> displacements(world_.size());
  if (world_.size() != 1) {
    for (int i = 1; i < world_.size(); i++) {
      displacements[i] = (i - 1) * del * (width_img_ + peding_) + ost * (width_img_ + peding_);
      displacements_res[i] = (i - 1) * del * width_img_ + ost * width_img_;
    }
  }

  int local_height_img = (world_.rank() == 0) ? ost : del;
  std::vector<int> local_img(send_counts[world_.rank()]);
  std::vector<int> local_simg(local_height_img * width_img_);
  int local_max_z = 1;

  boost::mpi::scatterv(world_, img_, send_counts, displacements, local_img.data(), send_counts[world_.rank()], 0);

  if (local_height_img != 0) {
    for (int i = 1; i < local_height_img + 1; i++) {
      for (int j = 1; j < width_img_ + 1; j++) {
        int g_x = (-1 * local_img[((i - 1) * (width_img_ + peding_)) + (j - 1)]) +
                  (-1 * local_img[((i - 1) * (width_img_ + peding_)) + (j + 1)]) +
                  (-2 * local_img[((i - 1) * (width_img_ + peding_)) + j]) +
                  (2 * local_img[((i + 1) * (width_img_ + peding_)) + j]) +
                  (1 * local_img[((i + 1) * (width_img_ + peding_)) + (j - 1)]) +
                  1 * local_img[((i + 1) * (width_img_ + peding_)) + (j + 1)];

        int g_y = (1 * local_img[((i - 1) * (width_img_ + peding_)) + (j - 1)]) +
                  (2 * local_img[(i * (width_img_ + peding_)) + (j - 1)]) +
                  (1 * local_img[((i + 1) * (width_img_ + peding_)) + (j - 1)]) +
                  (-1 * local_img[((i - 1) * (width_img_ + peding_)) + (j + 1)]) +
                  (-2 * local_img[(i * (width_img_ + peding_)) + (j + 1)]) +
                  -1 * local_img[((i + 1) * (width_img_ + peding_)) + (j + 1)];

        int temp = (int)std::sqrt(std::pow(g_x, 2) + std::pow(g_y, 2));
        local_max_z = std::max(local_max_z, temp);
        local_simg[((i - 1) * width_img_) + (j - 1)] = temp;
      }
    }
  }

  int max_z = 1;

  boost::mpi::all_reduce(world_, local_max_z, max_z, boost::mpi::maximum<int>());

  boost::mpi::gatherv(world_, local_simg.data(), local_height_img * width_img_, simg_.data(), send_counts_res,
                      displacements_res, 0);

  if (world_.rank() == 0) {
    for (int i = 0; i < width_img_; i++) {
      for (int j = 0; j < height_img_; j++) {
        simg_[(i * height_img_) + j] = (int)((double)simg_[(i * height_img_) + j] / max_z) * 255;
      }
    }
  }

  return true;
}

bool makadrai_a_sobel_mpi::Sobel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(simg_, reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}
