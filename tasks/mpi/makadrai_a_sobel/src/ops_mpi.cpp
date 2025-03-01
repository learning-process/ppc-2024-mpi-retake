#include "mpi/makadrai_a_sobel/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

bool makadrai_a_sobel_mpi::Sobel::PreProcessingImpl() {
  if (world.rank() == 0) {
    height_img = task_data->inputs_count[1];
    width_img = task_data->inputs_count[0];
  }

  boost::mpi::broadcast(world, height_img, 0);
  boost::mpi::broadcast(world, width_img, 0);

  if (world.rank() == 0) {
    img.resize((width_img + peding) * (height_img + peding));
    simg.resize(width_img * height_img, 0);
    const auto* in = reinterpret_cast<size_t*>(task_data->inputs[0]);

    for (size_t i = 0; i < height_img; i++) {
      std::copy(in + (i * width_img), in + ((i + 1) * width_img), img.begin() + ((i + 1) * (width_img + peding) + 1));
    }
  }

  return true;
}

bool makadrai_a_sobel_mpi::Sobel::ValidationImpl() {
  bool status = false;
  if (world.rank() == 0) {
    status = task_data->outputs_count[0] == task_data->inputs_count[0] &&
             task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[0] > 0 &&
             task_data->inputs_count[1] > 0;
  }
  boost::mpi::broadcast(world, status, 0);
  return status;
}

bool makadrai_a_sobel_mpi::Sobel::RunImpl() {
  int del = height_img;
  int ost = height_img;

  if (world.size() != 1) {
    del = height_img / (world.size() - 1);
    ost = height_img % (world.size() - 1);
  }

  std::vector<int> send_counts(world.size(), (del + peding) * (width_img + peding));
  std::vector<int> send_counts_res(world.size(), del * width_img);
  if (ost != 0) {
    send_counts[0] = (ost + peding) * (width_img + peding);
    send_counts_res[0] = ost * width_img;
  } else {
    send_counts[0] = 0;
    send_counts_res[0] = 0;
  }

  std::vector<int> displacements_res(world.size());
  std::vector<int> displacements(world.size());
  if (world.size() != 1) {
    for (int i = 1; i < world.size(); i++) {
      displacements[i] = (i - 1) * del * (width_img + peding) + ost * (width_img + peding);
      displacements_res[i] = (i - 1) * del * width_img + ost * width_img;
    }
  }

  size_t local_height_img = (world.rank() == 0) ? ost : del;
  std::vector<size_t> local_img(send_counts[world.rank()]);
  std::vector<size_t> local_simg(local_height_img * width_img);
  int local_max_z = 1;

  boost::mpi::scatterv(world, img, send_counts, displacements, local_img.data(), send_counts[world.rank()], 0);

  if (local_height_img != 0) {
    for (size_t i = 1; i < local_height_img + 1; i++) {
      for (size_t j = 1; j < width_img + 1; j++) {
        int G_x = -1 * local_img[(i - 1) * (width_img + peding) + (j - 1)] +
                  -1 * local_img[(i - 1) * (width_img + peding) + (j + 1)] +
                  -2 * local_img[(i - 1) * (width_img + peding) + j] +
                  2 * local_img[(i + 1) * (width_img + peding) + j] +
                  1 * local_img[(i + 1) * (width_img + peding) + (j - 1)] +
                  1 * local_img[(i + 1) * (width_img + peding) + (j + 1)];

        int G_y = 1 * local_img[(i - 1) * (width_img + peding) + (j - 1)] +
                  2 * local_img[i * (width_img + peding) + (j - 1)] +
                  1 * local_img[(i + 1) * (width_img + peding) + (j - 1)] +
                  -1 * local_img[(i - 1) * (width_img + peding) + (j + 1)] +
                  -2 * local_img[i * (width_img + peding) + (j + 1)] +
                  -1 * local_img[(i + 1) * (width_img + peding) + (j + 1)];

        int temp = std::sqrt(std::pow(G_x, 2) + std::pow(G_y, 2));
        local_max_z = std::max(local_max_z, temp);
        local_simg[(i - 1) * width_img + (j - 1)] = temp;
      }
    }
  }

  int max_z = 1;

  boost::mpi::all_reduce(world, local_max_z, max_z, boost::mpi::maximum<int>());

  boost::mpi::gatherv(world, local_simg.data(), local_height_img * width_img, simg.data(), send_counts_res,
                      displacements_res, 0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < width_img; i++) {
      for (size_t j = 0; j < height_img; j++) {
        simg[i * height_img + j] = ((double)simg[i * height_img + j] / max_z) * 255;
      }
    }
  }

  return true;
}

bool makadrai_a_sobel_mpi::Sobel::PostProcessingImpl() {
  if (world.rank() == 0) {
    std::copy(simg.begin(), simg.end(), reinterpret_cast<size_t*>(task_data->outputs[0]));
  }
  return true;
}
