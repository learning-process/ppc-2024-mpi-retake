#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

TEST(konkov_i_LinearHistStretchPerformance, StretchLargeImage) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int image_size = 1000000;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = rand() % 256;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (!lht.Validation()) {
    if (rank == 0) {
      delete[] image_data;
    }
    return;
  }

  if (!lht.PreProcessing()) {
    if (rank == 0) {
      delete[] image_data;
    }
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  lht.Run();
  auto end = std::chrono::high_resolution_clock::now();

  if (!lht.PostProcessing()) {
    if (rank == 0) {
      delete[] image_data;
    }
    return;
  }

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    std::cout << "Stretching execution time for image size " << image_size << ": " << elapsed.count() << " seconds"
              << '\n';
    delete[] image_data;
  }
}