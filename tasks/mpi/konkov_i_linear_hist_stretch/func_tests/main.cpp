#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdlib>

#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

TEST(konkov_i_LinearHistStretchTest, ValidImageData) {
  const int image_size = 100;
  int image_data[image_size];
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = rand() % 256;  // Random values between 0 and 255
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());
}

TEST(konkov_i_LinearHistStretchTest, InvalidImageData) {
  int image_size = 0;
  int* image_data = nullptr;

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_FALSE(lht.Validation());
}

TEST(konkov_i_LinearHistStretchTest, AllPixelsSameValueMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int image_size = 100;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = 128;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (rank == 0) {
    ASSERT_TRUE(lht.Validation());
  }

  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  if (rank == 0) {
    for (int i = 0; i < image_size; ++i) {
      EXPECT_EQ(image_data[i], 128);
    }
    delete[] image_data;
  }
}

TEST(konkov_i_LinearHistStretchTest, NegativeValuesMPI) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int image_size = 100;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = -100 + i;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  if (rank == 0) {
    for (int i = 0; i < image_size; ++i) {
      EXPECT_GE(image_data[i], 0);
      EXPECT_LE(image_data[i], 255);
    }
    delete[] image_data;
  }
}

TEST(konkov_i_LinearHistStretchTest, SinglePixelImage) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int image_size = 1;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    image_data[0] = 50;
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (rank == 0) {
    ASSERT_TRUE(lht.Validation());
  }

  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  if (rank == 0) {
    EXPECT_GE(image_data[0], 0);
    EXPECT_LE(image_data[0], 255);
    delete[] image_data;
  }
}