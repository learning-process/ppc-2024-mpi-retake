#include <gtest/gtest.h>

#include "seq/konkov_i_linear_hist_stretch/include/ops_seq.hpp"

TEST(konkov_i_LinearHistStretchTest, ValidImageData) {
  const int image_size = 100;
  int image_data[image_size];

  for (int i = 0; i < image_size; ++i) {
    image_data[i] = rand() % 256;
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  for (int i = 0; i < image_size; ++i) {
    EXPECT_GE(image_data[i], 0);
    EXPECT_LE(image_data[i], 255);
  }
}

TEST(konkov_i_LinearHistStretchTest, InvalidImageData) {
  int image_size = 0;
  int* image_data = nullptr;

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_FALSE(lht.Validation());
}

TEST(konkov_i_LinearHistStretchTest, AllPixelsSameValueSeq) {
  const int image_size = 100;
  int image_data[image_size];

  for (int i = 0; i < image_size; ++i) {
    image_data[i] = 128;
  }
  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  for (int i = 0; i < image_size; ++i) {
    EXPECT_EQ(image_data[i], 128);
  }
}

TEST(konkov_i_LinearHistStretchTest, NegativeValuesSeq) {
  const int image_size = 100;
  int image_data[image_size];

  for (int i = 0; i < image_size; ++i) {
    image_data[i] = -100 + i;
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  for (int i = 0; i < image_size; ++i) {
    EXPECT_GE(image_data[i], 0);
    EXPECT_LE(image_data[i], 255);
  }
}

TEST(konkov_i_LinearHistStretchTest, SinglePixelImageSeq) {
  const int image_size = 1;
  int image_data[image_size];
  image_data[0] = 50;

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.Validation());
  ASSERT_TRUE(lht.PreProcessing());
  ASSERT_TRUE(lht.Run());
  ASSERT_TRUE(lht.PostProcessing());

  EXPECT_GE(image_data[0], 0);
  EXPECT_LE(image_data[0], 255);
}