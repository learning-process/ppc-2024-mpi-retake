#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/Konstantinov_I_histogram_stretching/include/ops_mpi.hpp"

namespace konstantinov_i_linear_histogram_stretch_mpi {
namespace {
std::vector<int> GetRandomImage(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace
}  // namespace konstantinov_i_linear_histogram_stretch_mpi

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_image_imin_imax) {
  boost::mpi::communicator world;

  const int count_size_vector = 9;
  std::vector<int> in_vec = {255, 100, 25, 255, 100, 25, 255, 100, 25};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, in_vec);
    ASSERT_EQ(out_vec_seq, in_vec);
    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_correct_image) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 150;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_large_image) {
  boost::mpi::communicator world;

  const int width = 422;
  const int height = 763;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_large_image2) {
  boost::mpi::communicator world;

  const int width = 1024;
  const int height = 512;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_large_square_image) {
  boost::mpi::communicator world;

  const int width = 680;
  const int height = 680;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_small_square_image) {
  boost::mpi::communicator world;

  const int width = 64;
  const int height = 64;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> out_vec_seq(count_size_vector, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_seq->inputs_count.emplace_back(in_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_vec_seq.size());

    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq test_seq(task_data_seq);
    ASSERT_EQ(test_seq.ValidationImpl(), true);
    test_seq.PreProcessingImpl();
    test_seq.RunImpl();
    test_seq.PostProcessingImpl();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_incorrect_image_size) {
  boost::mpi::communicator world;

  const int count_size_vector = 5;
  std::vector<int> in_vec = {0, 20, 30, 0, 40};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
    ASSERT_EQ(test_mpi.ValidationImpl(), false);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_incorrect_rgb_image) {
  boost::mpi::communicator world;

  const int width = 3;
  const int height = 3;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {-2, 20, 30, 0, 355, -25, 10, 260, 255};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
    ASSERT_EQ(test_mpi.ValidationImpl(), false);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_incorrect_rgb_image2) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 3;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {-2, 20, 30, 0, 355, -25, 10, 260, 255, -2, 20, 30, 22, 33, 44, 72, 89, 90};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
    ASSERT_EQ(test_mpi.ValidationImpl(), false);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_incorrect_rgb_image3) {
  boost::mpi::communicator world;

  const int width = 70;
  const int height = 15;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
  in_vec[5] = -25;
  in_vec[17] = 266;
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
    ASSERT_EQ(test_mpi.ValidationImpl(), false);
  }
}

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_incorrect_empty_image) {
  boost::mpi::communicator world;

  const int width = 0;
  const int height = 0;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
    konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi test_mpi(task_data_par);
    ASSERT_EQ(test_mpi.ValidationImpl(), false);
  }
}