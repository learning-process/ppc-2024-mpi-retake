#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
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

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 3550;
  const int height = 3550;

  std::vector<int> in_vec;
  const int count_size_vector = width * height * 3;
  std::vector<int> out_vec_par(count_size_vector, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  auto test_mpi =
      std::make_shared<konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi>(task_data_par);
  ASSERT_EQ(test_mpi->ValidationImpl(), true);
  test_mpi->PreProcessingImpl();
  test_mpi->RunImpl();
  test_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

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

TEST(Konstantinov_i_linear_histogram_stretch_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int width = 3550;
  const int height = 3550;

  std::vector<int> in_vec;
  const int count_size_vector = width * height * 3;
  std::vector<int> out_vec_par(count_size_vector, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = konstantinov_i_linear_histogram_stretch_mpi::GetRandomImage(count_size_vector);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    task_data_par->inputs_count.emplace_back(in_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    task_data_par->outputs_count.emplace_back(out_vec_par.size());
  }

  auto test_mpi =
      std::make_shared<konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi>(task_data_par);
  ASSERT_EQ(test_mpi->ValidationImpl(), true);
  test_mpi->PreProcessingImpl();
  test_mpi->RunImpl();
  test_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

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