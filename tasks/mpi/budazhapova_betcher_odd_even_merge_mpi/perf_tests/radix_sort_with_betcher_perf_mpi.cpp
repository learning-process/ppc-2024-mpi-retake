#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {
    std::vector<int> GenerateRandomVector(int size, int minValue, int maxValue) {
        std::vector<int> randomVector(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(minValue, maxValue);
        for (int i = 0; i < size; ++i) {
            randomVector[i] = dis(gen);
        }

        return randomVector;
    }
}  // namespace budazhapova_betcher_odd_even_merge_mpi

TEST(budazhapova_betcher_odd_even_merge_mpi, test_pipeline_run) {
    boost::mpi::communicator world;
    std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(12000000, 5, 100);
    std::vector<int> out(12000000, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
        task_data_par->inputs_count.emplace_back(input_vector.size());
        task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
        task_data_par->outputs_count.emplace_back(out.size());
    }

    auto test_mpi_task_parallel = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(task_data_par);
    ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
    test_mpi_task_parallel->PreProcessing();
    test_mpi_task_parallel->Run();
    test_mpi_task_parallel->Post_Processing();

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
    perf_analyzer->pipeline_run(perf_attr, perf_results);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perf_results);
    }
}
TEST(budazhapova_betcher_odd_even_merge_mpi, test_task_run) {
    boost::mpi::communicator world;
    std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(12000000, 5, 100);
    std::vector<int> out(12000000, 0);
    std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
        task_data_par->inputs_count.emplace_back(input_vector.size());
        task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
        task_data_par->outputs_count.emplace_back(out.size());
    }

    auto test_mpi_task_parallel = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(task_data_par);
    ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
    test_mpi_task_parallel->PreProcessing();
    test_mpi_task_parallel->Run();
    test_mpi_task_parallel->PostProcessing();

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const boost::mpi::timer current_timer;
    perf_attr->current_timer = [&] { return current_timer.elapsed(); };
    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
    perf_analyzer->task_run(perf_attr, perf_results);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perf_results);
    }
}