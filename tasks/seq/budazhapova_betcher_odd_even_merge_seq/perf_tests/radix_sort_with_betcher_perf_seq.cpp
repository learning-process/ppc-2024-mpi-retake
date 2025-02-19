#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/budazhapova_betcher_odd_even_merge_seq/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_seq {
    std::vector<int> GenerateRandomVector(int size, int minValue, int maxValue) {
        std::vector<int> randomVector;
        randomVector.reserve(size);
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int i = 0; i < size; ++i) {
            int randomNum = std::rand() % (maxValue - minValue + 1) + minValue;
            randomVector.push_back(randomNum);
        }
        return randomVector;
    }
}  // namespace budazhapova_betcher_odd_even_merge_seq
TEST(budazhapova_betcher_odd_even_merge_seq, test_pipeline_run) {
    std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_seq::GenerateRandomVector(10000000, 5, 100);
    std::vector<int> out(10000000, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    auto test_task_sequential = std::make_shared<budazhapova_betcher_odd_even_merge_seq::MergeSequential>(task_data_seq);

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 100;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
        };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->pipeline_run(perf_attr, perf_results);
    ppc::core::Perf::print_perf_statistic(perf_results);
}

TEST(budazhapova_betcher_odd_even_merge_seq, test_task_run) {
    std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_seq::GenerateRandomVector(10000000, 5, 100);
    std::vector<int> out(10000000, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    auto test_task_sequential = std::make_shared<budazhapova_betcher_odd_even_merge_seq::MergeSequential>(task_data_seq);

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 100;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
        };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->task_run(perf_attr, perf_results);
    ppc::core::Perf::print_perf_statistic(perf_results);
}