#include "gtest/gtest.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#include "mpi/markin_i_rectangle_method/include/ops_mpi.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/perf/include/perf.hpp"

TEST(markin_i_rectangle_mpi, test_pipeline_run) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    int rank = world.rank();


    double a = 0.0;
    double b = 1.0;
    int n = 1000000;
    double global_sum = 0.0;

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();


    if (rank == 0) {
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
        task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(&global_sum));
    }


    auto test_task_sequential = std::make_shared<markin_i_rectangle_mpi::RectangleMpiTask>(task_data_seq, world);


    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };


    auto perf_results = std::make_shared<ppc::core::PerfResults>();


    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);


    perf_analyzer->PipelineRun(perf_attr, perf_results);


    if (rank == 0) {
        ppc::core::Perf::PrintPerfStatistic(perf_results);
    }
}

TEST(markin_i_rectangle_mpi, test_task_run) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    int rank = world.rank();


    double a = 0.0;
    double b = 1.0;
    int n = 1000000;
    double global_sum = 0.0;


    auto task_data_seq = std::make_shared<ppc::core::TaskData>();!


    if (rank == 0) {
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
        task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(&global_sum));
    }

    auto test_task_sequential = std::make_shared<markin_i_rectangle_mpi::RectangleMpiTask>(task_data_seq, world);


    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };


    auto perf_results = std::make_shared<ppc::core::PerfResults>();


    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);


    perf_analyzer->TaskRun(perf_attr, perf_results);

    if (rank == 0) {
        ppc::core::Perf::PrintPerfStatistic(perf_results);
    }
}