#include "gtest/gtest.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#include "mpi/markin_i_rectangle_method/include/ops_mpi.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/perf/include/perf.hpp"

namespace markin_i_rectangle_mpi {

TEST(rectangle_mpi, simple_test_correctness) {
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

    RectangleMpiTask test_task_sequential(task_data_seq, world);

    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();

    if (rank == 0) {
        double analytic_result = 1.0/3.0;
        double relative_error = std::abs(analytic_result - global_sum)/analytic_result;
        std::cout << "Rank: " << rank << " global_sum = " << global_sum << std::endl;
        ASSERT_NEAR(global_sum, analytic_result, 1e-6);
        std::cout << "Relative error: " << relative_error << std::endl;
    }
}
TEST(rectangle_mpi, negativ_test_correctness) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    int rank = world.rank();

    double a = -5.0;
    double b = -3.0;
    int n = 1000000;
    double global_sum = 0.0;

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

    if (rank == 0) {
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
        task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
        task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(&global_sum));
    }

    RectangleMpiTask test_task_sequential(task_data_seq, world);

    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();

    if (rank == 0) {
        double analytic_result = 98.0/3.0;
        double relative_error = std::abs(analytic_result - global_sum)/analytic_result;
        std::cout << "Rank: " << rank << " global_sum = " << global_sum << std::endl;
        ASSERT_NEAR(global_sum, analytic_result, 1e-6);
        std::cout << "Relative error: " << relative_error << std::endl;
    }
}}