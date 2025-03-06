#include <gtest/gtest.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"
#include "mpi/markin_i_gather/include/ops_mpi.hpp"

namespace markin_i_gather {

template <typename T>
void GenerateTestData(int size, std::vector<T>& data) {
    data.resize(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i);
    }
}

} // namespace markin_i_gather

TEST(markin_i_gather, test_gather_int_root_0) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 0;


    auto task_data = std::make_shared<ppc::core::TaskData>();


    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<int> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<int>(i);
        }

        std::vector<int> gathered_data = gather_task->GetIntRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}

TEST(markin_i_gather, test_gather_float_root_0) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 0;


    auto task_data = std::make_shared<ppc::core::TaskData>();

    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<float> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<float>(i * 1.1f);
        }

        std::vector<float> gathered_data = gather_task->GetFloatRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}

TEST(markin_i_gather, test_gather_double_root_0) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 0;


    auto task_data = std::make_shared<ppc::core::TaskData>();


    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<double> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<double>(i * 2.2);
        }

        std::vector<double> gathered_data = gather_task->GetDoubleRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}

TEST(markin_i_gather, test_gather_int_root_1) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 1;

    auto task_data = std::make_shared<ppc::core::TaskData>();

    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<int> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<int>(i);
        }

        std::vector<int> gathered_data = gather_task->GetIntRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}

TEST(markin_i_gather, test_gather_float_root_1) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 1;

    auto task_data = std::make_shared<ppc::core::TaskData>();

    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<float> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<float>(i * 1.1f);
        }

        std::vector<float> gathered_data = gather_task->GetFloatRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}

TEST(markin_i_gather, test_gather_double_root_1) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();
    int root = 1;

    auto task_data = std::make_shared<ppc::core::TaskData>();

    auto task = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data, world);
    task->SetRoot(root, world);

    ASSERT_EQ(task->Validation(), true);
    task->PreProcessing();
    task->Run();
    task->PostProcessing();

    if (rank == root) {
        markin_i_gather::MyGatherMpiTask* gather_task = dynamic_cast<markin_i_gather::MyGatherMpiTask*>(task.get());
        ASSERT_NE(gather_task, nullptr);

        std::vector<double> expected_data(size);
        for (int i = 0; i < size; ++i) {
            expected_data[i] = static_cast<double>(i * 2.2);
        }

        std::vector<double> gathered_data = gather_task->GetDoubleRecvData();

        ASSERT_EQ(gathered_data.size(), size);
        ASSERT_EQ(gathered_data, expected_data);
    }
    world.barrier();
}