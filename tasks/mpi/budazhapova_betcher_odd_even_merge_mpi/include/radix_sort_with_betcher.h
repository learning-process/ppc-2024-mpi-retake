#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {

	class MergeSequential : public ppc::core::Task {
	public:
		explicit MergeSequential(std::shared_ptr<ppc::core::TaskData> task_data_) : Task(std::move(task_data_)) {}
		bool PreProcessing() override;
		bool Validation() override;
		bool Run() override;
		bool PostProcessing() override;

	private:
		std::vector<int> res_;
		std::vector<int> local_res_;
		int n_el_ = 0;
	};
	class MergeParallel : public ppc::core::Task {
	public:
		explicit MergeParallel(std::shared_ptr<ppc::core::TaskData> task_data_) : Task(std::move(task_data_)) {}
		bool PreProcessing() override;
		bool Validation() override;
		bool Run() override;
		bool PostProcessing() override;

	private:
		std::vector<int> res_;
		std::vector<int> local_res_;
		std::vector<int> fin_res_;
		int n_el_ = 0;

		boost::mpi::communicator world;
	};
}  // namespace budazhapova_betcher_odd_even_merge_mpi
