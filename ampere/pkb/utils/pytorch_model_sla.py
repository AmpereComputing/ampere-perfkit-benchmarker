# Copyright (c) 2025, Ampere Computing LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module containing functions to run dlrm

"""
import logging
from absl import flags
from perfkitbenchmarker.virtual_machine import VirtualMachine
from ampere.pkb.common import download_utils
from ampere.pkb.utils import dlrm_base_utils

FLAGS = flags.FLAGS

INSTALL_DIR = download_utils.INSTALL_DIR


class SlaRunModel:
    """
    Run DLRM model for SLA
    """

    benchmark: str
    model: str
    vm: VirtualMachine
    aml_dir: str
    sla: int

    def __init__(self, expt_details: dict):
        """
        expt_details is dict containing all required arguments
        """
        self.benchmark = expt_details["benchmark"]
        self.model = expt_details["model"]
        self.vm = expt_details["vm"]
        self.aml_dir = expt_details["aml_dir"]
        self.sla = expt_details["SLA"]

    def max_throughput_under_sla(self, metadata: {}):
        """
        Executes DLRM for Maximum throughput under SLA

        Args:
            vm (BaseVirtualMachine): The target virtual machine where the Docker container runs.
            metadata (dict): Additional metadata to include in the performance results.

        """
        # call warm up function
        cmd_dict_common = dlrm_base_utils.DlrmRunModel.common_flags(self)
        dlrm_base_utils.DlrmRunModel.warm_up_dlrm(self, cmd_dict_common)
        num_cores = self.vm.NumCpusForBenchmark()
        max_tpt = None
        p99_latency = None
        final_max_result = []
        thread_range = [i for i in range(1, num_cores + 1) if num_cores % i == 0]
        for thread in thread_range:
            process = num_cores // thread
            batch_size_lower_bound = FLAGS[
                f"{self.benchmark}_batch_size_lower_bound"
            ].value
            # Start run with lower batch size
            try:
                results_data = self.run_pytorch_command(
                    batch_size_lower_bound, thread, process, metadata
                )
            except Exception as e:
                logging.info("An unexpected error occurred: %s", e)
            if results_data is not None:
                max_tpt = results_data.results[0][3]
                p99_latency = results_data.results[0][4]
                if p99_latency < self.sla:
                    (
                        best_batch_size,
                        best_max_tpt,
                        best_p90_latency,
                        best_p99_latency,
                        best_p999_latency,
                    ) = self.binary_search_on_batch_size(
                        thread, process, metadata, max_tpt
                    )
                    final_max_result.append(
                        {
                            "batch_size": best_batch_size,
                            "num_threads": thread,
                            "num_processes": process,
                            "max_tpt": best_max_tpt,
                            "p90_latency": best_p90_latency,
                            "p99_latency": best_p99_latency,
                            "p999_latency": best_p999_latency,
                        }
                    )
        get_max_data = None
        logging.info("final_max_result",final_max_result)
        if final_max_result:
            get_max_data = max(final_max_result, key=lambda x: float(x["max_tpt"]))
        return get_max_data

    def binary_search_on_batch_size(self, thread, process, metadata, max_tpt):
        """Run binary search on Batch Size
        Input:
            -thread
            -process
        Output:
            -Binary search result
        """
        best_max_tpt = None
        best_p90_latency = None
        best_p99_latency = None
        best_p999_latency = None
        best_batch_size = None
        batch_size_upper_bound = FLAGS[f"{self.benchmark}_batch_size_upper_bound"].value
        batch_size_lower_bound = FLAGS[f"{self.benchmark}_batch_size_lower_bound"].value
        while batch_size_lower_bound <= batch_size_upper_bound:
            batch_results_data = []
            batch_size_mid = (
                batch_size_lower_bound
                + (batch_size_upper_bound - batch_size_lower_bound) // 2
            )
            try:
                batch_results_data = self.run_pytorch_command(
                    batch_size_mid, thread, process, metadata
                )
            except Exception as e:
                logging.info("An unexpected error occurred: %s", e)
            current_tpt = batch_results_data.results[0][3]
            p90_latency = batch_results_data.results[0][4]
            p99_latency = batch_results_data.results[0][5]
            p999_latency = batch_results_data.results[0][6]
            if p99_latency >= self.sla:
                batch_size_upper_bound = batch_size_mid - 1
                continue
            if current_tpt > max_tpt:
                max_tpt = current_tpt
                best_max_tpt = max_tpt
                best_p90_latency = p90_latency
                best_p99_latency = p99_latency
                best_p999_latency = p999_latency
                best_batch_size = batch_size_mid
            batch_size_lower_bound = batch_size_mid + 1
        return (
            best_batch_size,
            best_max_tpt,
            best_p90_latency,
            best_p99_latency,
            best_p999_latency,
        )

    def run_pytorch_command(self, batch_size, thread, process, metadata):
        """First run for SLA with lower bound"""
        results_data = []
        cmd_dict_common = dlrm_base_utils.DlrmRunModel.common_flags(self)
        cmd_dict = {
            "batch_size": batch_size,
            "num_threads": thread,
            "num_processes": process,
            "run_type": "run",
            "metadata": metadata,
        }
        try:
            results_data = dlrm_base_utils.DlrmRunModel.run_pytorch_command(
                self, cmd_dict, cmd_dict_common
            )
        except Exception as e:
            logging.info("An unexpected error occurred: %s", e)
        return results_data

    def validate_max_tpt_result(self, get_max_data, metadata):
        """Validate the Max TPT Result to be correct"""
        final_max_tpt_result = []
        validation_runs = FLAGS[f"{self.benchmark}_sla_validation_runs"].value
        thread = get_max_data["num_threads"]
        process = get_max_data["num_processes"]
        batch_size = get_max_data["batch_size"]
        for i in range(validation_runs):
            print(f"Run Number:{i+1}")
            results_data = []
            results_data = self.run_pytorch_command(
                batch_size, thread, process, metadata
            )
            max_tpt = results_data.results[0][3]
            p90_latency = results_data.results[0][4]
            p99_latency = results_data.results[0][6]
            p999_latency = results_data.results[0][6]
            final_max_tpt_result.append(
                {
                    "num_threads": thread,
                    "num_processes": process,
                    "batch_size": batch_size,
                    "max_tpt": max_tpt,
                    "p90_latency": p90_latency,
                    "p99_latency": p99_latency,
                    "p999_latency": p999_latency,
                }
            )
        return final_max_tpt_result
