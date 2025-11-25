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

import csv
import os
import uuid
import time
import logging
from absl import flags
from perfkitbenchmarker import vm_util
from perfkitbenchmarker.virtual_machine import VirtualMachine
from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import docker as pytorch_docker
from ampere.pkb.linux_packages import dlrm
from ampere.pkb.linux_packages import pytorch

FLAGS = flags.FLAGS

INSTALL_DIR = download_utils.INSTALL_DIR

def validate():
    """
    Validates the required configuration flags for running the PyTorch benchmark.

    This function performs sanity checks on Docker-related and benchmark-specific
    FLAGS to ensure the environment is correctly configured before running.

    Validations:
    - Ensures that the Docker container name is provided.
    - For the 'throughput' scenario, ensures that the thread range is specified.

    Returns:
        bool: True if all validations pass.

    Raises:
        ValueError: If the Docker container name is missing or if the thread range
                    is not specified for the throughput scenario.
    """

    valid = True
    if not FLAGS[f"{pytorch_docker.PACKAGE_NAME}_bash_command"].value:
        FLAGS[f"{pytorch_docker.PACKAGE_NAME}_bash_command"].value = (
            "--entrypoint /bin/sh -it"
        )
    if not FLAGS[f"{pytorch_docker.PACKAGE_NAME}_name"].value:
        raise ValueError(
            "Docker is not running.Docker name should be provided to look into docker"
        )
    if (
        FLAGS["ampere_pytorch_dlrm_scenario"] == "throughput"
        and FLAGS["ampere_pytorch_dlrm_threads_range"] is None
    ):
        raise ValueError(
            "Range of threads to use needs to be set with --threads_range arg when "
            "running the throughput scenario."
        )
    return valid

class DlrmRunModel:
    """
    Run DLRM model
    """

    benchmark: str
    num_processes: list[int] = []
    batch_size: list[int] = []
    model: str
    num_available_threads: int
    vm: VirtualMachine
    aml_dir: str

    def __init__(self, expt_details: dict):
        """
        expt_details is dict containing all required arguments
        """
        self.benchmark = expt_details["benchmark"]
        self.model = expt_details["model"]
        self.num_available_threads = len(
            dlrm.parse_threads_range(FLAGS[f"{self.benchmark}_threads_range"].value)
        )
        self.vm = expt_details["vm"]
        self.aml_dir = expt_details["aml_dir"]


    def run_dlrm_model(
        self,
        batch_sizes_list: list[int],
        threads_per_process_list: list[int],
        metadata: {},
    ):
        """
        Executes the DLRM (Deep Learning Recommendation Model) benchmark
        inside a Docker container on the given VM.

        This function automates benchmarking for the DLRM model under varying configurations
        of batch size, thread count, and number of processes. It prepares the environment,
        constructs the benchmark command, and runs it within a Docker container. Results
        are collected, summarized, and saved for each configuration.

        Steps performed:
        - Parses the thread range and calculates process/thread configurations.
        - Sets up environment variables for mixed precision (e.g., FP16) if required.
        - Constructs and runs the benchmark command inside the container.
        - Collects runtime performance metrics and saves logs to a CSV.

        Args:
            vm (BaseVirtualMachine): The target virtual machine where the Docker container runs.
            metadata (dict): Additional metadata to include in the performance results.

        Raises:
            ValueError: If the number of available threads cannot support
                        the selected configuration.

        Side Effects:
            - Executes the benchmark script inside the container.
            - Logs performance summaries to standard output and saves CSV files.

        Returns:
            None
        """
        # set common flags for Dlrm run
        cmd_dict_common = DlrmRunModel.common_flags(self)
        num_available_threads = cmd_dict_common["num_available_threads"]
        # call warm up function
        DlrmRunModel.warm_up_dlrm(self, cmd_dict_common)
        for batch_size in sorted(batch_sizes_list):
            for num_threads in sorted(threads_per_process_list, reverse=True):
                num_processes = int(len(num_available_threads) / int(num_threads))
                current_case = f"{num_processes} x {num_threads} "
                current_case += f"[proc x threads per proc], bs = {batch_size}"
                logging.info("\nRunning %s", current_case)
                cmd_dict = {
                    "batch_size": batch_size,
                    "num_threads": num_threads,
                    "num_processes": num_processes,
                    "run_type": "run",
                    "metadata": metadata,
                }
                DlrmRunModel.run_pytorch_command(self, cmd_dict, cmd_dict_common)

    def warm_up_dlrm(self, cmd_dict_common):
        """warm up function to run dlrm"""
        metadata = {}
        cmd_dict = {
            "batch_size": 256,
            "num_threads": 1,
            "num_processes": 1,
            "run_type": "warmup",
            "metadata": metadata,
        }
        DlrmRunModel.run_pytorch_command(self, cmd_dict, cmd_dict_common)

    def common_flags(self):
        """set common flags in dictonary"""
        threads_range = FLAGS[f"{self.benchmark}_threads_range"].value
        precision = FLAGS[f"{self.benchmark}_precision"].value
        scenario = FLAGS[f"{self.benchmark}_scenario"].value
        duration = FLAGS[f"{self.benchmark}_duration"].value
        sleep_duration = FLAGS[f"{self.benchmark}_sleep_duration"].value
        num_available_threads = dlrm.parse_threads_range(threads_range)
        source_cmd = ""
        if precision == "fp16":
            source_cmd = (
                "export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; "
            )
        source_cmd += f"source {self.aml_dir.value}/set_env_variables.sh;"
        cmd_dict_common = {
            "model": self.model,
            "source_cmd": source_cmd,
            "precision": precision,
            "scenario": scenario,
            "duration": duration,
            "sleep_duration": sleep_duration,
            "num_available_threads": num_available_threads,
        }
        return cmd_dict_common

    def run_pytorch_command(self, cmd_dict, cmd_dict_common):
        """run dlrm command"""
        results = None
        duration = cmd_dict_common["duration"]
        sleep_duration = cmd_dict_common["sleep_duration"]
        threads_range = FLAGS[f"{self.benchmark}_threads_range"].value
        pytorch.set_flags(self.vm)
        volumes = FLAGS[f"{pytorch_docker.PACKAGE_NAME}_volume_names"].value
        volume_mountpoints = FLAGS[
            f"{pytorch_docker.PACKAGE_NAME}_volume_mountpoints"
        ].value
        output_dir = volumes[1]
        docker_out_dir = volume_mountpoints[1]
        batch_size = cmd_dict["batch_size"]
        num_processes = cmd_dict["num_processes"]
        num_threads = cmd_dict["num_threads"]
        args = {
            "model": cmd_dict_common["model"],
            "precision": cmd_dict_common["precision"],
            "batch_size": batch_size,
            "num_processes": num_processes,
            "num_threads": num_threads,
            "scenario": cmd_dict_common["scenario"],
        }
        log_dir_name = f"{num_processes}_{num_threads}_{batch_size}_{str(uuid.uuid4())}"
        docker_logs_dir = os.path.join(docker_out_dir, log_dir_name)
        logs_dir = os.path.join(output_dir, docker_logs_dir.split("/")[-1])
        cmd = (
            f"{cmd_dict_common['source_cmd']} "
            f"python3 /workspace/benchmark_pytorch_dlrm.py -p {str(cmd_dict_common['precision'])} "
            f"--aml_dir {str(self.aml_dir.value)} "
            f" --output_dir {str(docker_logs_dir)} "
            f" -n {str(num_processes)} -t {str(num_threads)} "
            f" -b {str(batch_size)} -r {threads_range} "
            f"--duration {str(duration)} --sleep_duration {str(sleep_duration)}"
        )
        run_type = cmd_dict["run_type"]
        FLAGS[f"{pytorch_docker.PACKAGE_NAME}_shell_type"].value = "bash"
        FLAGS[f"{pytorch_docker.PACKAGE_NAME}_exec_command"].value = cmd
        try:
            self.vm.RemoteCommand("sudo swapoff -a")
            start = time.time()
            pytorch_docker.exec_docker(self.vm)
            finish = time.time()
            metadata = cmd_dict["metadata"] | args
            if run_type == "run":
                results = Results(self.vm, metadata)
                results.summarize(
                    self.vm, logs_dir, start, finish, vm_util.GetTempDir()
                )
                results.save_csv(vm_util.GetTempDir())
        except Exception as e:
            self.vm.RemoteCommand("sudo pkill -9 python3")
            logging.info("An unexpected error occurred: %s", e)
        return results


# For postprocessing summarize logs in case of scenario = offline/throughput
# latency and throughput will be output


class Results:
    """
    A container for storing and managing benchmark results and metadata.

    Attributes:
        vm (BaseVirtualMachine): The virtual machine used for the benchmark run.
        params (dict): A dictionary of benchmark parameters (e.g., batch size, threads, process).
        proc_throughput : throughput
        p90_latency : 90th percentile latency
        p99_latency : 99th percentile latency
        p999_latency : 99.9th percentile latency
        lines (list): A list to store output lines or logs related to the benchmark.
        results (list): A list to accumulate parsed or final benchmark results.
    """

    def __init__(self, vm: VirtualMachine, params):
        self.params = params
        self.vm = vm
        self.lines = []
        self.results = []
        self.throughput = 0.0
        self.p90_latency = 0.0
        self.p99_latency = 0.0
        self.p999_latency = 0.0

    def summarize(self, vm: VirtualMachine, logs_dir, start, finish, save_logs_dir):
        print("summarize")
        """
        Summarizes the logs by computing averages and sums within a specified range.

        This method processes the log files from the provided directory, using the
        `start` and `finish` parameters to define the range of interest. It then calculates
        averages and sums based on the log data, which can be used for further analysis.
        The processed logs can optionally be saved to a specified directory.

        Args:
            vm (BaseVirtualMachine): The virtual machine object, potentially used for
                                      remote log access or command execution.
            logs_dir (str): The directory containing the log files to be summarized.
            start (int): The starting index or timestamp to filter logs from.
            finish (int): The ending index or timestamp to filter logs up to.
            save_logs_dir (str): The directory to save the processed log summaries (if applicable).

        Returns:
            None:
        """
        log_file_name = logs_dir.split("/")
        log_name_split = log_file_name[len(log_file_name) - 1].split("_")
        dlrm_throughput = []
        for n in range(self.params["num_processes"]):
            tpt_grep, _ = vm.RemoteCommand(
                f"cat {logs_dir}/log_{n} | grep observed | sed 's/\\s\\+/ /g' | cut -d' ' -f3"
            )
            p90_latency, _ = vm.RemoteCommand(
                f"cat {logs_dir}/log_{n} | grep p90 | sed 's/\\s\\+/ /g' | cut -d' ' -f3"
            )
            self.p90_latency = max(float(self.p90_latency), float(p90_latency))
            p90_latency = p90_latency.strip()
            p99_latency, _ = vm.RemoteCommand(
                f"cat {logs_dir}/log_{n} | grep -w 'p99 ' | sed 's/\\s\\+/ /g' | cut -d' ' -f3"
            )
            p99_latency = p99_latency.strip()
            self.p99_latency = max(float(self.p99_latency), float(p99_latency))
            p999_latency, _ = vm.RemoteCommand(
                f"cat {logs_dir}/log_{n} | grep p99.9 | sed 's/\\s\\+/ /g' | cut -d' ' -f3"
            )
            self.p999_latency = max(float(self.p999_latency), float(p999_latency))
            p999_latency = p999_latency.strip()
            proc_throughput = float(tpt_grep)
            num_processes = int(log_name_split[0])
            assert num_processes == int(self.params["num_processes"])
            num_threads = int(log_name_split[1])
            assert num_threads == int(self.params["num_threads"])
            batch_size = int(log_name_split[2])
            assert batch_size == int(self.params["batch_size"])
            dlrm_throughput.append(proc_throughput)
            log_line = f"{n}|{num_threads}|{batch_size}|{proc_throughput}\
            |{p90_latency}|{p99_latency}|{p999_latency}\n"
            self.lines.append(log_line)
        self.throughput = sum(dlrm_throughput)
        self.results.append(
            [
                self.params["num_processes"],
                self.params["num_threads"],
                self.params["batch_size"],
                self.throughput,
                self.p90_latency,
                self.p99_latency,
                self.p999_latency,
                start,
                finish,
            ]
        )
        log_filename = f"{save_logs_dir}/{self.params['model']}.log"
        with open(log_filename, "a", encoding="utf-8") as f1:
            f1.writelines(
                [
                    "Process number|",
                    "threads_per_process|",
                    "batch_size|",
                    "throughput|",
                    "p90_latency|",
                    "p99_latency|",
                    "p999_latency\n",
                ]
            )
            f1.writelines(self.lines)
        logging.info("Logs saved in %s", log_filename)

    def save_csv(self, save_dir):
        """
        saves csv in save_dir
        """
        results_filename = f"{save_dir}/{self.params['model']}.csv"

        if os.path.exists(results_filename):
            first_write = False
        else:
            first_write = True
        with open(results_filename, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            if first_write:
                writer.writerow(
                    [
                        "Processes",
                        "threads",
                        "batch_size",
                        "throughput",
                        "p90_latency",
                        "p99_latency",
                        "p999_latency",
                        "start",
                        "finish",
                    ]
                )
            writer.writerow(self.results[0])
        logging.info("Result saved in %s", results_filename)
