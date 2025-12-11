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

This is a set of benchmarks that measures performance of Resnet50

"""

import posixpath
import os
import glob
import csv
import dataclasses
import logging
from typing import Any, Dict, List
import six
from absl import flags
from perfkitbenchmarker import vm_util
from perfkitbenchmarker.virtual_machine import BaseVirtualMachine
from perfkitbenchmarker import configs
from perfkitbenchmarker import sample
from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import pytorch
from ampere.pkb.utils import pytorch_base_utils
from ampere.pkb.utils import pytorch_model_sla


BENCHMARK_NAME = "ampere_pytorch_resnet"

BENCHMARK_CONFIG = """
ampere_pytorch_resnet:
  description: Benchmark Pytorch
  vm_groups:
    servers:
      vm_spec: *default_single_core
      disk_spec: *default_50_gb
"""

FLAGS = flags.FLAGS

resnet_runner_path = flags.DEFINE_string(
    f"{BENCHMARK_NAME}_runner_path", "computer_vision/classification/resnet_50_v15/run.py"
    , "path to resnet run.py file inside the docker"
)

aml_dir = flags.DEFINE_string(
    f"{BENCHMARK_NAME}_aml_dir", None, "dir to ampere_model_library in docker"
)

threads_per_process_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_threads_per_process", [8, 16], "number of threads to use"
)

batch_sizes_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_batch_size", [512, 1024], "batch sizes to cover"
)

flags.DEFINE_integer(f"{BENCHMARK_NAME}_number_of_models", 0, "")

flags.DEFINE_float(f"{BENCHMARK_NAME}_duration", 900.00, "run duration for resnet50 test")

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_scenario", "throughput", "scenario can be latency or throughput"
)

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_precision", "fp32", "fp32 or fp16 precision of the model provided"
)

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_threads_range",
    "",
    "range of threads to use in offline/ throughput mode, "
    "e.g. '0-63,128-191', threads will be divided",
)

flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_sleep_duration",
    15,
    "sleep duration while polling for processes to complete",
)
flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_batch_size_upper_bound",
    32,
    "Use batch size upper bound for max throughput mode.",
)

flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_batch_size_lower_bound",
    1,
    "Use batch size lower bound for max throughput mode.",
)

resnet_sla_mode = flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_sla_mode",
    False,
    "Measure latency capped throughput. Use in conjunction with "
    "ampere_pytorch_resnet_sla . Defaults to False.",
)

flags.DEFINE_float(
    f"{BENCHMARK_NAME}_sla",
    100.00,
    "SLA under 100ms",
)
resnet_sla_validation = flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_sla_validation",
    False,
    "",
)
flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_sla_validation_runs",
    2,
    "Define sla validation runs",
)



def GetConfig(user_config):
    """Load and return benchmark config.

    Args:
      user_config: user supplied configuration (flags and config file)

    Returns:
      loaded benchmark configuration
    """
    config = configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)
    return config


def Prepare(benchmark_spec):
    """Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
    """

    server = benchmark_spec.vm_groups["servers"][0]
    pytorch_base_utils.check_threads_validity(BENCHMARK_NAME)
    pytorch.Install(server)
    pytorch_base_utils.validate()


def Run(benchmark_spec):
    """
    Executes the benchmark on the specified virtual machine and parses the output.

    This function retrieves the first server from the 'servers' group in the provided
    `benchmark_spec` and runs the benchmark by invoking the internal `_run` function.
    After the benchmark completes, it parses the output files and returns the results.

    Args:
        benchmark_spec: An object containing the benchmark specification, including
                         the virtual machine group and other necessary metadata.

    Returns:
        list: A list of benchmark samples or results extracted from the output files.
    """

    server = benchmark_spec.vm_groups["servers"][0]
    benchmark_metadata = {}
    if resnet_sla_mode.value:
        return _run_max_tpt(server, benchmark_metadata)
    return _run(server, benchmark_metadata)

def create_resnet_log_tar(server):
    """function to create and copy resnet logs tar from remote machine """
    server.RemoteCommand(f"cd {download_utils.INSTALL_DIR} && "
                         f"tar -cf resnet_logs.tar {download_utils.INSTALL_DIR}/out_dir")
    resnet_logs = posixpath.join(download_utils.INSTALL_DIR, "resnet_logs.tar")
    server.RemoteCopy(vm_util.GetTempDir(), resnet_logs, False)


def _run_max_tpt(server, benchmark_metadata):
    """Get maximum throughput under SLA"""
    expt_dict = {
            "benchmark": "ampere_pytorch_resnet",
            "model": "resnet",
            "vm": server,
            "aml_dir": aml_dir,
            "SLA": FLAGS[f"{BENCHMARK_NAME}_sla"].value,
        }
    sla_base = pytorch_model_sla.SlaRunModel(expt_dict)
    max_throughput_data = sla_base.max_throughput_under_sla(benchmark_metadata)
    docker_version = pytorch.get_pytorch_metadata()
    if max_throughput_data is not None:
        metadata = {
                "batch_size": max_throughput_data["batch_size"],
                "docker_version": docker_version,
                "n_proc": max_throughput_data["num_processes"],
                "n_threads": max_throughput_data["num_threads"],
                }
        if resnet_sla_validation.value:
            sla_base.validate_max_tpt_result(max_throughput_data, metadata)
        best_tps_sample = _parse_max_tpt_results(max_throughput_data, metadata)
        create_resnet_log_tar(server)
    else:
        metadata = {
                "docker_version": docker_version,
                }
        best_tps_sample = _empty_results(metadata)
    return best_tps_sample


def _run(vm, benchmark_metadata):
    """call benchmark.py using docker exec common utility    """
    expt_dict = {
            "benchmark": "ampere_pytorch_resnet",
            "model": "resnet",
            "vm": vm,
            "aml_dir": aml_dir,
            }
    resnet_base = pytorch_base_utils.PytorchRunModel(expt_dict)
    resnet_base.run_pytorch_model(
            batch_sizes_list.value, threads_per_process_list.value, benchmark_metadata
            )
    create_resnet_log_tar(vm)
    sample_data = _parse_output_files(benchmark_metadata)
    return sample_data


def _parse_output_files(benchmark_metadata):
    """
    Parses CSV output files from the temporary directory and extracts benchmark samples.

    This function looks for CSV files in the temporary directory, assumes each file is
    named with the model name followed by an '@' symbol (e.g., 'resnet@TH1BA2048.csv'),
    and reads the file contents. It uses `DLRMResult.parse_resnet_results` to parse the
    results and extracts samples, augmenting the provided benchmark metadata with the
    model name.

    Args:
        benchmark_metadata (dict): A dictionary of existing metadata to be merged with
                                   model-specific metadata extracted from each file.

    Returns:
        list: A list of sample dictionaries extracted from all parsed CSV files.
    """

    all_files_samples = []
    metadata = {}
    docker_version = pytorch.get_pytorch_metadata()
    csv_files = glob.glob(vm_util.GetTempDir() + "/*.csv")
    if csv_files:
        for filename in csv_files:
            csv_file = os.path.basename(filename)
            metadata = {
                "docker_version": docker_version,
            }
            csv_file_resnet = posixpath.join(vm_util.GetTempDir(), csv_file)
            with open(csv_file_resnet, "r", encoding="utf-8") as output:
                resnet_output_data = output.read()
            resnet_results = Resnet50Result.parse_resnet50_results(resnet_output_data)
            benchmark_metadata = benchmark_metadata | metadata
            all_files_samples.extend(resnet_results.get_samples(benchmark_metadata))
    return all_files_samples


def _parse_max_tpt_results(max_throughput_data, benchmark_metadata):
    """Parse the Max Tpt Results"""
    samples = [
        sample.Sample(
            "Max throughput",
            max_throughput_data["max_tpt"],
            "samples/s",
            benchmark_metadata,
        ),
        sample.Sample(
            "p90_latency",
            max_throughput_data["p90_latency"],
            "ms",
            benchmark_metadata,
        ),
        sample.Sample(
            "p99_latency",
            max_throughput_data["p99_latency"],
            "ms",
            benchmark_metadata,
        ),
        sample.Sample(
            "p999_latency",
            max_throughput_data["p999_latency"],
            "ms",
            benchmark_metadata,
        ),
    ]
    return samples


def _empty_results(benchmark_metadata):
    """returns empty sample if best throughput is not achieved """
    all_samples = []
    samples = [
        sample.Sample(
            "Best throughput not met under a given SLA",
            0,
            "",
            benchmark_metadata,
        ),
    ]
    all_samples.extend(samples)
    return all_samples


def Cleanup(benchmark_spec):
    """
    Cleans up the benchmark environment by uninstalling Docker from the server VM.

    This function retrieves the first VM from the 'servers' group in the benchmark
    specification and uninstalls the Docker package to ensure a clean state.

    Args:
        benchmark_spec: An object containing the VM group specifications used in the benchmark.
    """
    server = benchmark_spec.vm_groups["servers"][0]
    pytorch.Uninstall(server)


@dataclasses.dataclass
class Resnet50Result:
    """Class that represents Resnet50 results."""

    n_proc: list[int]
    n_threads: list[int]
    batch_size: list[int]
    throughput_tps: list[float]
    p90_latency: list[float]
    p99_latency: list[float]
    p999_latency: list[float]
    start: list[str]
    finish: list[str]

    @classmethod
    def parse_resnet50_results(cls, resnet50_results: str) -> "Resnet50Result":
        """Parse resnet50 result textfile and return results.
        Args:
          resnet50_results: Str output of running resnet50.
        Returns:
        """
        resnet50_csv_result = _parse_csv(resnet50_results)
        return cls(
            n_proc=resnet50_csv_result.n_proc,
            n_threads=resnet50_csv_result.n_threads,
            batch_size=resnet50_csv_result.batch_size,
            throughput_tps=resnet50_csv_result.throughput_tps,
            p90_latency=resnet50_csv_result.p90_latency,
            p99_latency=resnet50_csv_result.p99_latency,
            p999_latency=resnet50_csv_result.p999_latency,
            start=resnet50_csv_result.start,
            finish=resnet50_csv_result.finish,
        )

    def get_samples(self, metadata: Dict[str, Any]) -> List[sample.Sample]:
        """
        Generate a list of performance samples with associated metadata.

        This method constructs and returns a list of `sample.Sample` objects
        representing performance metrics (throughput and p90, p99 and p99.9 latency) across
        different processor/thread/batch size configurations.

        For each configuration (`n_proc`, `n_threads`, `batch_size`), it:
        - Updates the metadata with the current configuration.
        - Creates samples for:
            - Throughput (in samples/sec)
            - 90th percentile latency (in milliseconds)
            - 99th percentile latency (in milliseconds)
            - 99.9th percentile latency (in milliseconds)

        Args:
            metadata (Dict[str, Any]): A dictionary of common metadata to attach
                                       to each sample.

        Returns:
            List[sample.Sample]: A list of sample objects containing performance metrics.
        """
        all_samples = []
        metadata_new = {}
        for count_n_proc, _ in enumerate(self.n_proc):
            metadata_new["n_proc"] = self.n_proc[count_n_proc]
            metadata_new["n_threads"] = self.n_threads[count_n_proc]
            metadata_new["batch_size"] = self.batch_size[count_n_proc]
            metadata_sample = metadata | metadata_new
            samples = [
                sample.Sample(
                    "throughput",
                    self.throughput_tps[count_n_proc],
                    "samples/s",
                    metadata_sample,
                ),
                sample.Sample(
                    "p90_latency",
                    self.p90_latency[count_n_proc],
                    "ms",
                    metadata_sample,
                ),
                sample.Sample(
                    "p99_latency",
                    self.p99_latency[count_n_proc],
                    "ms",
                    metadata_sample,
                ),
                sample.Sample(
                    "p999_latency",
                    self.p999_latency[count_n_proc],
                    "ms",
                    metadata_sample,
                ),
            ]
            all_samples.extend(samples)
        return all_samples


def _parse_csv(resnet50_results: str) -> Resnet50Result:
    """Parses the output
    Yields:
    (n_proc,n_threads,batch_size,throughput_tps,p90_latency,
    p99_latency,p999_latency,start,finish) tuples.
    """
    n_proc: list[int] = []
    n_threads: list[int] = []
    batch_size: list[int] = []
    throughput_tps: list[float] = []
    p90_latency: list[float] = []
    p99_latency: list[float] = []
    p999_latency: list[float] = []
    start: list[str] = []
    finish: list[str] = []
    csv_fp = six.StringIO(str(resnet50_results))
    reader = csv.DictReader(csv_fp)
    if frozenset(reader.fieldnames) != frozenset(
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
    ):
        raise ValueError(f"Test Failed: {resnet50_results}")
    for row in reader:
        n_proc.append(row["Processes"])
        n_threads.append(row["threads"])
        batch_size.append(row["batch_size"])
        throughput_tps.append(row["throughput"])
        p90_latency.append(row["p90_latency"])
        p99_latency.append(row["p99_latency"])
        p999_latency.append(row["p999_latency"])
        start.append(row["start"])
        finish.append(row["finish"])
    return Resnet50Result(
        n_proc,
        n_threads,
        batch_size,
        throughput_tps,
        p90_latency,
        p99_latency,
        p999_latency,
        start,
        finish,
    )


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

    def __init__(self, vm: BaseVirtualMachine, params):
        self.params = params
        self.vm = vm
        self.lines = []
        self.results = []
        self.throughput = 0.0
        self.p90_latency = 0.0
        self.p99_latency = 0.0
        self.p999_latency = 0.0

    def summarize(self, vm: BaseVirtualMachine, logs_dir, start, finish, save_logs_dir):
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
        resnet50_throughput = []
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
            resnet50_throughput.append(proc_throughput)
            log_line = (f"{n}|{num_threads}|{batch_size}|{proc_throughput}|"
                        f"{p90_latency}|{p99_latency}|{p999_latency}\n")
            self.lines.append(log_line)
        self.throughput = sum(resnet50_throughput)
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
        log_filename = (
            f"{save_logs_dir}/{self.params['model']}.log"
        )
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
        results_filename = (
            f"{save_dir}/{self.params['model']}.csv"
        )

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
