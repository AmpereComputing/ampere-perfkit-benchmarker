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

This is a set of benchmarks that measures performance of DLRM

"""

import posixpath
import os
import glob
from absl import flags
from perfkitbenchmarker import vm_util
from perfkitbenchmarker import sample
from perfkitbenchmarker import configs

from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import pytorch
from ampere.pkb.linux_packages import dlrm
from ampere.pkb.utils import dlrm_base_utils
from ampere.pkb.utils import pytorch_model_sla

BENCHMARK_NAME = "ampere_pytorch_dlrm"

BENCHMARK_CONFIG = """
ampere_pytorch_dlrm:
  description: Benchmark Pytorch
  vm_groups:
    servers:
      vm_spec: *default_single_core
      disk_spec: *default_50_gb
"""

FLAGS = flags.FLAGS

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_util_path", None, "path to benchmark.py file inside the docker"
)

aml_dir = flags.DEFINE_string(
    f"{BENCHMARK_NAME}_aml_dir", None, "dir to Ampere model library in docker"
)

threads_per_process_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_threads_per_process", [8, 16], "number of threads to use"
)

batch_sizes_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_batch_size", [512, 1024], "batch sizes to cover"
)

flags.DEFINE_float(f"{BENCHMARK_NAME}_duration", 900.00, "run duration for dlrm test")

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_scenario", "throughput", "scenario can be latency or throughput"
)

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_precision",
    "fp32",
    "fp32 or fp16 precision of the model provided",
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

DATA_FILES = flags.DEFINE_string(
    f"{BENCHMARK_NAME}_data",
    "",
    "Must be in ./ampere/pkb_internal/data/",
)

flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_batch_size_upper_bound",
    2048,
    "Use batch size upper bound for max throughput mode.",
)

flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_batch_size_lower_bound",
    256,
    "Use batch size lower bound for max throughput mode.",
)

dlrm_sla_mode = flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_sla_mode",
    False,
    "",
)

flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_sla",
    10,
    "SLA under 10ms",
)
dlrm_sla_validation = flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_sla_validation",
    False,
    "",
)
flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_sla_validation_runs",
    5,
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
    dlrm.check_threads_validity()
    pytorch.Install(server)
    dlrm_base_utils.validate()


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

    def _run_max_tpt(server, benchmark_metadata):
        """Get maximum throughput under SLA"""
        expt_dict = {
            "benchmark": "ampere_pytorch_dlrm",
            "model": "dlrm",
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
            if dlrm_sla_validation.value:
                sla_base.validate_max_tpt_result(max_throughput_data, metadata)
            best_tps_sample = _parse_max_tpt_results(max_throughput_data, metadata)
            create_dlrm_log_tar(server)
        else:
            metadata = {
                "docker_version": docker_version,
            }
            best_tps_sample = _empty_results(metadata)
        return best_tps_sample

    def _run(vm, benchmark_metadata):
        # call benchmark.py using docker exec from dlrm.py
        # dlrm.run_dlrm_model(vm, benchark_metadata)
        expt_dict = {
            "benchmark": "ampere_pytorch_dlrm",
            "model": "dlrm",
            "vm": vm,
            "aml_dir": aml_dir,
        }
        dlrm_base = dlrm_base_utils.DlrmRunModel(expt_dict)
        dlrm_base.run_dlrm_model(
            batch_sizes_list.value, threads_per_process_list.value, benchmark_metadata
        )
        create_dlrm_log_tar(server)
        sample_data = _parse_output_files(benchmark_metadata)
        return sample_data
    server = benchmark_spec.vm_groups["servers"][0]
    benchmark_metadata = {}
    if dlrm_sla_mode.value:
        return _run_max_tpt(server, benchmark_metadata)
    return _run(server, benchmark_metadata)


def create_dlrm_log_tar(server):
    """Create a Tar file from the DLRM logs"""
    server.RemoteCommand(
        f"cd {download_utils.INSTALL_DIR} && "
        f"tar -cf dlrm_logs.tar {download_utils.INSTALL_DIR}/out_dir"
    )
    dlrm_logs = posixpath.join(download_utils.INSTALL_DIR, "dlrm_logs.tar")
    server.RemoteCopy(vm_util.GetTempDir(), dlrm_logs, False)

def _parse_max_tpt_results(max_throughput_data, benchmark_metadata):
    """Parse the Max Tpt Results"""
    all_samples = []
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
    all_samples.extend(samples)
    return all_samples


def _empty_results(benchmark_metadata):
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


def _parse_output_files(benchmark_metadata):
    """
    Parses CSV output files from the temporary directory and extracts benchmark samples.

    This function looks for CSV files in the temporary directory, assumes each file is
    named with the model name followed by an '@' symbol (e.g., 'dlrm@TH1BA2048.csv'),
    and reads the file contents. It uses `DLRMResult.parse_dlrm_results` to parse the
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
    metadata = {
        "docker_version": docker_version,
    }
    benchmark_metadata = benchmark_metadata | metadata
    docker_version = pytorch.get_pytorch_metadata()
    csv_files = glob.glob(vm_util.GetTempDir() + "/*.csv")
    if csv_files:
        for filename in csv_files:
            csv_file = os.path.basename(filename)
            metadata = {
                "docker_version": docker_version,
            }
            csv_file_dlrm = posixpath.join(vm_util.GetTempDir(), csv_file)
            with open(csv_file_dlrm, "r", encoding="utf-8") as output:
                dlrm_output_data = output.read()
            dlrm_results = dlrm.DLRMResult.parse_dlrm_results(dlrm_output_data)
            benchmark_metadata = benchmark_metadata | metadata
            all_files_samples.extend(dlrm_results.get_samples(benchmark_metadata))
    return all_files_samples


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
