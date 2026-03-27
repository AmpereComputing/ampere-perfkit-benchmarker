# Copyright (c) 2026, Ampere Computing LLC
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

This is a set of benchmarks that measures performance of llama

"""
import logging
import csv
import posixpath
from dataclasses import dataclass
import os
from typing import Any, Dict, List
import glob
import six
import pandas as pd
from absl import flags
from perfkitbenchmarker import vm_util
from perfkitbenchmarker import sample
from perfkitbenchmarker import configs
from perfkitbenchmarker.virtual_machine import BaseVirtualMachine
from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import docker as docker_package
from ampere.pkb.linux_packages import llama as ampere_llama
from ampere.pkb.utils import llm_base_utils
from ampere.pkb.utils import threads_validation
try:
    from ampere.pkb_internal.utils import llm_utils as llm_utils_internal
except ImportError as err:
    llm_utils_internal = None
    logging.info(f"Failed to import llm_utils_internal: {err}")


BENCHMARK_NAME = "ampere_llama_benchmark"

BENCHMARK_CONFIG = """
ampere_llama_benchmark:
  description: Benchmark Llama
  vm_groups:
    servers:
      vm_spec: *default_single_core
      disk_spec: *default_50_gb
"""

FLAGS = flags.FLAGS

model_names = flags.DEFINE_list(f"{BENCHMARK_NAME}_model_names", [], "")

flags.DEFINE_string(f"{BENCHMARK_NAME}_llama_exe_path", None, "")

threads_per_process_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_threads_per_process", [], ""
)

flags.DEFINE_integer(f"{BENCHMARK_NAME}_number_of_models", 0, "")

batch_sizes_list = flags.DEFINE_list(f"{BENCHMARK_NAME}_batch_size", [], "")

prompt_sizes_list = flags.DEFINE_list(f"{BENCHMARK_NAME}_prompt_size", [], "")

flags.DEFINE_integer(f"{BENCHMARK_NAME}_output_tokens", 256, "")

flags.DEFINE_string(f"{BENCHMARK_NAME}_threads_range", "", "")

flags.DEFINE_bool(f"{BENCHMARK_NAME}_flash_attention", False, "")

flags.DEFINE_float(f"{BENCHMARK_NAME}_timeout", 100.00, "")

flags.DEFINE_bool(f"{BENCHMARK_NAME}_stability", False, "")

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_memory_placement",
    "none",
    "memory placement policy - 'local','interleave' or 'none'",
)

@dataclass
class LlamaResult:
    """Class that represents llama results."""

    n_proc: list[int]
    n_threads: list[int]
    batch_size: list[int]
    prompt_size: list[int]
    output_tokens: list[int]
    pp_throughput_tps: list[float]
    pp_max_latency_sec: list[float]
    pp_avg_latency_sec: list[float]
    tg_throughput_tps: list[float]
    tg_max_latency_sec: list[float]
    tg_avg_latency_sec: list[float]
    tg_max_per_token_latency_sec: list[float]
    tg_avg_per_token_latency_sec: list[float]
    e2e_max_latency_sec: list[float]
    e2e_avg_latency_sec: list[float]
    pptg_throughput_tps: list[float]
    concurrency: list[float]
    min_tps_per_user_per_process: list[float]
    start: list[str]
    finish: list[str]

    @classmethod
    def parse_llama_results(cls, llama_results: str) -> "LlamaResult":
        """Parse llama result textfile and return results.
        Args:
          llama_results: Str output of running llama.
        Returns:
        """
        llama_csv_result = _parse_csv(llama_results)

        return cls(
            n_proc=llama_csv_result.n_proc,
            n_threads=llama_csv_result.n_threads,
            batch_size=llama_csv_result.batch_size,
            prompt_size=llama_csv_result.prompt_size,
            output_tokens=llama_csv_result.output_tokens,
            pp_throughput_tps=llama_csv_result.pp_throughput_tps,
            pp_max_latency_sec=llama_csv_result.pp_max_latency_sec,
            pp_avg_latency_sec=llama_csv_result.pp_avg_latency_sec,
            tg_throughput_tps=llama_csv_result.tg_throughput_tps,
            tg_max_latency_sec=llama_csv_result.tg_max_latency_sec,
            tg_avg_latency_sec=llama_csv_result.tg_avg_latency_sec,
            tg_max_per_token_latency_sec=llama_csv_result.tg_max_per_token_latency_sec,
            tg_avg_per_token_latency_sec=llama_csv_result.tg_avg_per_token_latency_sec,
            e2e_max_latency_sec=llama_csv_result.e2e_max_latency_sec,
            e2e_avg_latency_sec=llama_csv_result.e2e_avg_latency_sec,
            pptg_throughput_tps=llama_csv_result.pptg_throughput_tps,
            concurrency=llama_csv_result.concurrency,
            min_tps_per_user_per_process=llama_csv_result.min_tps_per_user_per_process,
            start=llama_csv_result.start,
            finish=llama_csv_result.finish,
        )

    def get_samples(self, metadata: Dict[str, Any]) -> List[sample.Sample]:
        """Return this result as a list of samples."""
        all_samples = []
        samples = []
        metadata_new = {}
        for count_n_proc, _ in enumerate(self.n_proc):
            metadata_new["n_proc"] = int(self.n_proc[count_n_proc])
            metadata_new["n_threads"] = int(self.n_threads[count_n_proc])
            metadata_new["batch_size"] = int(self.batch_size[count_n_proc])
            metadata_new["prompt_size"] = int(self.prompt_size[count_n_proc])
            metadata_sample = metadata | metadata_new
            samples = [
                sample.Sample(
                    "Processes",
                    self.n_proc[count_n_proc],
                    "",
                    metadata_sample,
                ),
                sample.Sample(
                    "Threads",
                    self.n_threads[count_n_proc],
                    "",
                    metadata_sample,
                ),
                sample.Sample(
                    "Batch Size",
                    self.batch_size[count_n_proc],
                    "",
                    metadata_sample,
                ),
                sample.Sample(
                    "prompt processing throughput",
                    self.pp_throughput_tps[count_n_proc],
                    "tps",
                    metadata_sample,
                ),
                sample.Sample(
                    "Maximum prompt processing latency",
                    self.pp_max_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Average prompt processing latency",
                    self.pp_avg_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "token generation throughput",
                    self.tg_throughput_tps[count_n_proc],
                    "tps",
                    metadata_sample,
                ),
                sample.Sample(
                    "Maximum token generation latency",
                    self.tg_max_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Average token generation latency",
                    self.tg_avg_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Maximum per token latency",
                    self.tg_max_per_token_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Average per token latency",
                    self.tg_avg_per_token_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Maximum end to end latency",
                    self.e2e_max_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Average end to end latency",
                    self.e2e_avg_latency_sec[count_n_proc],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "Concurrency",
                    int(self.concurrency[count_n_proc]),
                    "",
                    metadata_sample,
                ),
            ]
            all_samples.extend(samples)
        return all_samples


def _parse_csv(llama_results: str) -> LlamaResult:
    """Parses the output
    Yields:
    (n_proc,n_threads,batch_size,prompt_size,output_tokens,
    pp_throughput_tps,pp_max_latency_sec,pp_avg_latency_sec,
    tg_throughput_tps,tg_max_latency_sec,tg_avg_latency_sec,
    tg_max_per_token_latency_sec,tg_avg_per_token_latency_sec,
    e2e_max_latency_sec,e2e_avg_latency_sec,
    pp+tg_throughput_tps,concurrency,min_tps_per_process,start,finish) tuples.
    """
    n_proc: list[int] = []
    n_threads: list[int] = []
    batch_size: list[int] = []
    prompt_size: list[int] = []
    output_tokens: list[int] = []
    pp_throughput_tps: list[float] = []
    pp_max_latency_sec: list[float] = []
    pp_avg_latency_sec: list[float] = []
    tg_throughput_tps: list[float] = []
    tg_max_latency_sec: list[float] = []
    tg_avg_latency_sec: list[float] = []
    tg_max_per_token_latency_sec: list[float] = []
    tg_avg_per_token_latency_sec: list[float] = []
    e2e_max_latency_sec: list[float] = []
    e2e_avg_latency_sec: list[float] = []
    pptg_throughput_tps: list[float] = []
    concurrency: list[float] = []
    min_tps_per_user_per_process: list[float] = []
    start: list[str] = []
    finish: list[str] = []
    csv_fp = six.StringIO(str(llama_results))
    reader = csv.DictReader(csv_fp)
    try:
        if FLAGS.sla_per_process:
            min_tps_per_user_per_process_col = "min_tps_per_user_per_process"
        else:
            min_tps_per_user_per_process_col = "tps_per_user"
    except AttributeError:
        min_tps_per_user_per_process_col = "tps_per_user"

    if frozenset(reader.fieldnames) != frozenset(
        [
            "n_proc",
            "n_threads",
            "batch_size",
            "prompt_size",
            "output_tokens",
            "pp_throughput_tps",
            "pp_max_latency_sec",
            "pp_avg_latency_sec",
            "tg_throughput_tps",
            "tg_max_latency_sec",
            "tg_avg_latency_sec",
            "tg_max_per_token_latency_sec",
            "tg_avg_per_token_latency_sec",
            "e2e_max_latency_sec",
            "e2e_avg_latency_sec",
            "pp+tg_throughput_tps",
            "concurrency",
            min_tps_per_user_per_process_col,
            "start",
            "finish",
        ]
    ):
        raise ValueError(f"Test Failed: {llama_results}")
    for row in reader:
        n_proc.append(row["n_proc"])
        n_threads.append(row["n_threads"])
        batch_size.append(row["batch_size"])
        prompt_size.append(row["prompt_size"])
        output_tokens.append(row["output_tokens"])
        pp_throughput_tps.append(row["pp_throughput_tps"])
        pp_max_latency_sec.append(row["pp_max_latency_sec"])
        pp_avg_latency_sec.append(row["pp_avg_latency_sec"])
        tg_throughput_tps.append(row["tg_throughput_tps"])
        tg_max_latency_sec.append(row["tg_max_latency_sec"])
        tg_avg_latency_sec.append(row["tg_avg_latency_sec"])
        tg_max_per_token_latency_sec.append(row["tg_max_per_token_latency_sec"])
        tg_avg_per_token_latency_sec.append(row["tg_avg_per_token_latency_sec"])
        e2e_max_latency_sec.append(row["e2e_max_latency_sec"])
        e2e_avg_latency_sec.append(row["e2e_avg_latency_sec"])
        pptg_throughput_tps.append(row["pp+tg_throughput_tps"])
        concurrency.append(row["concurrency"])
        min_tps_per_user_per_process.append(row[min_tps_per_user_per_process_col])
        start.append(row["start"])
        finish.append(row["finish"])
    return LlamaResult(
        n_proc,
        n_threads,
        batch_size,
        prompt_size,
        output_tokens,
        pp_throughput_tps,
        pp_max_latency_sec,
        pp_avg_latency_sec,
        tg_throughput_tps,
        tg_max_latency_sec,
        tg_avg_latency_sec,
        tg_max_per_token_latency_sec,
        tg_avg_per_token_latency_sec,
        e2e_max_latency_sec,
        e2e_avg_latency_sec,
        pptg_throughput_tps,
        concurrency,
        min_tps_per_user_per_process,
        start,
        finish,
    )


class LlamaProcessLogResults:
    """Class that represents llama results."""

    proc_no: list[int] = []
    threads_no: list[int] = []
    batch_size: list[int] = []
    prompt_size: list[int] = []
    output_tokens: list[int] = []
    KV_cache_size: list[int] = []
    time_to_first_token: list[float] = []
    prompt_processing_throughput: list[float] = []
    token_gen_latency: list[float] = []
    token_gen_throughput: list[float] = []
    e2e_latency: list[float] = []
    row_index: list[int] = []

    def __init__(self, llama_process_logs_file):
        """
        initializes LlamaProcessLogResults object
        """
        try:
            df = pd.read_csv(
                llama_process_logs_file, delimiter="|", on_bad_lines="skip"
            )
            for index, row in df.iterrows():
                if row["Process_number"] != "Process_number":
                    self.proc_no.append(int(row["Process_number"]))
                    self.threads_no.append(int(row["threads_per_process"]))
                    self.batch_size.append(int(row["batch_size"]))
                    self.prompt_size.append(int(row["prompt_tokens_per_batch"]))
                    self.output_tokens.append(int(row["tokens_generated_per_batch"]))
                    self.KV_cache_size.append(int(row["KV_cache_size"]))
                    self.time_to_first_token.append(float(row["time_to_first_token"]))
                    self.prompt_processing_throughput.append(
                        float(row["prompt_processing_throughput"])
                    )
                    self.token_gen_latency.append(float(row["token_gen_latency"]))
                    self.token_gen_throughput.append(float(row["token_gen_throughput"]))
                    self.e2e_latency.append(float(row["total_time"]))
                    self.row_index.append(index)
        except TypeError as e:
            print(e)

    def get_samples(self, metadata: Dict[str, Any]) -> List[sample.Sample]:
        """Return this result as a list of samples."""
        all_samples = []
        samples = []
        metadata_new = {}
        for count_row, _ in enumerate(self.row_index):
            metadata_new["proc_no"] = self.proc_no[count_row]
            metadata_new["threads"] = self.threads_no[count_row]
            metadata_new["batch_size"] = self.batch_size[count_row]
            metadata_new["prompt_size"] = self.prompt_size[count_row]
            metadata_new["output_tokens"] = self.output_tokens[count_row]
            metadata_new["KV_cache_size"] = self.KV_cache_size[count_row]
            metadata_sample = metadata | metadata_new
            samples = [
                sample.Sample(
                    "process time to first token",
                    self.time_to_first_token[count_row],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "process prompt processing throughput",
                    self.prompt_processing_throughput[count_row],
                    "tps",
                    metadata_sample,
                ),
                sample.Sample(
                    "process token generation throughput",
                    self.token_gen_throughput[count_row],
                    "tps",
                    metadata_sample,
                ),
                sample.Sample(
                    "process token generation latency",
                    self.token_gen_latency[count_row],
                    "sec",
                    metadata_sample,
                ),
                sample.Sample(
                    "process end to end latency",
                    self.e2e_latency[count_row],
                    "sec",
                    metadata_sample,
                ),
            ]
            all_samples.extend(samples)
        return all_samples


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
    servers = benchmark_spec.vm_groups["servers"]
    server = servers[0]
    threads_validation.check_threads_validity(server, BENCHMARK_NAME)
    if llm_utils_internal:
        llm_utils_internal.validate_exclusive_run_modes()
    docker_package.Install(server)

    # Case 1: Build image
    if FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_dir"].value:
        docker_build = docker_package.build_docker(server)
        if not docker_build:
            raise ValueError(
                "Docker build failed, please check if env file provided for Dockerfile is correct"
            )
    else:
        # Case 2: Pull docker image
        docker_pull = docker_package.pull_docker(server)
        if not docker_pull:
            raise ValueError(
                "Docker cannot be pulled, please check docker image repository and image version"
            )
    ampere_llama.download_model(server)
    utils_benchmark_file = "ampere/pkb/utils/run_llama.py"
    server.RemoteCopy(utils_benchmark_file, download_utils.INSTALL_DIR)
    model_volume = posixpath.join(download_utils.INSTALL_DIR, "models")
    output_dir = posixpath.join(download_utils.INSTALL_DIR, "out_dir")
    server.RemoteCommand(f"mkdir -p {output_dir}")
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value = [
            model_volume,
            f"{download_utils.INSTALL_DIR}/run_llama.py",
            output_dir,
        ]
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_volume_mountpoints"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_volume_mountpoints"].value = [
            "/models/",
            "/utils/benchmark.py",
            "/out_dir/",
        ]
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_name"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_name"].value = "llama_aio_container"
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_daemon"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_daemon"].value = True
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_privileged_docker"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_privileged_docker"].value = False
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_bash_command"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_bash_command"].value = (
            "--entrypoint /bin/sh -it"
        )
    docker_package.run_docker(server)

    cmd_exec_installs = "apt-get update -y &&" " apt-get install -y numactl"
    FLAGS[f"{docker_package.PACKAGE_NAME}_shell_type"].value = "bash"
    FLAGS[f"{docker_package.PACKAGE_NAME}_exec_command"].value = cmd_exec_installs
    docker_package.exec_docker(server)


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
    model = model_names.value[0]
    sorted_prompt_sizes_list = sorted(prompt_sizes_list.value)

    all_samples = None
    if llm_utils_internal:
        all_samples = llm_utils_internal.controller(
            model, sorted_prompt_sizes_list, server
        )
    if not all_samples:
        _run(server, model)
        out_dir = posixpath.join(download_utils.INSTALL_DIR, "out_dir")
        server.RemoteCopy(vm_util.GetTempDir(), out_dir, False)
        all_samples = _parse_output_files()
    return all_samples


def _run(vm: BaseVirtualMachine, model):
    """
    Executes the Run stage
    """
    results_llama = []
    threads_range = FLAGS[f"{BENCHMARK_NAME}_threads_range"].value
    threads_range_list = threads_validation.parse_threads_range(vm.cpu_arch, threads_range)
    available_threads = " ".join(map(str, threads_range_list))
    for prompt_size in sorted(prompt_sizes_list.value):
        for batch_size in sorted(batch_sizes_list.value):
            for num_threads in sorted(threads_per_process_list.value):
                num_available_threads = len(threads_range_list)
                num_processes = int(int(num_available_threads) / int(num_threads))
                if FLAGS[f"{BENCHMARK_NAME}_number_of_models"].value > 0:
                    num_processes = FLAGS[f"{BENCHMARK_NAME}_number_of_models"].value
                expt_dict = {
                        'benchmark': 'ampere_llama_benchmark',
                        'num_threads': num_threads,
                        'num_processes': num_processes,
                        'batch_size': batch_size,
                        'model': model,
                        'prompt_size': prompt_size,
                        'vm': vm,
                        'available_threads': available_threads,
                        'tps_per_user':0.0,
                        }
                LlamaBase = llm_base_utils.LlamaExperiment(expt_dict)
                results_llama.extend(
                    LlamaBase.llama_batched_bench_base_run(
                        num_processes, num_threads, batch_size, model, prompt_size, vm
                    ).results
                )
    return results_llama


def update_llama_metadata(meta_data: dict):
    if FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_dir"].value:
        docker_version = (
            FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_image"].value
            + "-"
            + FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_image_version"].value
        )
    else:
        docker_version = (
            FLAGS[f"{docker_package.PACKAGE_NAME}_image"].value
            + "-"
            + FLAGS[f"{docker_package.PACKAGE_NAME}_image_version"].value
        )
    meta_data["gpu"] = FLAGS[f"{docker_package.PACKAGE_NAME}_gpus"].value
    meta_data["docker_version"] = docker_version
    meta_data["gpu_device_id"] = FLAGS[
            f"{docker_package.PACKAGE_NAME}_gpu_device_id"].value
    return meta_data


def collect_all_results_samples(all_files_samples):
    """function to collect all results in sample for json output"""
    csv_files = glob.glob(vm_util.GetTempDir() + "/*.csv")
    if csv_files:
        for filename in csv_files:
            csv_file = os.path.basename(filename)
            model_name = csv_file.split("@")
            metadata = {
                "model": model_name[0],
            }
            metadata = update_llama_metadata(metadata)
            csv_file_llama = posixpath.join(vm_util.GetTempDir(), csv_file)
            with open(csv_file_llama, "r", encoding="utf-8") as output:
                llama_output_data = output.read()
            results = LlamaResult.parse_llama_results(llama_output_data)
            all_files_samples.extend(results.get_samples(metadata))
    return all_files_samples


def collect_process_log_samples(all_files_samples):
    """function to collect all process wise logs in sample for json output"""
    process_log_files = glob.glob(vm_util.GetTempDir() + "/*.log")
    if process_log_files:
        for log_filename in process_log_files:
            process_log_file = os.path.basename(log_filename)
            if "@" in process_log_file:
                model_name = process_log_file.split("@")
                metadata = {
                    "model": model_name[0],
                }
                metadata = update_llama_metadata(metadata)
                log_file_llama = posixpath.join(vm_util.GetTempDir(), process_log_file)
                log_results = LlamaProcessLogResults(log_file_llama)
                all_files_samples.extend(log_results.get_samples(metadata))
    return all_files_samples


def _parse_output_files():
    """
    Parses CSV output files from the temporary directory and extracts benchmark samples.

    This function looks for CSV files in the temporary directory, assumes each file is
    named with the model name followed by an '@' symbol (e.g., 'llama@timestamp.csv'),
    and reads the file contents. It uses `LlamaResult.parse_llama_results` to parse the
    results and extracts samples, augmenting the provided benchmark metadata with the
    model name.

    Returns:
        list: A list of sample dictionaries extracted from all parsed CSV files.
    """

    all_files_samples = []
    all_files_samples = collect_all_results_samples(all_files_samples)
    all_files_samples = collect_process_log_samples(all_files_samples)
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
    docker_package.Uninstall(server)
