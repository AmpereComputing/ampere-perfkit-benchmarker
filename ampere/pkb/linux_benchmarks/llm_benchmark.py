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

This is a set of benchmarks that measures performance of llama

"""

import csv
import posixpath
from dataclasses import dataclass
import logging
import os
import uuid
import time
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

INSTABILITY_THRESHOLD = 1.01

model_names = flags.DEFINE_list(f"{BENCHMARK_NAME}_model_names", [], "")

flags.DEFINE_string(f"{BENCHMARK_NAME}_llama_exe_path", None, "")

threads_per_process_list = flags.DEFINE_list(
    f"{BENCHMARK_NAME}_threads_per_process", [], ""
)

batch_sizes_list = flags.DEFINE_list(f"{BENCHMARK_NAME}_batch_size", [], "")

prompt_sizes_list = flags.DEFINE_list(f"{BENCHMARK_NAME}_prompt_size", [], "")

TOKENS = flags.DEFINE_integer(f"{BENCHMARK_NAME}_output_tokens", 256, "")

flags.DEFINE_string(f"{BENCHMARK_NAME}_threads_range", "", "")

flags.DEFINE_bool(f"{BENCHMARK_NAME}_flash_attention", False, "")

flags.DEFINE_float(f"{BENCHMARK_NAME}_timeout", 100.00, "")

flags.DEFINE_bool(f"{BENCHMARK_NAME}_stability", False, "")


def parse_threads_range(threads_range: str) -> list[int]:
    """
    Parses a thread range string into a list of individual thread indices.

    Args:
        threads_range (str): A string specifying thread index ranges.

    Returns:
        list[int]: A list of individual thread indices.
    """

    logging.info("threads_range: %s", threads_range)
    threads_range = [s.split("-") for s in threads_range.split(",")]
    logging.info("threads_range: %s", threads_range)
    if not all(len(s) == 2 for s in threads_range):
        raise ValueError(
            "Format of --threads_range argument must be '{idx}-{idx},{idx}-{idx},...', "
            "e.g. '88-88' to use just thread idx 88"
        )
    designated_threads = []
    for s in threads_range:
        s_0, s_1 = int(s[0]), int(s[1])
        if s_1 < s_0:
            raise ValueError(
                f"Range {s_0}-{s_1} is not valid, second value has to be equal to or"
                "greater than the first value"
            )
        designated_threads += list(range(s_0, s_1 + 1))
    logging.info("designated_threads: %s", designated_threads)
    return designated_threads


def check_threads_validity():
    """
    Validates the requested thread count against available threads.

    This function checks if the number of threads specified in the benchmark
    configuration (through `threads_range` and `threads_per_process`) does not
    exceed the available threads after parsing the `threads_range` argument.
    If the requested number of threads exceeds the available ones, a
    `ValueError` is raised.

    It accesses the global flags to get the `threads_range` and `threads_per_process`
    values, parses the thread range, and then compares the total available threads
    with the requested threads per process.

    """

    threads_range = FLAGS[f"{BENCHMARK_NAME}_threads_range"].value
    threads_per_proc_list = FLAGS[f"{BENCHMARK_NAME}_threads_per_process"].value
    available_threads_list = parse_threads_range(threads_range)
    if len(available_threads_list) < max(threads_per_proc_list):
        raise ValueError(
            f"Requested number of threads ({max(threads_per_proc_list)})"
            f"exceeds threads available ({len(available_threads_list)})"
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
    pp+tg_throughput_tps,concurrency,start,finish) tuples.
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
    start: list[str] = []
    finish: list[str] = []
    csv_fp = six.StringIO(str(llama_results))
    reader = csv.DictReader(csv_fp)
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

    def __init__(
        self,
        llama_process_logs_file
        ):
        """
        initializes LlamaProcessLogResults object
        """
        try:
            df = pd.read_csv(llama_process_logs_file, delimiter='|',on_bad_lines='skip')
            for index, row in df.iterrows():
                if row['Process_number'] != 'Process_number':
                    self.proc_no.append(int(row['Process_number']))
                    self.threads_no.append(int(row['threads_per_process']))
                    self.batch_size.append(int(row['batch_size']))
                    self.prompt_size.append(int(row['prompt_tokens_per_batch']))
                    self.output_tokens.append(int(row['tokens_generated_per_batch']))
                    self.KV_cache_size.append(int(row['KV_cache_size']))
                    self.time_to_first_token.append(float(row['time_to_first_token']))
                    self.prompt_processing_throughput.append(float(row['prompt_processing_throughput']))
                    self.token_gen_latency.append(float(row['token_gen_latency']))
                    self.token_gen_throughput.append(float(row['token_gen_throughput']))
                    self.e2e_latency.append(float(row['total_time']))
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
    check_threads_validity()
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

    cmd_exec_installs = ("apt-get update -y &&"
                         " apt-get install -y numactl")
    FLAGS[f"{docker_package.PACKAGE_NAME}_shell_type"].value = (
            "bash"
            )
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
    benchmark_metadata = {}
    _run(server)
    out_dir = posixpath.join(download_utils.INSTALL_DIR, "out_dir")
    server.RemoteCopy(vm_util.GetTempDir(), out_dir, False)    
    return _parse_output_files(benchmark_metadata)


def _run(vm):
    """
    Executes the Run stage
    """
    threads_range = FLAGS[f"{BENCHMARK_NAME}_threads_range"].value
    num_available_threads = len(parse_threads_range(threads_range))
    llama_exe_path = FLAGS[f"{BENCHMARK_NAME}_llama_exe_path"].value
    stability = FLAGS[f"{BENCHMARK_NAME}_stability"].value
    volumes = FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value
    volume_mountpoints = FLAGS[
        f"{docker_package.PACKAGE_NAME}_volume_mountpoints"
    ].value
    output_dir = volumes[2]
    docker_out_dir = volume_mountpoints[2]
    if FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_dir"].value:
        LD_LIBRARY_PATH = ""
    else:
        LD_LIBRARY_PATH = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{llama_exe_path} &&"
    args = {}
    for model in model_names.value:
        for prompt_size in sorted(prompt_sizes_list.value):
            for batch_size in sorted(batch_sizes_list.value):
                for num_threads in sorted(threads_per_process_list.value):
                    num_processes = int(num_available_threads / num_threads)
                    current_case = f"{num_processes} x {num_threads} "
                    current_case += f"[proc x threads per proc], bs = {batch_size}"
                    logging.info("\nRunning %s", current_case)
                    args = {
                        "model": model,
                        "prompt_size": prompt_size,
                        "tokens": TOKENS.value,
                        "batch_size": batch_size,
                        "num_processes": num_processes,
                        "num_threads": num_threads,
                        "stability": stability,
                    }
                    results = Results(vm, args)
                    while not results.is_stable():
                        docker_logs_dir = os.path.join(
                            docker_out_dir, str(uuid.uuid4())
                        )
                        logs_dir = os.path.join(
                            output_dir, docker_logs_dir.split("/")[-1]
                        )
                        cmd = (
                            f"cd / && "
                            f" {LD_LIBRARY_PATH}"
                            f" python3 utils/benchmark.py --exe_path {llama_exe_path}/llama-batched-bench "
                            f" --output_dir {docker_logs_dir} -m models/{model}"
                            f" -n {str(num_processes)} "
                            f"-t {str(num_threads)} -b {str(batch_size)} -p {str(prompt_size)}"
                            f" -k {str(TOKENS.value)} -r {threads_range}"
                        )
                        if FLAGS[f"{BENCHMARK_NAME}_stability"].value:
                            cmd += " --stability"
                        if FLAGS[f"{BENCHMARK_NAME}_flash_attention"].value:
                            cmd += " -fa 1"
                        FLAGS[f"{docker_package.PACKAGE_NAME}_shell_type"].value = (
                            "bash"
                        )
                        FLAGS[f"{docker_package.PACKAGE_NAME}_exec_command"].value = cmd
                        start = time.time()
                        docker_package.exec_docker(vm)
                        finish = time.time()
                        results.summarize(
                            vm, logs_dir, start, finish, vm_util.GetTempDir()
                        )
                    results.save_csv(vm_util.GetTempDir())


def _parse_output_files(benchmark_metadata):
    """
    Parses CSV output files from the temporary directory and extracts benchmark samples.

    This function looks for CSV files in the temporary directory, assumes each file is
    named with the model name followed by an '@' symbol (e.g., 'llama@timestamp.csv'),
    and reads the file contents. It uses `LlamaResult.parse_llama_results` to parse the
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
    if FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_dir"].value:
        docker_version = FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_image"].value + "-" + FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_image_version"].value 
    else:
        docker_version = FLAGS[f"{docker_package.PACKAGE_NAME}_image"].value + "-" + FLAGS[f"{docker_package.PACKAGE_NAME}_image_version"].value
    csv_files = glob.glob(vm_util.GetTempDir() + "/*.csv")
    if csv_files:
        for filename in csv_files:
            csv_file = os.path.basename(filename)
            model_name = csv_file.split("@")
            metadata = {
                "model": model_name[0],
                "docker_version": docker_version,
            }
            csv_file_llama = posixpath.join(vm_util.GetTempDir(), csv_file)
            with open(csv_file_llama, "r", encoding="utf-8") as output:
                llama_output_data = output.read()
            results = LlamaResult.parse_llama_results(llama_output_data)
            benchmark_metadata = benchmark_metadata | metadata
            all_files_samples.extend(results.get_samples(benchmark_metadata))
    process_log_files = glob.glob(vm_util.GetTempDir() + "/*.log")
    if process_log_files:
        for log_filename in process_log_files:
            process_log_file = os.path.basename(log_filename)
            if "@" in process_log_file:
                model_name = process_log_file.split("@")
                metadata = {
                        "model": model_name[0],
                        "docker_version": docker_version,
                        }
                log_file_llama = posixpath.join(vm_util.GetTempDir(), process_log_file)
                log_results = LlamaProcessLogResults(log_file_llama)
                benchmark_metadata = benchmark_metadata | metadata
                all_files_samples.extend(log_results.get_samples(benchmark_metadata))
    return all_files_samples


class Results:
    """
    A container for storing and managing benchmark results and metadata.

    Attributes:
        vm (BaseVirtualMachine): The virtual machine used for the benchmark run.
        params (dict): A dictionary of benchmark parameters (e.g., batch size, prompt size).
        tg_runs (list): A list to store token generation run measurements.
        lines (list): A list to store output lines or logs related to the benchmark.
        results (list): A list to accumulate parsed or final benchmark results.
    """

    def __init__(self, vm: BaseVirtualMachine, params):
        self.params = params
        self.vm = vm
        self.tg_runs = []
        self.lines = []
        self.results = []

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
        time_to_first_token_list = []
        token_generation_latency_list = []
        e2e_latency_list = []
        self.lines = []
        for n in range(self.params["num_processes"]):
            line, _ = vm.RemoteCommand(f"head -6 {logs_dir}/log_{n} | tail -1")
            results = line.strip()[:-1].split("|")
            line = str(n) + "|" + str(self.params["num_threads"]) + line.strip()[:-1] + "\n"
            self.lines.append(line)
            prompt_size = int(results[1])
            assert prompt_size == self.params["prompt_size"]
            tokens_generated = int(results[2])
            assert tokens_generated == TOKENS.value
            batch_size = int(results[3])
            assert batch_size == self.params["batch_size"]
            time_to_first_token_list.append(float(results[5]))
            token_generation_latency_list.append(float(results[7]))
            e2e_latency_list.append(float(results[9]))
        pp_throughput = sum(
            self.params["batch_size"] * self.params["prompt_size"] / time_to_first_token
            for time_to_first_token in time_to_first_token_list
        )
        avg_pp_latency = sum(time_to_first_token_list) / len(time_to_first_token_list)
        max_pp_latency = max(time_to_first_token_list)
        tg_throughput = sum(
            self.params["batch_size"] * TOKENS.value / lat
            for lat in token_generation_latency_list
        )
        avg_tg_latency = sum(token_generation_latency_list) / len(token_generation_latency_list)
        max_tg_latency = max(token_generation_latency_list)
        tg_per_token_lats = [
            lat / TOKENS.value for lat in token_generation_latency_list
        ]
        avg_tg_per_token_latency = sum(tg_per_token_lats) / len(tg_per_token_lats)
        max_tg_per_token_latency = max(tg_per_token_lats)
        avg_total_speed = (
            self.params["num_processes"]
            * self.params["batch_size"]
            * (self.params["prompt_size"] + TOKENS.value)
            / max(
                time_to_first_token + token_generation_lat
                for time_to_first_token, token_generation_lat in zip(
                    time_to_first_token_list, token_generation_latency_list
                )
            )
        )
        avg_e2e_latency = sum(e2e_latency_list) / len(e2e_latency_list)
        max_e2e_latency = max(e2e_latency_list)

        self.tg_runs.append(tg_throughput)
        self.results.append(
            [
                self.params["num_processes"],
                self.params["num_threads"],
                self.params["batch_size"],
                self.params["prompt_size"],
                TOKENS.value,
                pp_throughput,
                max_pp_latency,
                avg_pp_latency,
                tg_throughput,
                max_tg_latency,
                avg_tg_latency,
                max_tg_per_token_latency,
                avg_tg_per_token_latency,
                max_e2e_latency,
                avg_e2e_latency,
                avg_total_speed,
                self.params["batch_size"] * self.params["num_processes"],
                start,
                finish,
            ]
        )
        log_filename = (
            f"{save_logs_dir}/{self.params['model'].split('/')[-1]}@"
            f"PP{str(self.params['prompt_size'])}@"
            f"TG{str(TOKENS.value)}@{len(self.tg_runs)}.log"
        )
        with open(log_filename, "a", encoding="utf-8") as f1:
            f1.writelines(
                    [
                        "Process_number|",
                        "threads_per_process|",
                        "prompt_tokens_per_batch|",
                        "tokens_generated_per_batch|",
                        "batch_size|",
                        "KV_cache_size|",
                        "time_to_first_token|",
                        "prompt_processing_throughput|",
                        "token_gen_latency|",
                        "token_gen_throughput|",
                        "total_time|",
                        "total_speed\n",
                    ]
                    )
            f1.writelines(self.lines)
        logging.info("Logs saved in %s", log_filename)

    def calc_avg_tg(self, n):
        """Calculate the average of the first n throughput runs."""
        return sum(self.tg_runs[:n]) / n

    def is_stable(self):
        """
        checks stability of run
        """
        logging.info(self.params)
        runs_completed = len(self.tg_runs)
        if self.params["stability"] is False and runs_completed > 0:
            return True
        if runs_completed < 3:
            return False
        prev_avg_tg = self.calc_avg_tg(runs_completed - 1)
        avg_tg = self.calc_avg_tg(runs_completed)
        return max(prev_avg_tg / avg_tg, avg_tg / prev_avg_tg) <= INSTABILITY_THRESHOLD

    def save_csv(self, save_dir):
        """
        saves csv in save_dir
        """
        results_filename = (
            f"{save_dir}/{self.params['model'].split('/')[-1]}@"
            f"PP{str(self.params['prompt_size'])}@"
            f"TG{str(TOKENS.value)}.csv"
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
                        "start",
                        "finish",
                    ]
                )
            if self.params["stability"] is True:
                avg_tg = sum(self.tg_runs) / len(self.tg_runs)
                tg_diff = [abs(avg_tg - tg) for tg in self.tg_runs]
                writer.writerow(self.results[tg_diff.index(min(tg_diff))])
            else:
                writer.writerow(self.results[0])
        logging.info("Result saved in %s", results_filename)


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
