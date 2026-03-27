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

"""Module for sla llama experiments"""

import logging
import csv
import os
import uuid
import time
from absl import flags
from perfkitbenchmarker import vm_util
from perfkitbenchmarker.virtual_machine import VirtualMachine
from ampere.pkb.linux_packages import docker as docker_package
from ampere.pkb.utils import threads_validation

FLAGS = flags.FLAGS
INSTABILITY_THRESHOLD = 1.05


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

    def __init__(self, vm: VirtualMachine, params):
        self.params = params
        self.vm = vm
        self.tg_runs = []
        self.lines = []
        self.results = []
        self.tps_per_process_list = []

    def summarize(self, vm: VirtualMachine, logs_dir, start, finish, save_logs_dir):
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
        tokens = FLAGS.ampere_llama_benchmark_output_tokens
        time_to_first_token_list = []
        token_generation_latency_list = []
        e2e_latency_list = []
        self.tps_per_process_list = []
        self.lines = []
        for n in range(self.params["num_processes"]):
            line, _ = vm.RemoteCommand(f"head -6 {logs_dir}/log_{n} | tail -1")
            results = line.strip()[:-1].split("|")
            tg_tpt_process = float(results[8])
            #tps_per_process = tg_tpt/ batch_size
            tps_per_process = tg_tpt_process / int(results[3])
            self.tps_per_process_list.append(tps_per_process)
            line = str(n) + "|" + str(self.params["num_threads"]) + line.strip()[:-1] + f"|{tps_per_process}\n"
            self.lines.append(line)
            prompt_size = int(results[1])
            assert prompt_size == self.params["prompt_size"]
            tokens_generated = int(results[2])
            assert tokens_generated == tokens
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
            self.params["batch_size"] * tokens / lat
            for lat in token_generation_latency_list
        )
        avg_tg_latency = sum(token_generation_latency_list) / len(token_generation_latency_list)
        max_tg_latency = max(token_generation_latency_list)
        tg_per_token_lats = [
            lat / tokens for lat in token_generation_latency_list
        ]
        avg_tg_per_token_latency = sum(tg_per_token_lats) / len(tg_per_token_lats)
        max_tg_per_token_latency = max(tg_per_token_lats)
        avg_total_speed = (
            self.params["num_processes"]
            * self.params["batch_size"]
            * (self.params["prompt_size"] + tokens)
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
        concurrency = self.params["batch_size"] * self.params["num_processes"]
        try:
            if FLAGS.sla_per_process:
                save_tps = min(self.tps_per_process_list)
            else:
                save_tps = tg_throughput / concurrency
        except AttributeError as aerr:
            logging.info(f"{aerr}, sla is applied to tg_throughput / concurrency")
            save_tps = tg_throughput / concurrency        
        self.results.append(
            [
                self.params["num_processes"],
                self.params["num_threads"],
                self.params["batch_size"],
                self.params["prompt_size"],
                tokens,
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
                concurrency,
                save_tps,
                start,
                finish,
            ]
        )
        log_filename = (
            f"{save_logs_dir}/{self.params['model'].split('/')[-1]}@"
            f"PP{str(self.params['prompt_size'])}@"
            f"TG{str(tokens)}@{len(self.tg_runs)}.log"
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
                        "total_speed|",
                        "tps_per_user\n",
                    ]
                    )
            f1.writelines(self.lines)
        logging.info("Logs saved in %s", log_filename)
        return save_tps


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
        tokens = FLAGS.ampere_llama_benchmark_output_tokens
        results_filename = (
            f"{save_dir}/{self.params['model'].split('/')[-1]}@"
            f"PP{str(self.params['prompt_size'])}@"
            f"TG{str(tokens)}.csv"
        )
        if os.path.exists(results_filename):
            first_write = False
        else:
            first_write = True
        try:
            if FLAGS.sla_per_process:
                save_tps_column = "min_tps_per_user_per_process"
            else:
                save_tps_column = "tps_per_user"
        except AttributeError as aerr:
            logging.info(f"{aerr}, sla is applied to tg_throughput / concurrency")
            save_tps_column = "tps_per_user"

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
                        f"{save_tps_column}",
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

class LlamaExperiment:
    """executes llama-batched-bench """
    benchmark: str
    num_threads: int
    num_processes: int
    batch_size: int
    model: str
    prompt_size: int
    output_tokens_size: int
    pp_throughput_tps: float
    pp_max_latency_sec: float
    tg_throughput_tps: float
    tg_max_latency_sec: float
    tg_max_per_token_latency_sec: float
    e2e_max_latency_sec: float
    pptg_throughput_tps: float
    concurrency: int    
    num_available_threads: int
    available_threads: str
    vm: VirtualMachine
    tps_per_user: float

    def __init__(self, expt_details: dict):
        """
        expt_details is dict containing all required arguments
        """
        tokens = FLAGS.ampere_llama_benchmark_output_tokens
        self.benchmark = expt_details['benchmark']
        self.num_threads = expt_details['num_threads']
        self.num_processes = expt_details['num_processes']
        self.batch_size = expt_details['batch_size']
        self.model = expt_details['model']
        self.prompt_size = expt_details['prompt_size']
        self.output_tokens_size = tokens
        self.pp_throughput_tps = 0.0
        self.pp_max_latency_sec = 0.0
        self.tg_throughput_tps = 0.0
        self.tg_max_latency_sec = 0.0
        self.tg_max_per_token_latency_sec = 0.0
        self.e2e_max_latency_sec = 0.0
        self.pptg_throughput_tps = 0.0
        self.concurrency = 0.0
        self.vm = expt_details['vm']
        threads_list = threads_validation.parse_threads_range(self.vm.cpu_arch,
            FLAGS[f"{self.benchmark}_threads_range"].value)
        self.num_available_threads = len(threads_list)
        self.available_threads = " ".join(map(str, threads_list))
        self.tps_per_user = expt_details['tps_per_user']

    def llama_batched_bench_base_run(self, num_processes: int, num_threads: int,
                                     batch_size: int,model: str, prompt_size: str, vm):
        """
        runs llama-batched-bench with specified thread, batch-size for model and prompt_size
        """
        llama_exe_path = FLAGS[f"{self.benchmark}_llama_exe_path"].value
        volumes = FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value
        volume_mountpoints = FLAGS[
                f"{docker_package.PACKAGE_NAME}_volume_mountpoints"
                ].value
        threads_range = FLAGS[f"{self.benchmark}_threads_range"].value
        stability = FLAGS[f"{self.benchmark}_stability"].value
        output_dir = volumes[2]
        docker_out_dir = volume_mountpoints[2]
        memory_place = FLAGS[f"{self.benchmark}_memory_placement"].value
        if FLAGS[f"{docker_package.PACKAGE_NAME}_build_docker_dir"].value:
            ld_library_path = ""
        else:
            ld_library_path = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{llama_exe_path} &&"
        args = {}
        current_case = f"{num_processes} x {num_threads} "
        current_case += f"[proc x threads per proc], bs = {batch_size}"
        logging.info("\nRunning %s", current_case)
        args = {
                "model": model,
                "prompt_size": int(prompt_size),
                "tokens": self.output_tokens_size,
                "batch_size": int(batch_size),
                "num_processes": int(num_processes),
                "num_threads": int(num_threads),
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
                    "cd / && "
                    f" {ld_library_path}"
                    f" python3 utils/benchmark.py --exe_path {llama_exe_path}/llama-batched-bench "
                    f" --output_dir {docker_logs_dir} -m models/{model}"
                    f" -n {str(num_processes)} "
                    f"-t {str(num_threads)} -b {str(batch_size)} -p {str(prompt_size)}"
                    f" -k {str(self.output_tokens_size)} -r {self.available_threads}"
                    f" --mp {str(memory_place)}"
                    )
            if FLAGS[f"{self.benchmark}_stability"].value:
                cmd += " --stability"
            if FLAGS[f"{self.benchmark}_flash_attention"].value:
                cmd += " -fa 1"
            if FLAGS[f"{docker_package.PACKAGE_NAME}_gpus"].value:
                cmd += " -gpus 1"
            FLAGS[f"{docker_package.PACKAGE_NAME}_shell_type"].value = (
                    "bash"
                    )
            FLAGS[f"{docker_package.PACKAGE_NAME}_exec_command"].value = cmd
            start = time.time()
            docker_package.exec_docker(vm)
            finish = time.time()
            self.tps_per_user = results.summarize(
                    vm, logs_dir, start, finish, vm_util.GetTempDir()
                    )
            results.save_csv(vm_util.GetTempDir())
        return results

    def check_run_config_validity(self, n_threads, n_procs):
        """checks if using threads is valid"""
        cores_used = n_threads*n_procs
        return cores_used <= self.num_available_threads
