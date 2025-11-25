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
Module containing utility functions to run DLRM
"""

import os
import time
import argparse
import subprocess


online_threads = None


def parse_args():
    """
    Parse Args
    """
    parser = argparse.ArgumentParser(description="Run offline benchmark.")
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        required=True,
        help="precision of the model provided",
    )

    parser.add_argument(
        "--aml_dir",
        type=str,
        required=True,
        help="dir to ampere model library in docker",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory path to save logs and csv",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=True,
        help="batch size to feed the model with",
    )
    parser.add_argument(
        "-r",
        "--threads_range",
        type=str,
        required=True,
        help="range of threads to use, e.g. '0-63,128-191', "
        "threads will be divided between processes "
        "- hint: 'lscpu | grep NUMA'",
    )
    parser.add_argument(
        "-n",
        "--num_processes",
        type=int,
        default=1,
        help="number of processes to spawn",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=1,
        help="number of threads to use per process",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=900,
        help="duration to apply per single benchmark case",
    )
    parser.add_argument(
        "--sleep_duration",
        type=int,
        default=5,
        help="sleep in polling for checking completion of all processes",
    )

    return parser.parse_args()


def parse_threads_range(threads_range: str) -> list[int]:
    """
    Parses a string representing a range of threads and returns a list of individual thread indices.

    The input string is expected to be a comma-separated list of ranges in the format 'start-end',
    where 'start' and 'end' are integers. Each range specifies a contiguous block of thread indices,
    check_process_statusand the function will expand these ranges into a list of individual thread indices.

    Args:
        threads_range (str): A string representing one or more ranges of thread indices,
                              e.g., '0-3,5-6' results in [0, 1, 2, 3, 5, 6].

    Returns:
        list[int]: A list of individual thread indices parsed from the input string.

    Raises:
        ValueError: If the input string is not in the correct format or if any range is invalid
                    (e.g., when the 'end' index is smaller than the 'start' index).

    """
    threads_range = [s.split("-") for s in threads_range.split(",")]
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
                f"Range {s_0}-{s_1} is not valid, second value has to be "
                f"equal to or greater than the first value"
            )
        designated_threads += list(range(s_0, s_1 + 1))
    return designated_threads


def gen_threads_config(num_threads, process_id):
    """
    Generates a comma-separated string of thread indices for a given process.

    This function selects a specific range of threads based on the total number
    of threads and the process ID, and returns the thread indices as a comma-separated string.
    The range of threads is determined by slicing the `online_threads` list.

    Args:
        num_threads (int): The number of threads to assign to a process.
        process_id (int): The process ID that determines which portion of the
                          `online_threads` list to use.

    Returns:
        str: A comma-separated string of thread indices for the given process.
    """

    threads_to_use = [
        str(thread)
        for thread in online_threads[
            num_threads * process_id : num_threads * (process_id + 1)
        ]
    ]
    assert len(threads_to_use) == num_threads
    return " ".join(threads_to_use), ",".join(threads_to_use)


def init_env_variables(num_threads):
    """
    initialize required environment variables
    """
    os.environ["IGNORE_DATASET_LIMITS"] = "1"

    os.environ["AIO_NUM_THREADS"] = str(num_threads)
    os.environ["DLS_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    return True


def main():
    """
    Main function
    """
    global online_threads

    args = parse_args()
    os.environ["AIO_SKIP_MASTER_THREAD"] = "1"
    designated_threads = parse_threads_range(args.threads_range)
    numa_config = subprocess.run(
        ["numactl", "--show"], capture_output=True, text=True, check=True
    )
    online_threads = [
        int(thread)
        for thread in numa_config.stdout.split("physcpubind: ")[1]
        .split(" \ncpubind:")[0]
        .split()
        if int(thread) in designated_threads
    ]
    if len(online_threads) < args.num_processes * args.num_threads:
        raise ValueError(
            f"Requested config requires {args.num_processes * args.num_threads} threads, "
            f"while only {len(online_threads)} threads are both online and designated"
        )

    path_to_runner = os.path.join(args.aml_dir, "recommendation/dlrm_torchbench/run.py")

    logs_dir = args.output_dir
    os.mkdir(logs_dir)
    results_dir = os.path.join(logs_dir, ".cache_aml")
    os.environ["RESULTS_DIR"] = results_dir
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        for filepath in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, filepath))
    else:
        os.mkdir(results_dir)
    current_subprocesses = []
    os.environ["IGNORE_DATASET_LIMITS"] = "1"
    os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] = ""
    init_env_variables(args.num_threads)
    precision = "fp32"

    if args.precision == "fp16":
        os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] = ".*"
        os.environ["ALLOW_IMPLICIT_FP16"] = "1"

    for n in range(args.num_processes):
        num_threads_per_cpus, physcpubind = gen_threads_config(args.num_threads, n)
        logfile = f"{logs_dir}/log_{n}"
        os.environ["AIO_NUMA_CPUS"] = num_threads_per_cpus
        os.environ["DLS_NUMA_CPUS"] = num_threads_per_cpus
        cmd = [
            "numactl",
            f"--physcpubind={physcpubind}",
            "python3",
            f"{path_to_runner}",
            "-p",
            f"{precision}",
            "-f",
            "pytorch",
            "-b",
            f"{args.batch_size}",
            "--timeout",
            f"{args.duration}",
        ]

        current_subprocesses.append(
            subprocess.Popen(
                cmd, stdout=open(logfile, "wb"), stderr=open(logfile, "wb")
            )
        )

    completed = False
    while not completed:
        time.sleep(args.sleep_duration)
        completed_count = 0
        for process in current_subprocesses:
            status = process.poll()
            if status is not None:
                if status == 0:
                    completed_count += 1
                    if completed_count == len(current_subprocesses):
                        completed = True
                        break
                else:
                    raise ValueError(
                        "FAIL: At least one process returned exit code other than 0 or died!"
                    )


if __name__ == "__main__":
    main()
