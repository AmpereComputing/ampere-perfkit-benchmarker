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
Module containing utility functions to run llama
"""

import os
import time
import argparse
import subprocess

INSTABILITY_THRESHOLD = 1.01

online_threads = None


def parse_args():
    """
    Parse Args
    """
    parser = argparse.ArgumentParser(description="Run offline benchmark.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory path to save logs and csv",
    )
    parser.add_argument(
        "--exe_path",
        type=str,
        required=True,
        help="path to executable, e.g. /llama.aio/bin/llama-batched-bench ",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="name of the model"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=True,
        help="batch size to feed the model with",
    )
    parser.add_argument(
        "-p",
        "--prompt_size",
        type=int,
        required=True,
        help="prompt size to feed the model with",
    )
    parser.add_argument(
        "-k",
        "--tokens",
        type=int,
        required=True,
        help="number of tokens to generate with the model",
    )
    parser.add_argument(
        "-r",
        "--threads_range",
        nargs='+',
        type=str,
        required=True,
        help="list of threads separated by space, being divided in processes "
        "- hint: 'lscpu | grep NUMA'",
    )
    parser.add_argument("--kv_cache", type=int, default=65536, help="kv cache size")
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
        "--stability", action="store_true", help="run till the result is stable"
    )
    parser.add_argument("--mp",
                        type=str, default="local",
                        help="memory placement policy, 'local','interleave' or 'none'")
    parser.add_argument("-fa",
                        type=int, default=0, choices=range(0, 2),
                        help="enable flash attention")
    parser.add_argument("-gpus",
                        type=int, default=0,choices=range(0, 2), help="gpus is in use")
    return parser.parse_args()

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
        str(t)
        for t in online_threads[
            num_threads * process_id : num_threads * (process_id + 1)
        ]
    ]
    assert len(threads_to_use) == num_threads
    return ",".join(threads_to_use)


def main():
    """
    Main function
    """
    global online_threads

    args = parse_args()

    llama_bench_exe_path = args.exe_path

    online_threads = args.threads_range

    if len(online_threads) < args.num_processes * args.num_threads:
        raise ValueError(
            f"Requested config requires {args.num_processes * args.num_threads} threads, "
            f"while only {len(online_threads)} threads are both online and designated"
        )

    logs_dir = args.output_dir
    os.mkdir(logs_dir)
    current_subprocesses = []

    if args.mp == "local":
        mem_place = "--localalloc"
    elif args.mp == "interleave":
        mem_place = "--interleave=all"
    else:
        mem_place = "none"

    print(f"Mem placement policy is = {mem_place}")

    for n in range(args.num_processes):
        logfile = f"{logs_dir}/log_{n}"
        if mem_place == "none":
            cmd = [
                "numactl",
                f"--physcpubind={gen_threads_config(args.num_threads, n)}",
                llama_bench_exe_path,
                "-m",
                args.model,
                "-c",
                str(args.kv_cache),
                "-b",
                "2048",
                "-ub",
                "512",
                "-npp",
                str(args.prompt_size),
                "-ntg",
                str(args.tokens),
                "-npl",
                str(args.batch_size),
                "-t",
                str(args.num_threads),
                "-tb",
                str(args.num_threads),
                "--no-mmap"
            ]
        else:
            cmd = [
                "numactl",
                f"--physcpubind={gen_threads_config(args.num_threads, n)}",
                str(mem_place),
                llama_bench_exe_path,
                "-m",
                args.model,
                "-c",
                str(args.kv_cache),
                "-b",
                "2048",
                "-ub",
                "512",
                "-npp",
                str(args.prompt_size),
                "-ntg",
                str(args.tokens),
                "-npl",
                str(args.batch_size),
                "-t",
                str(args.num_threads),
                "-tb",
                str(args.num_threads),
                "--no-mmap"
            ]
        if args.gpus == 1:
            cmd.append("-ngl")
            cmd.append(str(999))
        if args.fa != 0:
            cmd.append("-fa")
            cmd.append("on")
        else:
            cmd.append("-fa")
            cmd.append("off")

        p = subprocess.Popen(
                cmd, stdout=open(logfile, "wb"), stderr=open(logfile, "wb")
            )
        current_subprocesses.append(
            (n, p, logfile)
        )

    completed = False
    while not completed:
        time.sleep(1)
        completed_count = 0
        for (n, p, logfile) in current_subprocesses:
            status = p.poll()
            if status is not None:
                if status == 0:
                    completed_count += 1
                    if completed_count == len(current_subprocesses):
                        completed = True
                        break
                else:
                    if status < 0:
                        raise ValueError(
                            f"FAIL: Llama process {n} exited with code {-status}. "
                            f"Check Log: {logfile}"
                        )
                    else:
                        raise ValueError(
                            f"FAIL: Llama process {n} exited with code {status}. "
                            f"Check Log: {logfile}"
                        )


if __name__ == "__main__":
    main()
