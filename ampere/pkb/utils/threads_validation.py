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

"""Module for architecture-wise threads validation"""

import logging
from absl import flags

FLAGS = flags.FLAGS

def parse_threads_range(arch, threads_range: str) -> list[int]:
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
    if arch == "x86_64":
        len_designated_threads = 0
        i_range = -1
        for s in threads_range:
            i_range +=1
            s_0, s_1 = int(s[0]), int(s[1])
            if s_1 < s_0:
                raise ValueError(
                        f"Range {s_0}-{s_1} is not valid, second value has to be "
                        f"equal to or greater than the first value"
                        )
            if i_range % 2 == 1:
                i_pos = ((i_range - 1) * len_designated_threads) - 1
                for i_thread in list(range(s_0, s_1 + 1)):
                    i_pos = i_pos + 2
                    designated_threads.insert(i_pos, i_thread)
            else:
                designated_threads += list(range(s_0, s_1 + 1))
                len_designated_threads = len(list(range(s_0, s_1 + 1)))
    else:
        for s in threads_range:
            s_0, s_1 = int(s[0]), int(s[1])
            if s_1 < s_0:
                raise ValueError(
                        f"Range {s_0}-{s_1} is not valid, second value has to be "
                        f"equal to or greater than the first value"
                        )
            designated_threads += list(range(s_0, s_1 + 1))
    logging.info("designated_threads: %s", designated_threads)
    return designated_threads


def check_threads_validity(server, BENCHMARK_NAME):
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
    arch = server.cpu_arch
    available_threads_list = parse_threads_range(arch,threads_range)
    if len(available_threads_list) < max(threads_per_proc_list):
        raise ValueError(
            f"Requested number of threads ({max(threads_per_proc_list)})"
            f"exceeds threads available ({len(available_threads_list)})"
        )
