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
import dataclasses
import logging
from typing import Any, Dict, List
import six
from absl import flags
from perfkitbenchmarker import sample
from ampere.pkb.common import download_utils

PACKAGE_NAME = "ampere_dlrm"
BENCHMARK_NAME = "ampere_pytorch_dlrm"


INSTALL_DIR = download_utils.INSTALL_DIR
OUT_DIR = 1

FLAGS = flags.FLAGS


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



@dataclasses.dataclass
class DLRMResult:
    """Class that represents DLRM results."""

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
    def parse_dlrm_results(cls, dlrm_results: str) -> "DLRMResult":
        """Parse dlrm result textfile and return results.
        Args:
          dlrm_results: Str output of running dlrm.
        Returns:
        """
        dlrm_csv_result = _parse_csv(dlrm_results)
        return cls(
            n_proc=dlrm_csv_result.n_proc,
            n_threads=dlrm_csv_result.n_threads,
            batch_size=dlrm_csv_result.batch_size,
            throughput_tps=dlrm_csv_result.throughput_tps,
            p90_latency=dlrm_csv_result.p90_latency,
            p99_latency=dlrm_csv_result.p99_latency,
            p999_latency=dlrm_csv_result.p999_latency,
            start=dlrm_csv_result.start,
            finish=dlrm_csv_result.finish,
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


def _parse_csv(dlrm_results: str) -> DLRMResult:
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
    csv_fp = six.StringIO(str(dlrm_results))
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
        raise ValueError(f"Test Failed: {dlrm_results}")
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
    return DLRMResult(
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
