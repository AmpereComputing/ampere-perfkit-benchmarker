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
finds jdk version and adds to sample
"""

import logging
from typing import Any, List
from absl import flags
from perfkitbenchmarker import events
from perfkitbenchmarker.linux_virtual_machine import BaseLinuxMixin
from perfkitbenchmarker.sample import Sample
from perfkitbenchmarker.benchmark_spec import BenchmarkSpec


flags.DEFINE_bool(
    "ampere_find_jdk_version",
    True,
    help="Enable/disable printing jdk_version in report Supported Values: True/False ",
)

FLAGS = flags.FLAGS


def register_all(_: Any, parsed_flags: flags.FLAGS):

    events.benchmark_samples_created.connect(benchmark_samples_created, weak=False)


def benchmark_samples_created(
        _, benchmark_spec: BenchmarkSpec, samples: List[Sample]
        ):
    """
    registers find jdk event
    :param _:
    :param List[Sample]:
    :return:
    """
    if not FLAGS.ampere_find_jdk_version:
        return
    if not benchmark_spec.vms:
        return
    try:
        samples.extend(collect_sample_jdk_version(benchmark_spec))
    except Exception as e:
        logging.exception(
                f"Unknown exception during ampere_find_jdk_version! Details: {str(e)}"
                )


def collect_sample_jdk_version(benchmark_spec: BenchmarkSpec) -> List[Sample]:
    """
    collects sample for jdk_version
    :param benchmark_spec:
    :return: sample list
    """
    jdk_version_samples: List[Sample] = []
    for _, vm in enumerate(benchmark_spec.vms):
        if not isinstance(vm, BaseLinuxMixin):
            continue

        version, _ = vm.RemoteCommand("java -version 2>&1 | head -n 1")
        version = version.strip()
        if "java: command not found" in version:
            version = "Java not found"
        metadata = {
            "hostname": vm.hostname,
            "jdk_version": version,
        }
        jdk_version_samples.append(
            Sample("jdk_version", value=0, unit="", metadata=metadata)
        )
    return jdk_version_samples
