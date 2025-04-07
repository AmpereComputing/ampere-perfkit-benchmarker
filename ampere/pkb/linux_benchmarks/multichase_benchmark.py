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

"""Runs Multichase from open source 

https://github.com/google/multichase
"""
import logging

from absl import flags
from perfkitbenchmarker import configs
from perfkitbenchmarker.benchmark_spec import BenchmarkSpec
from perfkitbenchmarker.linux_virtual_machine import BaseLinuxVirtualMachine
from perfkitbenchmarker import sample
from typing import Any, List, Dict, Optional

from ampere.pkb.linux_packages import multichase


BENCHMARK_NAME = "ampere_multichase_benchmark"
BENCHMARK_CONFIG = """
ampere_multichase_benchmark:
  description: Runs multichase 
  vm_groups:
    default:
      vm_spec: *default_single_core
      disk_spec: *default_50_gb
"""

VALID_CHASE = (
    "chaseload",
    "chase",
    "simple",
    "work:N",
    "incr",
    "parallel12",
    "parallel13",
    "parallel14",
    "parallel15",
    "parallel16",
    "parallel17",
    "parallel18",
    "parallel19",
    "parallel110",
    "critword:N",
)

VALID_LOAD_GENERATOR = ("stream-triad", "memset-libc", "memcpy-libc", "stream-sum")

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    f"{BENCHMARK_NAME}_hugepages", False, "Enable hugepages during the benchmark."
)
flags.DEFINE_string(
    f"{BENCHMARK_NAME}_numactl", None, "Optional numactl prefix to command."
)
flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_tlb_locality", 262144, "TLB locality in bytes (default 262144)"
)
flags.DEFINE_integer(f"{BENCHMARK_NAME}_stride_size", 256, "Stide size (default 256)")
flags.DEFINE_integer(
    f"{BENCHMARK_NAME}_nr_samples",
    5,
    "nr of 0.5 second samples to use (default 5, 0 = infinite)",
)
flags.DEFINE_list(
    f"{BENCHMARK_NAME}_threads_list",
    [1],
    "List of values to use with the `LoadThds` parameter.",
)
flags.DEFINE_string(
    f"{BENCHMARK_NAME}_total_memory_size",
    "268435456",
    "total memory size (default 268435456)",
)
flags.DEFINE_enum(
    f"{BENCHMARK_NAME}_chase",
    "chaseload",
    VALID_CHASE,
    "Chase to use (default chaseload)",
)
flags.DEFINE_enum(
    f"{BENCHMARK_NAME}_load_generator",
    "stream-triad",
    VALID_LOAD_GENERATOR,
    "Load generator (default stream-triad)",
)


def GetConfig(user_config: Dict[str, Any]) -> Dict[str, Any]:
    return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def Prepare(bm_spec: BenchmarkSpec) -> None:
    """Installs multichase from open source on the target vm."""
    vm: BaseLinuxVirtualMachine = bm_spec.vms[0]
    vm.Install(multichase.PACKAGE_NAME)


def Run(bm_spec: BenchmarkSpec) -> List[sample.Sample]:
    """Runs multichase on the target vm."""
    vm: BaseLinuxVirtualMachine = bm_spec.vms[0]
    # Run multiload for each thread value in list
    threads_list_flag = FLAGS[f"{BENCHMARK_NAME}_threads_list"].value
    all_thread_samples = []
    for thread in threads_list_flag:
        samples = _run_single_threaded(vm, thread)
        [all_thread_samples.append(sample) for sample in samples]
    return all_thread_samples


def _run_single_threaded(vm, thread_value):
    """Runs Multichase with a single LoadThds value

    Args:
        vm: The VM representing the SUT
        thread_value: The value desired for LoadThds

    Returns:
        A list of sample.Sample objects
    """
    hugepages_flag = FLAGS[f"{BENCHMARK_NAME}_hugepages"].value
    numactl_flag = FLAGS[f"{BENCHMARK_NAME}_numactl"].value
    tlb_locality_flag = FLAGS[f"{BENCHMARK_NAME}_tlb_locality"].value
    stride_size_flag = FLAGS[f"{BENCHMARK_NAME}_stride_size"].value
    nr_samples_flag = FLAGS[f"{BENCHMARK_NAME}_nr_samples"].value
    total_memory_size_flag = FLAGS[f"{BENCHMARK_NAME}_total_memory_size"].value
    chase_flag = FLAGS[f"{BENCHMARK_NAME}_chase"].value
    load_generator_flag = FLAGS[f"{BENCHMARK_NAME}_load_generator"].value

    timeout_minutes = FLAGS.timeout_minutes
    timeout = timeout_minutes * 60 if timeout_minutes else None

    cmd = get_multiload_cmd(
        vm,
        hugepages=hugepages_flag,
        numactl=numactl_flag,
        tlb_locality=tlb_locality_flag,
        stride_size=stride_size_flag,
        nr_samples=nr_samples_flag,
        threads=thread_value,
        memory_size=total_memory_size_flag,
        chase=chase_flag,
        load_generator=load_generator_flag,
    )

    metadata = get_metadata(
        cmd,
        hugepages=hugepages_flag,
        numactl=numactl_flag,
        tlb_locality=tlb_locality_flag,
        stride_size=stride_size_flag,
        nr_samples=nr_samples_flag,
        threads=thread_value,
        memory_size=total_memory_size_flag,
        chase=chase_flag,
        load_generator=load_generator_flag,
    )

    logging.debug(f"Multichase: {cmd}")
    stdout, stderr = vm.RemoteCommand(cmd, timeout=timeout)
    stdout = stdout if stdout else stderr
    logging.debug(f"Multichase output: \n{stdout}")
    samples = ParseMultiloadOutput(metadata, stdout)
    if not samples:
        raise Exception(f"Sample not found \n{stdout}")
    return samples


def Cleanup(bm_spec: BenchmarkSpec) -> None:
    """Cleans up multichase from the target vm."""
    vm: BaseLinuxVirtualMachine = bm_spec.vms[0]
    vm.Uninstall(multichase.PACKAGE_NAME)


def ParseMultiloadOutput(metadata: dict, output: str) -> Optional[List[sample.Sample]]:
    """
    Example Output
    Samples , Byte/thd      , ChaseThds     , ChaseNS       , ChaseMibs     , ChDeviate     , LoadThds      , LdMaxMibs
    8       , 536870912     , 1             , 5601.406      , 1             , 0.042         , 79            , 25767
    ...
        , LdAvgMibs     , LdDeviate     , ChaseArg      , MemLdArg
        , 22525         , 0.211         , chaseload     , memset-libc
    """
    output_list = output.splitlines()
    output_matrix = [i.split() for i in output_list]
    if len(output_matrix) != 2:
        logging.error(f"output is missing!")
        return None

    labels = output_matrix[0]
    labels = [x for x in labels if x != ","]
    results = output_matrix[1]
    results = [x for x in results if x != ","]

    # For thread scaling runs: create list for tracking thread data in all sample labels
    threads = metadata["nr_threads"]
    labels_with_threads = [f"{x}_threads_{threads}" for x in labels if x != ","]
    threads_list_flag = FLAGS[f"{BENCHMARK_NAME}_threads_list"].value
    if threads_list_flag:
        labels = labels_with_threads

    logging.debug(
        f"Multichase Parsing\n" f"{'    '.join(labels)}\n" f"{'    '.join(results)}\n"
    )
    sample_params = [
        (labels[0], int(results[0]), "count"),  # samples
        (labels[1], int(results[1]), "bytes"),  # byte/thd
        (labels[2], int(results[2]), "count"),  # chasethds
        (labels[3], float(results[3]), "ns"),  # chasens
        (labels[4], float(results[4]), "Mibs"),  # chasemibs
        (labels[5], float(results[5]), "stddev"),  # chasedev
        (labels[6], int(results[6]), "count"),  # loadthds
        (labels[7], float(results[7]), "Mibs"),  # loadmax
        (labels[8], float(results[8]), "Mibs"),  # loadavg
        (labels[9], float(results[9]), "Mibs"),  # loaddev
    ]
    return [
        sample.Sample(metric=metric, value=value, unit=unit, metadata=metadata.copy())
        for metric, value, unit in sample_params
    ]


def get_metadata(
    cmd: str,
    *,
    hugepages: bool = False,
    numactl: str = None,
    tlb_locality: str = None,
    stride_size: str = None,
    nr_samples: int = None,
    threads: int = None,
    memory_size: str = None,
    chase: str = None,
    load_generator: str = None,
):
    return {
        "hugepages": hugepages,
        "multichase_src": FLAGS.ampere_multichase_zip,
        "numactl": numactl,
        "tlb_locality": tlb_locality,
        "stride_size": stride_size,
        "nr_samples": nr_samples,
        "nr_threads": threads,
        "total_memory_size": memory_size,
        "chase": chase,
        "load_generator": load_generator,
        "cmd": cmd,
    }


def get_multiload_cmd(
    vm: BaseLinuxVirtualMachine,
    *,
    hugepages: bool = False,
    numactl: str = None,
    tlb_locality: str = None,
    stride_size: str = None,
    nr_samples: int = None,
    threads: int = None,
    memory_size: str = None,
    chase: str = None,
    load_generator: str = None,
):
    multiload_bin = multichase.get_multiload_bin()
    numa_cmd = f"{numactl} " if numactl else ""
    cmd = f"{numa_cmd}{multiload_bin} "
    if hugepages:
        cmd += "-H "
    if tlb_locality:
        cmd += f"-T {tlb_locality} "
    if stride_size:
        cmd += f"-s {stride_size} "
    if nr_samples:
        cmd += f"-n {nr_samples} "
    if threads:
        cmd += f"-t {threads} "
    if memory_size:
        cmd += f"-m {memory_size} "
    if chase:
        cmd += f"-c {chase} "
    if load_generator:
        cmd += f"-l {load_generator} "

    return cmd
