# Modifications Copyright (c) 2025 Ampere Computing LLC
# Copyright 2014 PerfKitBenchmarker Authors. All rights reserved.
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

"""Runs Ampere SpecINT
"""

import logging
import re
from typing import List

from absl import flags

from perfkitbenchmarker import configs, sample
from perfkitbenchmarker.benchmark_spec import BenchmarkSpec

from ampere.pkb.linux_packages import specint2017

FLAGS = flags.FLAGS

BENCHMARK_NAME = 'specint2017_benchmark'
BENCHMARK_CONFIG = """
specint2017_benchmark:
  description: Runs SpecINT
  vm_groups:
    default:
      vm_spec: *default_single_core
      disk_spec: *default_50_gb
"""

flags.DEFINE_bool(f'{BENCHMARK_NAME}_sudo', False,
                  'Optionally run with sudo. Specint reports can use root privilege for a more comprehensive report.')
flags.DEFINE_string(f'{BENCHMARK_NAME}_numactl', None, 'Optional numactl prefix to command.')
flags.DEFINE_integer(f'{BENCHMARK_NAME}_iterations', 1, 'Number of iterations. Default is 1.')
flags.DEFINE_integer(f'{BENCHMARK_NAME}_copies', None, 'Number of Copies. Default is CPU count.')
flags.DEFINE_string(f'{BENCHMARK_NAME}_suite', 'intrate', help=f'Spec suite')
flags.DEFINE_string(f'{BENCHMARK_NAME}_config', None,
                    f'Spec2017 build/configuration. Can be a url, gs uri or config name. '
                    f'It must be a tar.gz or .tgz file.')
flags.DEFINE_string(f'{BENCHMARK_NAME}_config_sha256', None,
                    'SHA256 of config tar.gz file. If provided this will be checked')
flags.DEFINE_bool(f'{BENCHMARK_NAME}_tune', False, 'If enabled, run a tuning script before run.')
flags.DEFINE_string(f'{BENCHMARK_NAME}_tuning_script', None, 'Tuning script for before run.')
flags.DEFINE_string(f'{BENCHMARK_NAME}_optimization_command', '', 'Optional optimization flag/command')


def GetConfig(user_config):
    return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def Prepare(benchmark_spec: BenchmarkSpec):
    """Installs SpecINT on the target vm.

    Args:
        benchmark_spec: The benchmark specification. Contains all data that is required to run the benchmark.
    """
    vm = benchmark_spec.vms[0]

    vm.Install(specint2017.PACKAGE_NAME)

    tune_flag = FLAGS[f'{BENCHMARK_NAME}_tune'].value
    tune_script = FLAGS[f'{BENCHMARK_NAME}_tuning_script'].value
    spec_dir = specint2017.spec_dir()

    if tune_flag:
        logging.debug(f'Tuning with {tune_script}')
        stdout, stderr = vm.RemoteCommand(f'cd {spec_dir} && sudo ./{tune_script}', ignore_failure=True)
        logging.debug(f'Tuning stdout: {stdout}')
        logging.debug(f'Tuning stderr: {stderr}')


def Run(benchmark_spec: BenchmarkSpec):
    """Runs SpecINT on the target vm.

    Args:
        benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.

    Returns:
        A list of sample.Sample objects.
    """

    vm = benchmark_spec.vms[0]
    numactl_flag = FLAGS[f'{BENCHMARK_NAME}_numactl'].value
    suite_flag = FLAGS[f'{BENCHMARK_NAME}_suite'].value
    iterations_flag = FLAGS[f'{BENCHMARK_NAME}_iterations'].value
    copies_flag = FLAGS[f'{BENCHMARK_NAME}_copies'].value
    tune_flag = FLAGS[f'{BENCHMARK_NAME}_tune'].value
    sudo_flag = FLAGS[f'{BENCHMARK_NAME}_sudo'].value

    timeout_minutes = FLAGS.timeout_minutes
    timeout = timeout_minutes * 60 if timeout_minutes else None

    config = specint2017.config_name()
    lscpu = vm.CheckLsCpu()
    copies = copies_flag if copies_flag else int(lscpu.data['CPU(s)'])
    spec_dir = specint2017.spec_dir()

    optimization_cmd = ''

    # Remove old runs
    vm.RemoteCommand(f'cd {spec_dir} && rm -rf spec2017/result/*', ignore_failure=True)
    vm.RemoteCommand(f'cd {spec_dir} && rm -rf spec2017/benchspec/CPU/*/run/*', ignore_failure=True)

    # Run Spec2017
    if FLAGS[f'{BENCHMARK_NAME}_optimization_command'].value:
        optimization_cmd = FLAGS[f'{BENCHMARK_NAME}_optimization_command'].value

    numa_cmd = f'{numactl_flag} ' if numactl_flag else ''
    sudo_cmd = f'sudo ' if sudo_flag else ''
    spec_cmd = f'{optimization_cmd} {sudo_cmd}{numa_cmd}./run_spec2017.sh --iterations {iterations_flag} --copies {copies} ' \
               f'--nobuild --action run --noreportable --tune=base --size=ref {suite_flag}'
    cmd = f'cd {spec_dir} && {spec_cmd}'

    metadata = {
        'numactl': numactl_flag,
        'config': config,
        'iterations': iterations_flag,
        'copies': copies,
        'suite': suite_flag,
        'tuned': tune_flag,
        'specint_cmd': spec_cmd
    }

    logging.debug(f'{BENCHMARK_NAME} cmd: {cmd}')
    stdout, stderr = vm.RobustRemoteCommand(cmd, timeout=timeout)
    logging.debug(f'{BENCHMARK_NAME} stdout: {stdout}')
    logging.debug(f'{BENCHMARK_NAME} stderr: {stderr}')

    # Collect output
    logging.debug(f'Collecting intrate.refrate.rsf')
    stdout, stderr = vm.RemoteCommand(f'cd {spec_dir} && cat spec2017/result/CPU2017.001.intrate.refrate.rsf')
    logging.debug(f'Collecting intrate.refrate.rsf stdout: {stdout}')
    logging.debug(f'Collecting intrate.refrate.rsf stderr: {stdout}')
    stdout = stdout if stdout else stderr

    samples = _ParseResult(metadata, stdout)

    # Cleanup results
    vm.RemoteCommand(f'cd {spec_dir} && rm -rf spec2017/result/*', ignore_failure=True)
    vm.RemoteCommand(f'cd {spec_dir} && rm -rf spec2017/benchspec/CPU/*/run/*', ignore_failure=True)
    return samples


def Cleanup(benchmark_spec: BenchmarkSpec):
    vm = benchmark_spec.vms[0]
    vm.Uninstall(specint2017.PACKAGE_NAME)


def _ParseResult(metadata: dict, output: str) -> List[sample.Sample]:
    """Returns SpecINT data as a sample.

    Sample output eg:

    Args:
        metadata: metadata of the sample.
        output: the output of the specint benchmark.
    """
    samples = []
    metadata = metadata if metadata else {}
    output_list = output.splitlines()

    for line in output_list:
        # GeoMean
        match = re.search(r'^spec\.cpu2017\.basemean:\s(\d*\.\d*)$', line)
        if match:
            samples.append(sample.Sample(
                metric='rate',
                unit='rate',
                value=float(match.group(1)),
                metadata=metadata.copy()
            ))

        # Microbenchmark Ratios
        match = re.search(r'^spec\.cpu2017\.results\.(.*)\.base\.\d{3}\.ratio:\s(\d*\.\d*)$', line)
        if match:
            samples.append(sample.Sample(
                metric=f'{match.group(1)}_ratio',
                unit='rate',
                value=float(match.group(2)),
                metadata=metadata.copy()
            ))

    if not samples:
        raise ValueError(f'Sample not found: \n{output}')

    samples.append(sample.Sample(
        metric='spec2017_raw',
        value=0,
        unit='',
        metadata={
            'spec2017_raw': output,
            **metadata
        }))
    return samples
