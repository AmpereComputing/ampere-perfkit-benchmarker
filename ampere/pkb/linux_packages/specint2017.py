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

"""Module to install,uninstall Specint 2017
"""

import posixpath

from absl import flags
from perfkitbenchmarker.linux_virtual_machine import BaseLinuxVirtualMachine

from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import libxcrypt_compat

FLAGS = flags.FLAGS
PACKAGE_NAME = 'specint2017'
DOWNLOAD_TIMEOUT = 600


def flags_config() -> str:
    return FLAGS.specint2017_benchmark_config


def Install(vm: BaseLinuxVirtualMachine):
    """
    Installs SpecINT 2017
    """
    vm.Install('numactl')
    vm.Install(libxcrypt_compat.PACKAGE_NAME)
    vm.Install('tuned')
    sha256 = FLAGS.specint2017_benchmark_config_sha256

    config_flag = flags_config()
    tgz_file = cache_filename()
    if not is_cached(vm):
        download_utils.mk_cache_dir(vm)

        if config_flag.startswith('https://') or config_flag.startswith('http://'):
            dst_file = download_utils.download(config_flag, [vm],
                                               dst=download_utils.CACHE_DIR,
                                               sha256=sha256,
                                               timeout=DOWNLOAD_TIMEOUT,
                                               force=True)
            vm.RemoteCommand(f'mv {dst_file} {tgz_file}', ignore_failure=True)
        elif config_flag.startswith('gs://'):
            dst_file = download_utils.gsutil(config_flag, [vm],
                                             dst=download_utils.CACHE_DIR,
                                             sha256=sha256,
                                             timeout=DOWNLOAD_TIMEOUT,
                                             force=True)
            vm.RemoteCommand(f'mv {dst_file} {tgz_file}', ignore_failure=True)
        else:
            raise ValueError(
                f"Unknown src format for specint2017! Only HTTP(s) and gs:// are supported!"
            )

    config_dir = remote_config_dir()
    vm.RemoteCommand(f'ls {tgz_file}')  # Verify file exists
    vm.RemoteCommand(f'mkdir -p {config_dir}')
    vm.RemoteCommand(f'tar -xf {tgz_file} -C {config_dir}')


def Uninstall(vm: BaseLinuxVirtualMachine):
    """
    Removes SpecINT 2017
    """
    vm.RemoteCommand(f'rm -rf {remote_config_dir()}', ignore_failure=True)


def cache_filename() -> str:
    return posixpath.join(download_utils.CACHE_DIR, f'{config_name()}.tgz')


def config_name():
    config_flag = flags_config()

    if config_flag.startswith('https://') or config_flag.startswith('http://'):
        return posixpath.basename(config_flag).replace('.tgz', '').replace('.tar.gz', '')
    if config_flag.startswith('gs://'):
        return posixpath.basename(config_flag).replace('.tgz', '').replace('.tar.gz', '')
    return config_flag.replace('.tgz', '').replace('.tar.gz', '')


def remote_config_dir():
    return posixpath.join(posixpath.join(download_utils.INSTALL_DIR, config_name()))


def spec_dir():
    return posixpath.join(download_utils.INSTALL_DIR,
                          config_name(),
                          'ampere_spec2017')


def is_cached(vm: BaseLinuxVirtualMachine) -> bool:
    sha256 = FLAGS.specint2017_benchmark_config_sha256
    tgz_file = cache_filename()
    stdout, _ = vm.RemoteCommand(f'sha256sum {tgz_file}', ignore_failure=True)
    if stdout and not sha256:
        return True
    if stdout and sha256 and stdout.split()[0] == sha256:
        return True
    return False
