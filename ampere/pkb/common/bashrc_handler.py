# Copyright (c) 2025, Ampere Computing LLC
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
handles .bashrc update,source and restore
"""

import posixpath
import logging
from perfkitbenchmarker.virtual_machine import BaseVirtualMachine


def update_bashrc(vm: BaseVirtualMachine, bashrc_path: str, update_command: str):
    """
    updates bashrc file at provided path and with update_command
    :param vm:
    :param bashrc_path:
    :param update_command:
    :return:
    """
    bashrc_backup = posixpath.join(bashrc_path, ".bashrc_backup")
    bashrc_path = posixpath.join(bashrc_path, ".bashrc")
    _, _, retcode = vm.RemoteCommandWithReturnCode(
        f"test -f {bashrc_backup}", ignore_failure=True
    )
    file_exists = retcode == 0
    if not file_exists:
        vm.RemoteCommand(f"cp {bashrc_path} {bashrc_backup}")
    command = f'grep -nr "^# If not running interactively" {bashrc_path}'
    command += f' && sed -i "/^# If not running interactively/i {update_command}" {bashrc_path}'
    command += f' || echo "unset rc\n'
    command += f'{update_command}" >> {bashrc_path}'
    vm.RemoteCommand(command)


def source_bashrc(vm: BaseVirtualMachine, bashrc_path: str):
    """
    sources with bashrc file
    :param vm:
    :param bashrc_path:
    :return:
    """
    bashrc_path = posixpath.join(bashrc_path, ".bashrc")
    vm.RemoteCommand(f"source {bashrc_path}")


def restore_bashrc(vm: BaseVirtualMachine, bashrc_path: str):
    """
    restores earlier copied backup file
    :param vm:
    :param bashrc_path:
    :return:
    """
    bashrc_backup = posixpath.join(bashrc_path, ".bashrc_backup")
    bashrc_path = posixpath.join(bashrc_path, ".bashrc")
    _, _, retcode = vm.RemoteCommandWithReturnCode(
        f"test -f {bashrc_backup}", ignore_failure=True
    )
    file_exists = retcode == 0
    if not file_exists:
        logging.debug(f"{bashrc_backup} does not exist on the VM.")
    else:
        vm.RemoteCommand(f"mv {bashrc_backup} {bashrc_path}")
