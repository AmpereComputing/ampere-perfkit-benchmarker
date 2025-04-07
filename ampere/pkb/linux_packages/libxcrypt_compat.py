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

from contextlib import suppress
from perfkitbenchmarker import errors


PACKAGE_NAME = 'ampere_libxcrypt_compat'


def YumInstall(vm):
    """
    Installs the libxcrypt-compat package on the VM.
    """
    # Allow install failure of libxcrypt-compat
    with suppress(errors.VirtualMachine.RemoteCommandError):
        vm.InstallPackages(f'libxcrypt-compat')


def AptInstall(vm):
    pass
