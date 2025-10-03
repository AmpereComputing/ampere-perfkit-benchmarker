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

from perfkitbenchmarker.linux_virtual_machine import Fedora37Mixin
from perfkitbenchmarker.static_virtual_machine import StaticVirtualMachine

from ampere.pkb import os_types


# RHEL package managers
YUM = 'yum'
DNF = 'dnf'


class Fedora41Mixin(Fedora37Mixin):
    """Re-use Fedora37Mixin"""
    OS_TYPE = os_types.FEDORA41

    def InstallPackageGroup(self, package_group):
        """Installs a 'package group' with correct syntax for Fedora versions >=41
        See:
            https://discussion.fedoraproject.org/t/unable-to-find-development-tools-in-fedora-41/135154
        """
        # Catch case where `perfkitbenchmarker/linux_packages/build_tools.py`
        # attempts to install 'Development Tools' and override with correct
        # syntax and group name
        if package_group == 'Development Tools':
            package_group = 'development-tools'
        cmd = f'sudo {self.PACKAGE_MANAGER} install -y @{package_group}'
        if self.PACKAGE_MANAGER == DNF:
            cmd += ' --allowerasing'
        self.RemoteCommand(cmd)


class Fedora41BasedStaticVirtualMachine(StaticVirtualMachine,
                                        Fedora41Mixin):
    pass
