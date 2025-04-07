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

"""Module to install,uninstall open source Multichase

https://github.com/google/multichase
"""
import posixpath
from absl import flags

from ampere.pkb.common import download_utils
from perfkitbenchmarker.linux_virtual_machine import BaseLinuxVirtualMachine


PACKAGE_NAME = "ampere_multichase"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    f"{PACKAGE_NAME}_zip",
    "https://github.com/google/multichase/archive/refs/heads/master.zip",
    help="Multichase archive from open-source in .zip format.",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_src_dir",
    "multichase-master",
    help="Location of Multichase source within package",
)


def Install(vm):
    """
    Installs multichase to vm
    """
    vm.Install("numactl")
    vm.Install("build_tools")
    # Get flag values
    multichase_zip = FLAGS[f"{PACKAGE_NAME}_zip"].value
    # Send zip to VM
    dst_file = download_utils.download(
        multichase_zip, [vm], dst=download_utils.INSTALL_DIR, force=True
    )
    # Unzip to /opt/pkb/multichase-master by default
    vm.RemoteCommand(f"unzip {dst_file} -d {download_utils.INSTALL_DIR}")
    # Build and grant permissions
    vm.RemoteCommand(f"cd {_get_package_dir()} && chmod +x gen_expand && STATIC=1 make")


def Uninstall(vm):
    """Cleans up Ampere multichase from the target vm.

    Args:
      vm: The vm on which Ampere multichase is uninstalled.
    """
    vm.RemoteCommand(f"rm -rf {_get_package_dir()}", ignore_failure=True)


def get_multiload_bin():
    """Returns /opt/pkb/<src_dir>/multiload, e.g. /opt/pkb/multichase-master/multiload"""
    return posixpath.join(_get_package_dir(), "multiload")


def _get_package_dir():
    """Returns /opt/pkb/<src_dir>, e.g. /opt/pkb/multichase-master"""
    src_dir = FLAGS[f"{PACKAGE_NAME}_src_dir"].value
    return posixpath.join(download_utils.INSTALL_DIR, src_dir)
