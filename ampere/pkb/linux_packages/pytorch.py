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
Module containing functions to pull, run and exec into pytorch docker

"""

import os
import posixpath
from absl import flags
from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import docker as docker_package

PACKAGE_NAME = "ampere_pytorch"
BENCHMARK_NAME = "ampere_dlrm_benchmark"

INSTALL_DIR = download_utils.INSTALL_DIR

FLAGS = flags.FLAGS

DATA_FILES = flags.DEFINE_string(
    f"{PACKAGE_NAME}_data",
    "",
    "Must be in ./ampere/pkb_internal/data/",
)


def get_pytorch_metadata():
    """
    Retrieve the Docker image version string for the PyTorch package.

    This function checks whether a custom build directory is specified via
    FLAGS for the PyTorch Docker package. Based on that, it constructs and
    returns the appropriate Docker image version string using either the build
    or default image and version flags.

    Returns:
        str: A string representing the Docker image name and version,
             formatted as "<image>-<version>".
    """
    docker_version = (
            FLAGS[f"{docker_package.PACKAGE_NAME}_image"].value
            + "-"
            + FLAGS[f"{docker_package.PACKAGE_NAME}_image_version"].value
        )
    return docker_version

def Install(vm):
    """
    Installs and sets up the PyTorch Docker environment on the given VM.

    This function handles the installation of a PyTorch-based benchmark environment
    using either a custom-built Docker image or a pulled image from a repository.
    It also sets up required benchmark and utility scripts, mounts them inside
    the container, and executes the initial setup script within the container.

    Steps performed:
    1. Installs Docker on the VM.
    2. Builds a Docker image from a provided directory or pulls a prebuilt image.
    3. Copies benchmark scripts to the VM.
    4. Prepares volume mappings between the host and Docker container.
    5. Sets default values for container name, daemon mode, privileges, and entrypoint if unset.
    6. Starts the Docker container and runs the setup script inside the container.

    Args:
        vm (BaseVirtualMachine): The target virtual machine to install and run the container on.

    Raises:
        ValueError: If Docker build or image pull fails due to misconfiguration.
    """
    docker_package.Install(vm)
    # Pull docker image
    docker_pull = docker_package.pull_docker(vm)
    if not docker_pull:
        raise ValueError(
                "Docker cannot be pulled, please check docker image repository and image version"
                )
    set_flags(vm)
    volume_mountpoints = FLAGS[f"{docker_package.PACKAGE_NAME}_volume_mountpoints"].value

    docker_package.run_docker(vm)
    cmd_exec_installs = "cd /workspace && "
    cmd_exec_installs += "git clone --recursive https://github.com/AmpereComputingAI/ampere_model_library.git && "
    cmd_exec_installs += f"cp {volume_mountpoints[2]} ampere_model_library/setup_ampere_aml.sh && "
    cmd_exec_installs += f"cp {volume_mountpoints[3]} ampere_model_library/utils/setup/install_pytorch.py && "
    cmd_exec_installs += "bash ampere_model_library/setup_ampere_aml.sh && "
    cmd_exec_installs += "source ampere_model_library/set_env_variables.sh && "
    cmd_exec_installs += "PYTHONPATH=$(pwd) && sleep 1 && "
    cmd_exec_installs += "pip3 install --break-system-packages python-Levenshtein O365 || pip3 install python-Levenshtein O365 flake8 urlextract "
    # exec into docker to setup AML
    FLAGS[f"{docker_package.PACKAGE_NAME}_shell_type"].value = "bash"
    FLAGS[f"{docker_package.PACKAGE_NAME}_exec_command"].value = cmd_exec_installs
    docker_package.exec_docker(vm)


def set_flags(vm):
    """checking and setting default values for pytorch docker flags"""
    utils_benchmark_file = "ampere/pkb/utils/run_dlrm.py"
    utils_setup_file = "ampere/pkb/utils/setup_ampere_aml.sh"
    utils_setup_pytorch = "ampere/pkb/utils/install_pytorch.py"
    vm.RemoteCopy(utils_benchmark_file, download_utils.INSTALL_DIR)
    vm.RemoteCopy(utils_setup_file, download_utils.INSTALL_DIR)
    vm.RemoteCopy(utils_setup_pytorch, download_utils.INSTALL_DIR)
    output_dir = posixpath.join(download_utils.INSTALL_DIR, "out_dir")
    vm.RemoteCommand(f"mkdir -p {output_dir}")
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_volume_names"].value = [
            f"{download_utils.INSTALL_DIR}/run_dlrm.py",
            output_dir,
            f"{download_utils.INSTALL_DIR}/setup_ampere_aml.sh",
            f"{download_utils.INSTALL_DIR}/install_pytorch.py"
        ]
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_volume_mountpoints"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_volume_mountpoints"].value = [
            "/workspace/benchmark_pytorch_dlrm.py",
            "/out_dir/",
            "/workspace/setup_ampere_aml.sh",
            "/workspace/install_pytorch.py"
        ]
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_name"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_name"].value = "pytorch_container"
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_daemon"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_daemon"].value = True
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_privileged_docker"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_privileged_docker"].value = False
    if not FLAGS[f"{docker_package.PACKAGE_NAME}_bash_command"].value:
        FLAGS[f"{docker_package.PACKAGE_NAME}_bash_command"].value = (
            "--entrypoint /bin/sh -it"
        )


def Uninstall(vm):
    """
    Uninstalls the PyTorch Docker environment from the given VM.

    This function delegates to the generic Docker uninstallation routine
    to clean up Docker-related resources (containers, images, etc.) on the VM.

    Args:
        vm (BaseVirtualMachine): The virtual machine from which to remove the Docker environment.
    """

    docker_package.Uninstall(vm)
