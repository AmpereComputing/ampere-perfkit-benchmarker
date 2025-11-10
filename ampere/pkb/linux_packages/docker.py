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


"""
Module containing docker related flags and functions

"""

import json
import logging
import posixpath
import os
from absl import flags
from perfkitbenchmarker import errors
from perfkitbenchmarker import vm_util
from perfkitbenchmarker import errors
from ampere.pkb.common import download_utils

FLAGS = flags.FLAGS

PACKAGE_NAME = "ampere_docker"
flags.DEFINE_string(
    f"{PACKAGE_NAME}_image",
    None,
    "Custom docker image name"
    "e.g. hello-world"
    "In case of elastic/rally:2.7.0"
    "rally is custom_image",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_image_version",
    None,
    "Custom docker image version e.g. latest"
    "In case of elastic/rally:2.7.0"
    "2.7.0 is image version to be pulled",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_image_repo",
    None,
    "Custom docker image repository "
    "e.g. default is docker.io.library"
    "In case of elastic/rally:2.7.0"
    "elastic is repo name from which rally docker of version 2.7.0 will be pulled",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_build_docker_image",
    None,
    "build docker image name e.g. python_test",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_build_docker_image_version",
    None,
    "build docker image version e.g. latest",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_build_dockerfile",
    None,
    "Name of build dockerfile on launch machine",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_build_docker_dir",
    None,
    "build docker directory on launch machine"
    "e.g. ./ampere/pkb_internal/data/docker_test "
    "docker_test is directory containing Dockerfile for docker python_test",
)
flags.DEFINE_integer(
    f"{PACKAGE_NAME}_use_cpus",
    None,
    "Number of cores to run docker is used with --cpus docker option,",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_use_cpuset_cpus",
    None,
    "sets cores to run docker is used with --cpuset-cpus docker option,"
    "range also can be provided",
)
flags.DEFINE_integer(
    f"{PACKAGE_NAME}_use_memory",
    None,
    "memory in gb to run docker. It is used with -m option",
)
flags.DEFINE_integer(
    f"{PACKAGE_NAME}_use_memory_swap",
    None,
    "Total memory in gb to run docker. It is used with --memory-swap options",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_parameters",
    None,
    "docker parameters"
    "e.g. following are the parameters for rally docker"
    " race --track=http_logs --target-hosts=172.17.8.3 "
    "--challenge=challenge "
    "--pipeline=benchmark-only "
    "--track-params='bulk_indexing_clients:10'"
    "--track-params='bulk_size:1000'"
    "--report-format=csv",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_port",
    None,
    "docker run on specific port e.g. 11344 used with option -p",
)
flags.DEFINE_string(f"{PACKAGE_NAME}_name", None, "docker run with specific name ")
flags.DEFINE_string(f"{PACKAGE_NAME}_output_path", None, "Output path")
flags.DEFINE_list(
    f"{PACKAGE_NAME}_volume_names",
    None,
    "docker run with specific valume names, "
    "volume names and mount points list size should be same.",
)
flags.DEFINE_list(
    f"{PACKAGE_NAME}_volume_mountpoints",
    None,
    "docker run with specific valume mount points",
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_shell_type", None, "shell type to use in docker exec"
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_exec_command", None, "command to use in docker exec"
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_copy_host_path", None, "source path to use in docker copy"
)
flags.DEFINE_string(
    f"{PACKAGE_NAME}_copy_dest_path", None, "destination path to use in docker copy"
)

flags.DEFINE_bool(
    f"{PACKAGE_NAME}_daemon",
    False,
    "Run container in background and print container ID",
)


flags.DEFINE_boolean(
    f"{PACKAGE_NAME}_privileged_docker",
    False,
    "If True, will attempt to create Docker containers "
    "in a privileged mode. Note that some benchmarks execute "
    "commands which are only allowed in privileged mode.",
)
flags.DEFINE_string(f"{PACKAGE_NAME}_bash_command", None, "sets bash command")

def Install(vm):
    """Installs the docker on the VM."""
    # install docker package
    vm.Install("build_tools")
    vm.InstallPackages("numactl")
    os_type = vm.os_info
    version, _, _ = vm.RemoteCommandWithReturnCode("docker --version",ignore_failure=True)
    version = version.strip()
    if "Docker version" in version:
        logging.info(f"{version} is already installed")
    else:
        if "Oracle" in os_type:
            vm.RemoteCommand("sudo dnf update -y;sudo dnf install -y dnf-plugins-core;")
            vm.RemoteCommand("sudo dnf config-manager --add-repo "
                             "https://download.docker.com/linux/centos/docker-ce.repo;")
            vm.InstallPackages("docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin")
            logging.info("Docker is installed using Oracle9 repo for docker")
        else:
            _, err = vm.RemoteCommand("sudo curl -fsSL https://get.docker.com -o get-docker.sh && "
                                      "sudo sh get-docker.sh"
                                      )
            if err:
                vm.Install("docker")
                logging.info("Docker is installed using docker package")
            else:
                logging.info("Docker is installed using docker script")
    #on some os docker service needs to be enabled explicitely after starting 
    vm.RemoteCommand("sudo systemctl start docker;sudo systemctl enable docker;")
    clean_existing_containers(vm)

def validate_docker_flags():
    if FLAGS[f"{PACKAGE_NAME}_build_docker_dir"].value:
        directory = FLAGS[f"{PACKAGE_NAME}_build_docker_dir"].value
        dockerfile = FLAGS[f"{PACKAGE_NAME}_build_dockerfile"].value or "Dockerfile"
        if not os.path.exists(posixpath.join(directory, dockerfile)):
            raise errors.Setup.InvalidSetupError(
                f"Dockerfile from {directory} does not exist: {dockerfile}"
            )
        if not FLAGS[f"{PACKAGE_NAME}_build_docker_image"].value:
            raise errors.Setup.InvalidSetupError(
                    "ampere_docker_build_docker_image is not specified for"
                    f" {dockerfile} from {directory}")
    else:
        if not FLAGS[f"{PACKAGE_NAME}_image"].value:
            raise errors.Setup.InvalidSetupError(
                    "ampere_docker_image flag is not specified for"
                    " pulling docker")
        
    if FLAGS[f"{PACKAGE_NAME}_volume_names"].value:
        if len(FLAGS[f"{PACKAGE_NAME}_volume_names"].value) != len(
                FLAGS[f"{PACKAGE_NAME}_volume_mountpoints"].value
                ):
            raise errors.Setup.InvalidSetupError(
                    "Size of Lists for Volume names and mountpoints need to be same.")
    return True


def pull_docker(vm):
    """
    Pull docker image
    :param vm:
    :return:
    """
    docker_pulled = False
    validate_docker_flags()
    if FLAGS[f"{PACKAGE_NAME}_image"].value:
        docker_image = _get_docker_image()
        _, _, return_code = vm.RemoteCommandWithReturnCode(
            f"sudo docker pull {docker_image}"
        )
        if return_code == 0:
            if _get_container_info(vm):
                docker_pulled = True
    return docker_pulled


def _get_docker_image():
    """
    to get docker image from provided flags
    """
    if FLAGS[f"{PACKAGE_NAME}_build_docker_image"].value:
        docker_image = FLAGS[f"{PACKAGE_NAME}_build_docker_image"].value
        if FLAGS[f"{PACKAGE_NAME}_build_docker_image_version"].value:
            docker_image_version = FLAGS[
                f"{PACKAGE_NAME}_build_docker_image_version"
            ].value
        else:
            docker_image_version = "latest"
        docker_image = f"{docker_image}:{docker_image_version}"
    elif FLAGS[f"{PACKAGE_NAME}_image"].value:
        docker_image = FLAGS[f"{PACKAGE_NAME}_image"].value
        if FLAGS[f"{PACKAGE_NAME}_image_repo"].value:
            docker_repo = FLAGS[f"{PACKAGE_NAME}_image_repo"].value
        else:
            docker_repo = "library"
        if FLAGS[f"{PACKAGE_NAME}_image_version"].value:
            docker_image_version = FLAGS[f"{PACKAGE_NAME}_image_version"].value
        else:
            docker_image_version = "latest"
        docker_image = f"{docker_repo}/{docker_image}:{docker_image_version}"
    return docker_image


def read_env_file(directory):
    """
    read env file to pass build-arg for building docker
    :param directory:
    :return:
    """
    build_args = ""
    for env_file_name in os.listdir(directory):
        if env_file_name.endswith(".env"):
            with open(posixpath.join(directory, env_file_name), "r", encoding='utf-8') as env_file:
                # Read each line in the file
                for line in env_file:
                    # Print each line
                    build_args = build_args + f" --build-arg '{line.strip()}'"
    return build_args


def build_docker(vm):
    """
    build docker on vm
    """
    docker_build = False
    remote_dir = download_utils.INSTALL_DIR
    validate_docker_flags()
    docker_image = _get_docker_image()
    dockerfile = FLAGS[f"{PACKAGE_NAME}_build_dockerfile"].value or "Dockerfile"
    if docker_image:
        vm.RemoteCommand(f"sudo docker rmi -f {docker_image}")    
    if FLAGS[f"{PACKAGE_NAME}_build_docker_dir"].value:
        directory = FLAGS[f"{PACKAGE_NAME}_build_docker_dir"].value
        build_args = read_env_file(directory)
        vm.RemoteCopy(directory, remote_dir)
        remote_dir = posixpath.join(
            remote_dir, FLAGS[f"{PACKAGE_NAME}_build_docker_dir"].value.split("/")[-1]
        )
        dockerfile_path = posixpath.join(remote_dir, dockerfile)
        build_cmd = f"cd {remote_dir} && "
        build_cmd += f" sudo docker build --no-cache -t {docker_image} {build_args}"
        build_cmd += f" -f {dockerfile_path} . > build_file.log"
        _, _, return_code = vm.RemoteCommandWithReturnCode(build_cmd)
        if return_code == 0:
            if _get_container_info(vm):
                docker_build = True
    return docker_build


def run_docker(vm):
    """
    run docker on vm
    """
    privileged = ""
    numa_prefix = ""
    docker_memory = ""
    docker_memory_swap = ""
    docker_parameters = ""
    volumes = ""
    port = ""
    daemon = ""
    docker_name = ""
    docker_image = _get_docker_image()
    result_path = os.path.join(vm_util.GetTempDir(), "test.log")
    vm.RemoteCommand(f"cd {download_utils.INSTALL_DIR} && touch test.log")
    output_path = posixpath.join(download_utils.INSTALL_DIR, "test.log")
    bash_command = ""
    memory_swap = vm.total_memory_kb / (1024 * 1024)
    if FLAGS[f"{PACKAGE_NAME}_use_cpus"].value:
        cores = min(vm.num_cpus, FLAGS[f"{PACKAGE_NAME}_use_cpus"].value)
        numa_prefix = f" --cpus={cores} "
    elif FLAGS[f"{PACKAGE_NAME}_use_cpuset_cpus"].value:
        cores = FLAGS[f"{PACKAGE_NAME}_use_cpuset_cpus"].value
        numa_prefix = f' --cpuset-cpus="{cores}" '
    if FLAGS[f"{PACKAGE_NAME}_parameters"].value:
        docker_parameters = FLAGS[f"{PACKAGE_NAME}_parameters"].value
    if FLAGS[f"{PACKAGE_NAME}_use_memory_swap"].value:
        memory_swap = FLAGS[f"{PACKAGE_NAME}_use_memory_swap"].value
        docker_memory_swap = f' --memory-swap="{memory_swap}G"'
    if FLAGS[f"{PACKAGE_NAME}_use_memory"].value:
        memory = FLAGS[f"{PACKAGE_NAME}_use_memory"].value
        if not FLAGS[f"{PACKAGE_NAME}_use_memory_swap"].value:
            if memory * 1024 * 1024 > vm.total_memory_kb:
                memory = vm.total_memory_kb / (1024 * 1024)
        else:
            memory = min(memory, memory_swap)
        docker_memory = f' --memory="{memory}G"'
    if FLAGS[f"{PACKAGE_NAME}_volume_names"].value and FLAGS[f"{PACKAGE_NAME}_volume_mountpoints"].value:
        if len(FLAGS[f"{PACKAGE_NAME}_volume_names"].value) == len(
                FLAGS[f"{PACKAGE_NAME}_volume_mountpoints"].value):
            for index_volumes, _ in enumerate(FLAGS[f"{PACKAGE_NAME}_volume_names"].value):
                vol_string = (
                        FLAGS[f"{PACKAGE_NAME}_volume_names"].value[index_volumes]
                        + ":"
                        + FLAGS[f"{PACKAGE_NAME}_volume_mountpoints"].value[index_volumes]
                        )
                volumes += f" -v {vol_string} "
    if FLAGS[f"{PACKAGE_NAME}_output_path"].value:
        output_path = FLAGS[f"{PACKAGE_NAME}_output_path"].value
    if FLAGS[f"{PACKAGE_NAME}_privileged_docker"].value:
        privileged = " --privileged"
    if FLAGS[f"{PACKAGE_NAME}_daemon"].value:
        daemon = " -d"
    if FLAGS[f"{PACKAGE_NAME}_port"].value:
        port_val = FLAGS[f"{PACKAGE_NAME}_port"].value
        port = f" -p {port_val}"
    if FLAGS[f"{PACKAGE_NAME}_name"].value:
        name_val = FLAGS[f"{PACKAGE_NAME}_name"].value
        docker_name = f" --name {name_val}"
    if FLAGS[f"{PACKAGE_NAME}_bash_command"].value:
        bash_command = FLAGS[f"{PACKAGE_NAME}_bash_command"].value
    docker_run_command = f"sudo docker run {daemon} {numa_prefix} {docker_memory}"
    docker_run_command += f" {docker_memory_swap} {privileged} {docker_name}"
    docker_run_command += f" {volumes} {port} {docker_parameters}"
    docker_run_command += f" {bash_command} {docker_image} "
    docker_run_command += f" &> {output_path}"
    vm.RemoteCommand(docker_run_command)
    if docker_name and FLAGS[f"{PACKAGE_NAME}_daemon"].value:
        _, json_info, _ = _get_container_info(vm, FLAGS[f"{PACKAGE_NAME}_name"].value)
    vm.PullFile(vm_util.GetTempDir(), output_path)
    with open(result_path, "r", encoding='utf-8') as output:
        summary_data = output.read()
    return summary_data


@vm_util.Retry(poll_interval=5, max_retries=10)
def _get_container_info(vm, docker_name=""):
    """Returns information about a container.

    Gets Container information from Docker Inspect. Returns the information,
    if there is any and a return code. 0
    """
    logging.info("Checking Container Information")
    if docker_name:
        docker_image = docker_name
    else:
        docker_image = _get_docker_image()
    inspect_cmd = f"sudo docker inspect {docker_image}"
    info, _, return_code = vm.RemoteCommandWithReturnCode(inspect_cmd)
    info = json.loads(info)
    return docker_image, info, return_code


def exec_docker(vm):
    """
    Used to docker exec
    exec command and docker name should be set
    :param vm:
    :return:
    """
    if (
        not FLAGS[f"{PACKAGE_NAME}_exec_command"].value
        and FLAGS[f"{PACKAGE_NAME}_name"].value
    ):
        raise ValueError(
            "For docker exec, following flags need to be set"
            " ampere_docker_exec_command "
            " ampere_docker_name"
        )
    docker_name = FLAGS[f"{PACKAGE_NAME}_name"].value
    shell_type = FLAGS[f"{PACKAGE_NAME}_shell_type"].value
    exec_command = FLAGS[f"{PACKAGE_NAME}_exec_command"].value
    out, _, return_code = vm.RemoteCommandWithReturnCode(
        f'sudo docker exec -i {docker_name} {shell_type} -c "{exec_command}"'
    )
    print(return_code)
    return out


def copy_docker(vm, from_docker=True):
    """
    Used to copy docker files to machine
    :param vm:
    :param from_docker:
    :return:
    """
    if (
        not FLAGS[f"{PACKAGE_NAME}_name"].value
        and FLAGS[f"{PACKAGE_NAME}_copy_host_path"].value
        and FLAGS[f"{PACKAGE_NAME}_copy_dest_path"].value
    ):
        raise ValueError(
            "For docker copy, following flags need to be set"
            "ampere_docker_name"
            "ampere_docker_copy_host_path"
            "ampere_docker_copy_dest_path"
        )
    docker_name = FLAGS[f"{PACKAGE_NAME}_name"].value
    host_path = FLAGS[f"{PACKAGE_NAME}_copy_host_path"].value
    dest_path = FLAGS[f"{PACKAGE_NAME}_copy_dest_path"].value
    if from_docker:
        _, _, return_code = vm.RemoteCommandWithReturnCode(
            f"sudo docker cp {docker_name}:{host_path} {dest_path}"
        )
    else:
        _, _, return_code = vm.RemoteCommandWithReturnCode(
            f"sudo docker cp {host_path} {docker_name}:{dest_path}"
        )
    return return_code


def _docker_exists(vm):
    """Returns whether the container is up and running."""

    docker_image, info, return_code = _get_container_info(vm)
    logging.info("Checking if Docker Container Exists")
    if info and return_code == 0:
        status = info[0]["State"]["Running"]
        if status:
            logging.info("Docker Container %s is up and running.", docker_image)
            return True
    return False


def clean_existing_containers(vm):
    """stops docker container on vm
    Args:
      vm: The vm on which Ampere docker is uninstalled.
    """
    container_name = FLAGS[f"{PACKAGE_NAME}_name"].value
    if container_name:
        out, err, return_code = vm.RemoteCommandWithReturnCode(
                f"sudo docker ps -a | grep {container_name}",ignore_failure=True)
        if return_code == 0:
            logging.info(f"stdout and stderr in clean_existing_containers: {out}, {err}")
            vm.RemoteCommand(f"sudo docker stop {container_name}",ignore_failure=True)
            vm.RemoteCommand(f"sudo docker rm -f {container_name}",ignore_failure=True)


def Uninstall(vm):
    """Cleans up Ampere docker from the target vm.

    Args:
      vm: The vm on which Ampere docker is uninstalled.
    """
    out, err, return_code = vm.RemoteCommandWithReturnCode("sudo docker ps -a -q")
    logging.info(f"Uninstalling docker: {out}, {err}, {return_code}")
    if err == "":
        vm.RemoteCommand("sudo docker stop $(sudo docker ps -a -q)")
        vm.RemoteCommand("sudo docker rm -f $(sudo docker ps -a -q)")
    vm.RemoteCommand("sudo systemctl stop docker")
