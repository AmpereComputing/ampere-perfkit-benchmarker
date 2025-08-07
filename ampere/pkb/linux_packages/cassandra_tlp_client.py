# Copyright (c) 2024-2025, Ampere Computing LLC
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


import dataclasses
import posixpath
import time
import statistics
from typing import Any, Dict, List, Text
from absl import flags
from ampere.pkb.common import download_utils
from perfkitbenchmarker import sample
from perfkitbenchmarker import vm_util
from perfkitbenchmarker import flag_util
from ampere.pkb.common import bashrc_handler
from ampere.pkb.common import firewall_handler

PACKAGE_NAME = "ampere_cassandra_tlp_client"

GIT_REPO = "https://github.com/thelastpickle/tlp-stress.git"
TLP_DIR = f"{download_utils.INSTALL_DIR}/tlp-stress"

FLAGS = flags.FLAGS

flag_util.DEFINE_integerlist(
    f"{PACKAGE_NAME}_threads",
    [16],
    "Number of threads used in cassandra-stress tool on each loader node.",
)

flags.DEFINE_string(
    f"{PACKAGE_NAME}_duration", "10m", "Duration to run Cassandra TLP duration"
)

flags.DEFINE_string(
    f"{PACKAGE_NAME}_readrate",
    "0.5",
    "read rate for Cassandra TLP relative to writes, between 0-1 e.g. 0.1 means 90% writes",
)

flags.DEFINE_string(
    f"{PACKAGE_NAME}_jdk_url_for_tlp", "None", "jdk url for cassandra tlp"
)

flags.DEFINE_string(
    f"{PACKAGE_NAME}_workload_select_type", "row", "Cassandra TLP workload select type"
)


flags.DEFINE_string(f"{PACKAGE_NAME}_populate", "500k", "Cassandra TLP populate value.")

flags.DEFINE_string(
    f"{PACKAGE_NAME}_partitions", "10m", "Cassandra TLP partition duration"
)

flags.DEFINE_string(
    f"{PACKAGE_NAME}_workload", "RandomPartitionAccess", "workload for cassandra tlp"
)


flags.DEFINE_integer(
    f"{PACKAGE_NAME}_instances", 1, "total number of instances on client"
)

flags.DEFINE_bool(
    f"{PACKAGE_NAME}_use_numactl", False, "active numactl to run Cassandra tlp"
)

flags.DEFINE_list(
    f"{PACKAGE_NAME}_use_cores", [], "add cores to specific instances of Cassandra tlp"
)

CASSANDRA_PORT = 9042
CASSANDRA_PROMETHEUS_PORT = 7199

OPENJDK_VERSION = flags.DEFINE_integer(
    f"{PACKAGE_NAME}_jdk_version",
    8,
    "Version of openjdk to use. By default, the oldest non-end-of-life LTS "
    "version of openjdk is automatically detected.",
)


def GetJDKURL(arch) -> str:
    """Gets jdk url for passed architecture"""
    return f"https://download.bell-sw.com/java/8u362+9/bellsoft-jdk8u362+9-linux-{arch}.tar.gz"


def _Install(vm):
    """Installs Cassandra-TLP from a git."""
    vm.InstallPackages("curl")
    vm.InstallPackages("wget numactl")
    vm.Install("build_tools")
    vm.RemoteCommand(f"sudo rm -rf {TLP_DIR}")
    vm.RemoteCommand(f"cd {download_utils.INSTALL_DIR} && git clone {GIT_REPO}")

    # trial code
    # install java from tarball
    lscpu = vm.CheckLsCpu()
    arch = lscpu.data["Architecture"]
    version = OPENJDK_VERSION.value

    if arch == "x86_64":
        arch = "amd64"
    url = GetJDKURL(arch)
    java_dir = posixpath.join(download_utils.INSTALL_DIR, "java")
    vm.RemoteCommand(f"mkdir -p {java_dir}")
    # set JAVA_HOME and jdk_path
    java_home = posixpath.join(java_dir, f"jdk-{version}")
    vm.RemoteCommand(
        f"cd {java_dir} && mkdir -p {java_home} && curl -L {url} | "
        f"tar --strip-components=1 -C {java_home} -xzf -"
    )
    jdk_path = posixpath.join(java_dir, f"jdk-{version}", "bin")
    update_bashrc_commands = f"export JAVA_HOME={java_home};"
    update_bashrc_commands += f"export PATH={jdk_path}:$PATH;"
    bashrc_handler.update_bashrc(vm, "~/", update_bashrc_commands)
    bashrc_handler.source_bashrc(vm, "~/")
    vm.RemoteCommand(f"cd {TLP_DIR} && ./gradlew shadowJar")
    time.sleep(5)


def YumInstall(vm):
    """Installs Cassandra on the VM."""
    _Install(vm)


def AptInstall(vm):
    """Installs Cassandra on the VM."""
    _Install(vm)


def CleanNode(vm):
    """Remove Cassandra data from 'vm'.

    Args:
      vm: VirtualMachine. VM to clean.
    """
    vm.RemoteCommand(f"sudo rm -rf {download_utils.INSTALL_DIR}")


def _ResultFilePath(vm, thread_num, instance, port):
    tlp_stress_result = (
        f"{vm.hostname}.tlp_results_{port}_instance{instance}_thread_{thread_num}.csv"
    )
    return posixpath.join(vm_util.VM_TMP_DIR, tlp_stress_result)


def GetTLPStressVersion(client_vm, instance) -> str:
    tlp_path = posixpath.join(
        download_utils.INSTALL_DIR, f"cassandra_tlp_client_{instance}"
    )
    version, _ = client_vm.RemoteCommand(
        f'grep "^version" {tlp_path}/build.gradle | cut -d" " -f2'
    )
    version = version.strip()
    return version


def GetCassandraTlpStressPath(cl, instance):
    """Get tlp Binary Absolute Path"""
    client_instances = FLAGS[f"{PACKAGE_NAME}_instances"].value
    instance1 = (cl * client_instances) + instance
    folder_name = "cassandra_tlp_client_" + str(instance1)
    tlp_stress_binary = "tlp-stress" + str(instance1)
    return posixpath.join(
        download_utils.INSTALL_DIR, folder_name, "bin", tlp_stress_binary
    )


def CreateInstances(vm, cl):
    """Create user specified number of Cassandra TLP Instances.
    Create copies of Cassandra TLP instances on Client.
    Args:
        vm: Virtual Machine. The VM to condfigure
        cl: client number
    """
    client_instances = FLAGS[f"{PACKAGE_NAME}_instances"].value
    for instance in range(client_instances):
        instance1 = (cl * client_instances) + instance
        folder_name = "cassandra_tlp_client_" + str(instance1)
        tlp_stress_binary = "tlp-stress" + str(instance1)
        vm.RemoteCommand(
            f"cd {download_utils.INSTALL_DIR} && cp -r {TLP_DIR} {folder_name}"
        )
        tlp_stress_dir = posixpath.join(download_utils.INSTALL_DIR, folder_name, "bin")
        vm.RemoteCommand(f"cd {tlp_stress_dir} && mv tlp-stress {tlp_stress_binary}")
    vm.RemoteCommand("sudo pkill -9 java", ignore_failure=True)
    time.sleep(10)


def RunCassandraTlpStressOverAllPorts(cassandra_vms, client, cl, thread_num):
    """Run Cassandra TLP Stress on Client VM."""

    client_instances = FLAGS[f"{PACKAGE_NAME}_instances"].value
    read_rate = FLAGS[f"{PACKAGE_NAME}_readrate"].value
    # for thread_num in thread_data:
    query = ""
    ports = []
    for cl_val in range(client_instances):
        instance = (cl * client_instances) + cl_val
        if FLAGS[f"{PACKAGE_NAME}_use_numactl"].value:
            cmd = [
                " numactl -C "
                + str(FLAGS.ampere_cassandra_tlp_client_use_cores[instance])
                + GetCassandraTlpStressPath(cl, cl_val)
                + " run "
                + FLAGS[f"{PACKAGE_NAME}_workload"].value
            ]
        else:
            cmd = [
                GetCassandraTlpStressPath(cl, cl_val)
                + " run "
                + FLAGS[f"{PACKAGE_NAME}_workload"].value
            ]
        cassandra_query = ""
        cassandra_server_port = CASSANDRA_PORT + instance
        cassandra_prometheus_port = CASSANDRA_PROMETHEUS_PORT + instance
        client.AllowPort(cassandra_server_port)
        client.AllowPort(cassandra_prometheus_port)
        ports.append(cassandra_server_port)
        ports.append(cassandra_prometheus_port)

        data_node_ips = [vm.internal_ip for vm in cassandra_vms]
        data_node_ips_ind = ",".join(data_node_ips)
        args = {
            "--host": f"{data_node_ips_ind}",
            "--populate": FLAGS[f"{PACKAGE_NAME}_populate"].value,
            "--partitions": FLAGS[f"{PACKAGE_NAME}_partitions"].value,
            "--duration": FLAGS[f"{PACKAGE_NAME}_duration"].value,
            "--csv": _ResultFilePath(client, thread_num, cl_val, cassandra_server_port),
            "--port": cassandra_server_port,
            "--prometheusport": cassandra_prometheus_port,
            "--threads": thread_num,
            "--readrate": read_rate,
        }
        for arg, value in args.items():
            if value is not None:
                cmd.extend([f"{arg}", str(value)])
        cassandra_query = cassandra_query + " ".join(cmd)
        cassandra_query = cassandra_query + " --drop & "
        query = query + cassandra_query
    ports_string = ", ".join(str(port) for port in ports)
    firewall_handler.add_port_to_nftables(client, f"{{ {ports_string} }}")
    client.RemoteCommand(query, ignore_failure=True)
    client.RemoteCommand("sudo pkill -9 java", ignore_failure=True)
    time.sleep(10)


def Stop(vm):
    """Stops Cassandra TLP on 'vm'."""
    vm.RemoteCommand("sudo pkill -9 java", ignore_failure=True)
    bashrc_handler.restore_bashrc(vm, "~/")
    time.sleep(10)
    #restore nftables.conf in /etc folder
    firewall_handler.restore_nftables(vm)


def GenerateMetadataFromFlags(clients, thread_num):
    """Generate metadata from flags."""
    metadata = {}

    metadata.update(
        {
            "num_server_nodes": 1,
            "num_client_nodes": len(clients),
            "Threads": thread_num,
            "TLP Workload": FLAGS[f"{PACKAGE_NAME}_workload"].value,
            "tlp_duration": FLAGS[f"{PACKAGE_NAME}_duration"].value,
            "tlp_partition": FLAGS[f"{PACKAGE_NAME}_partitions"].value,
            "tlp_populate": FLAGS[f"{PACKAGE_NAME}_populate"].value,
            "cassandra_version": FLAGS.ampere_cassandra_server_version,
        }
    )
    return metadata


def GenerateMetadataPerClient(client_vm, thread_num, client_num):
    """Generate metadata per client run."""
    return {
        "client_vm": client_vm.hostname,
        "Threads": thread_num,
        "client_num": client_num,
        "cassandra_tlp_version": GetTLPStressVersion(client_vm, client_num),
    }


def CollectResults(clients, thread_num):
    """Collect results from CSV files"""
    samples = []
    meta = {}
    # for thread_num in thread_data:
    metadata1 = GenerateMetadataFromFlags(clients, thread_num)
    num_clients = len(clients)
    for cl in range(num_clients):
        vm = clients[cl]
        meta = GenerateMetadataPerClient(vm, thread_num, cl)
        metadata = metadata1 | meta
        results = _Run(vm, thread_num, cl)
        for sam in results:
            samples.append(sam.GetSamples(metadata))
    return samples


def _Run(vm, thread_num, client_num):
    client_instances = FLAGS[f"{PACKAGE_NAME}_instances"].value
    summary_data = []
    for cl_val in range(client_instances):
        instance = (client_num * client_instances) + cl_val
        cassandra_server_port = CASSANDRA_PORT + instance
        result_path = _ResultFilePath(vm, thread_num, cl_val, cassandra_server_port)
        check_file, _ = vm.RemoteCommand(
            f'if [ -f {result_path} ]; then echo "File found!" > checkfile.log; '
            f'else echo "File not found!" > checkfile.log;  fi;'
            f"  cat checkfile.log;"
        )
        check_file = check_file.strip()
        if check_file == "File found!":
            stress_result = (
                vm.hostname
                + f".tlp_results_{cassandra_server_port}_instance"
                + str(cl_val)
                + "_thread_"
                + str(thread_num)
                + ".csv"
            )
            output_path = posixpath.join(vm_util.GetTempDir(), stress_result)
            vm_util.IssueCommand(["rm", "-f", output_path])
            vm.PullFile(vm_util.GetTempDir(), result_path)
            with open(output_path, encoding="utf-8", mode="r") as output:
                data_file = output.readlines()
            count = 0
            write_op_rate_data = []
            read_op_rate_data = []
            write_latency_data = []
            read_latency_data = []
            elapsed_time_data = []
            for row in data_file:
                row_data = row.split(",")
                if count > 63:
                    op_rate_value = row_data[4]
                    write_op_rate_data.append(float(op_rate_value))
                    write_latency_data.append(float(row_data[3]))
                    elapsed_time_data.append(row_data[1])
                    read_op_rate_data.append(float(row_data[7]))
                    read_latency_data.append(float(row_data[6]))
                count += 1
            total_write_op_rate = statistics.median(write_op_rate_data)
            total_read_op_rate = statistics.median(read_op_rate_data)
            total_write_latency = statistics.median(write_latency_data)
            total_read_latency = statistics.median(read_latency_data)
            instance_value = cl_val + 1
            summary = (
                str(total_write_op_rate)
                + ","
                + str(total_read_op_rate)
                + ","
                + str(total_write_latency)
                + ","
                + str(total_read_latency)
                + ","
                + str(thread_num)
                + ","
                + str(instance_value)
                + ","
                + str(cassandra_server_port)
            )
            local_data = CassandraStressTlpResult.Parse(summary)
            summary_data.append(local_data)
            vm.RemoteCommand(f"rm -rf {result_path}")
        else:
            instance_value = cl_val + 1
            summary = (
                "0,0,0,0,"
                + str(thread_num)
                + ","
                + str(instance_value)
                + ","
                + str(cassandra_server_port)
            )
            local_data = CassandraStressTlpResult.Parse(summary)
            summary_data.append(local_data)
    return summary_data


@dataclasses.dataclass
class CassandraStressTlpResult:
    """Class that represents stress results."""

    write_ops_per_sec: float
    write_p99_latency: float
    read_ops_per_sec: float
    read_p99_latency: float
    thread_num: int
    instance_value: int
    port: int

    @classmethod
    def Parse(cls, stress_results: List) -> "CassandraStressTlpResult":
        """Parse the Cassandra tlp results to be written in json"""
        aggregated_result = _ParseTotalThroughputAndLatency(stress_results)
        return cls(
            write_ops_per_sec=aggregated_result.write_ops_per_sec,
            write_p99_latency=aggregated_result.write_p99_latency,
            read_ops_per_sec=aggregated_result.read_ops_per_sec,
            read_p99_latency=aggregated_result.read_p99_latency,
            thread_num=aggregated_result.thread_num,
            instance_value=aggregated_result.instance_value,
            port=aggregated_result.port,
        )

    def GetSamples(self, metadata: Dict[str, Any]) -> List[sample.Sample]:
        """Return this result as a list of samples."""
        samples = [
            sample.Sample(
                f"{self.thread_num}_Write Ops Throughput",
                self.write_ops_per_sec,
                "ops/s",
                metadata,
            ),
            sample.Sample(
                f"{self.thread_num}_Read Ops Throughput",
                self.read_ops_per_sec,
                "ops/s",
                metadata,
            ),
            sample.Sample(
                f"{self.thread_num}_Number of Threads", self.thread_num, "", metadata
            ),
            sample.Sample(f"{self.thread_num}_port", self.port, "", metadata),
            sample.Sample(
                f"{self.thread_num}_Number of Instances",
                self.instance_value,
                "",
                metadata,
            ),
            sample.Sample(
                f"{self.thread_num}_write_p99_latency",
                self.write_p99_latency,
                "",
                metadata,
            ),
            sample.Sample(
                f"{self.thread_num}_read_p99_latency",
                self.read_p99_latency,
                "",
                metadata,
            ),
        ]
        return samples


@dataclasses.dataclass(frozen=True)
class CassandraStressAggregateResult:
    """Parsed aggregated stress results."""

    write_ops_per_sec: float
    write_p99_latency: float
    read_ops_per_sec: float
    read_p99_latency: float
    thread_num: int
    instance_value: int
    port: int


def _ParseTotalThroughputAndLatency(
    stress_results: Text,
) -> "CassandraStressAggregateResult":
    """Parses the 'TOTALS' output line and return throughput and latency."""
    write_op_rate = 0
    write_latency = 0
    read_op_rate = 0
    read_latency = 0
    thread_num = 0
    instance_value = 0
    port = 0
    stress_data = stress_results.split(",")
    write_op_rate = stress_data[0]
    read_op_rate = stress_data[1]
    write_latency = stress_data[2]
    read_latency = stress_data[3]
    thread_num = stress_data[4]
    instance_value = stress_data[5]
    port = stress_data[6]
    return CassandraStressAggregateResult(
        write_ops_per_sec=write_op_rate,
        write_p99_latency=write_latency,
        read_ops_per_sec=read_op_rate,
        read_p99_latency=read_latency,
        thread_num=thread_num,
        instance_value=instance_value,
        port=port,
    )
