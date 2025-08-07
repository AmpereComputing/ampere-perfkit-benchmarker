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
Handles firewall inside of OS
"""

import posixpath
import logging
from perfkitbenchmarker.virtual_machine import BaseVirtualMachine


def stop_firewall_service(vm: BaseVirtualMachine):
    """
    stop firewall service
    : param vm:
    :return:
    """
    vm.RemoteCommand("sudo systemctl stop firewalld")


def start_firewall_service(vm: BaseVirtualMachine):
    """
    start firewall service
    : param vm:
    :return:
    """
    vm.RemoteCommand("sudo systemctl start firewalld")


def add_port_to_nftables(vm: BaseVirtualMachine, port: str):
    """
    add ports to nftables
    :param vm:
    :param port:
    port can be defined in 3 ways:
        1. Single port value
        2. Comma separated values e.g. {{ 9942, 7199 }}
        3. Port range e.g. 1131-1135
    :return:
    """
    # In case of Oracle linux, stop firewalld service
    os_type = vm.os_info
    if "Oracle" in os_type:
        stop_firewall_service(vm)
    # check nftables.conf is present in /etc folder
    nftables_path = posixpath.join("/etc", "nftables.conf")
    nftables_backup_path = posixpath.join("/etc", "nftables_backup.conf")
    _, _, retcode = vm.RemoteCommandWithReturnCode(
        f"test -f {nftables_path}", ignore_failure=True
    )
    file_exists = retcode == 0
    if file_exists:
        vm.RemoteCommand(f"sudo cp {nftables_path} {nftables_backup_path}")
    nftables_conf = f"""\n#!/usr/sbin/nft -f

flush ruleset

table inet filter {{
  chain input {{
    type filter hook input priority 0; policy drop;

    # Allow loopback
    iif "lo" accept

    # Allow established/related connections
    ct state established,related accept

    # Allow SSH
    tcp dport 22 accept

    # Allow your application ports
    tcp dport {port} accept
  }}

  chain forward {{
    type filter hook forward priority 0; policy drop;
  }}

  chain output {{
    type filter hook output priority 0; policy accept;
  }}
}}

EOF
"""
    vm.RemoteCommand(f"cat <<'EOF' | sudo tee /etc/nftables.conf {nftables_conf}")
    vm.RemoteCommand("sudo nft flush ruleset")
    vm.RemoteCommand("sudo nft -f /etc/nftables.conf")
    vm.RemoteCommand("sudo systemctl enable nftables")
    vm.RemoteCommand("sudo systemctl restart nftables")


def restore_nftables(vm: BaseVirtualMachine):
    """
    restore nftables.conf to /etc folder
    :param vm:
    :return:
    """
    nftables_path = posixpath.join("/etc", "nftables.conf")
    nftables_backup_path = posixpath.join("/etc", "nftables_backup.conf")
    _, _, retcode = vm.RemoteCommandWithReturnCode(
        f"test -f {nftables_backup_path}", ignore_failure=True
    )
    file_exists = retcode == 0
    if file_exists:
        vm.RemoteCommand(f"sudo mv {nftables_backup_path} {nftables_path}")
        vm.RemoteCommand("sudo nft flush ruleset")
        vm.RemoteCommand("sudo nft -f /etc/nftables.conf")
        vm.RemoteCommand("sudo systemctl enable nftables")
        vm.RemoteCommand("sudo systemctl restart nftables")
    else:
        logging.debug(f"{nftables_backup_path} does not exist on the VM.")
        # delete nftables.conf from /etc folder
        vm.RemoteCommand(f"sudo rm -rf {nftables_path}")
    os_type = vm.os_info
    if "Oracle" in os_type:
        start_firewall_service(vm)
