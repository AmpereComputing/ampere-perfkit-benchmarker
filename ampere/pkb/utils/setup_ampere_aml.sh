#!/bin/bash

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

set -eo pipefail

log() {
  echo -e "$1"
}

ARCH=$(uname -m)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

log "Checking for Debian or RHEL based Linux ..."
sleep 1
if [ -f "/etc/debian_version" ]; then
  debian_version=$(</etc/debian_version)
  log "Detected Debian $debian_version. Be advised that this script supports Debian >=11.0."
  sleep 3
  if [ -z ${AML_DIR+x} ]; then
    log "installing setup_docker.sh ..."
    log "Installing system dependencies ..."
    sleep 1
    AML_DIR="/workspace/ampere_model_library"
    apt-get update -y
    apt-get install -y build-essential ffmpeg libsm6 libxext6 wget git unzip numactl libhdf5-dev cmake
    if ! python3 -c ""; then
	    apt-get install -y python3 python3-pip
    fi
    if ! pip3 --version; then
	    apt-get install -y python3-pip
    fi
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PYTHON_DEV_SEARCH=$(apt-cache search --names-only "python${PYTHON_VERSION}-dev")
    if [[ -n "$PYTHON_DEV_SEARCH" ]]; then
	    apt-get install -y "python${PYTHON_VERSION}-dev"
     fi
     log "done.\n"
     log "Setup LD_PRELOAD ..."
     sleep 1
     if [ "${ARCH}" = "aarch64" ]; then
	     python3 "$AML_DIR"/utils/setup/gen_ld_preload.py
	     LD_PRELOAD=$(cat "$AML_DIR"/utils/setup/.ld_preload)
	     echo "LD_PRELOAD=$LD_PRELOAD"
     fi
     export LD_PRELOAD=$LD_PRELOAD
     log "done.\n"
     log "Installing python dependencies ..."
     sleep 1
     ARCH=$ARCH python3 "$AML_DIR"/utils/setup/install_pytorch.py

     # get almost all python deps
     PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed --upgrade pip
	python3 -m pip install --break-system-packages -r "$(dirname "$0")/requirements.txt" ||
    	python3 -m pip3 install -r "$(dirname "$0")/requirements.txt"
     apt install -y autoconf autogen automake build-essential libasound2-dev \
	     libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev \
	     libmpg123-dev pkg-config
     cat /etc/machine-id > "$AML_DIR"/.setup_completed
  fi
elif [ -f "/etc/redhat-release" ]; then
  rhel_version=$(</etc/redhat-release)
  log "Detected $rhel_version. Be advised that this script supports RHEL>=9.4."
  sleep 3

  if [ -z ${AML_DIR+x} ]; then
	  AML_DIR="/workspace/ampere_model_library"
          yum install epel-release || :
          yum groupinstall -y 'Development Tools'
	  python_version=$(python3 --version | cut -d " " -f2 | cut -d "." -f2)
	  echo $python_version
          yum install -y python3.$python_version-devel python3.$python_version-pip libSM libXext wget git unzip numactl cmake gcc-c++
          if [ "${ARCH}" = "aarch64" ]; then
                  yum install -y hdf5-devel
          fi
          git clone -b n4.3.7 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && ./configure && make -j && make install && cd .. && rm -r FFmpeg
          log "done.\n"
          log "Setup LD_PRELOAD ..."
          sleep 1
          #if [ "${ARCH}" = "aarch64" ]; then
          python3 "$AML_DIR"/utils/setup/gen_ld_preload.py
          LD_PRELOAD=$(cat "$AML_DIR"/utils/setup/.ld_preload)
          echo "LD_PRELOAD=$LD_PRELOAD"
          #fi
          export LD_PRELOAD=$LD_PRELOAD
          log "done.\n"
          log "Installing python dependencies ..."
          sleep 1
          # get almost all python deps
          #pip3 install --break-system-packages -r "$(dirname "$0")/requirements.txt" ||
          pip3 install -r "$AML_DIR"/requirements.txt
          yum install -y autoconf automake alsa-lib-devel pkg-config
          cat /etc/machine-id > "$AML_DIR"/.setup_completed
  fi
else
   log "\nNeither Debian-based nor RHEL-based Linux has been detected! Quitting."
   exit 1
fi

sleep 1
log "done.\n"

log "Setup completed. Please run: source $SCRIPT_DIR/ampere_model_library/set_env_variables.sh"
