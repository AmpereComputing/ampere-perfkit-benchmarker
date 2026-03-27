#!/bin/bash

# Copyright (c) 2026, Ampere Computing LLC
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

print_failure() {
    echo "[ERROR]: This project requires a Python version >=3.11.x"
    echo "[ERROR]: Please install a compatible version (package manager, pyenv, or build from source)"
    echo
}

print_13_warning() {
    echo "[WARNING]: Python >=3.13.x is not yet a supported target for this project."
    echo "[WARNING]: Compatibility is not guaranteed and behavior may vary."
    echo
}

print_log() {
    local python_major=$1
    local python_minor=$2
    echo "[LOG]: Python version $python_major.$python_minor detected"
    echo "[LOG]: Proceeding with dependency installation..."
    echo
}

setup_venv() {
	local python_interpreter=$1
	rm -rf venv
	eval "$python_interpreter -m venv venv"
	if [ $? -ne 0  ]; then
		echo "[ERROR]: Command failed with exit code $?"
		echo "[ERROR]: Failed to create virtual environment."
        return 1
    fi
	source venv/bin/activate
	eval "$python_interpreter -m pip install --upgrade pip"
	pip install -r requirements.txt
	if [ $? -eq 0  ]; then
		echo
		echo "[SUCCESS]: Python virtual environment successfully created, all dependencies resolved."
		echo "[SUCCESS]: To stop virtual environment run 'deactivate'"
		echo "[SUCCESS]: To restart virtual environment run 'source venv/bin/activate'"
        return 0
	else
		echo
		echo "[ERROR]: Command failed with exit code $?"
		echo "[ERROR]: Failed to install dependencies in the virtual environment."
        return 1
	fi
}

check_default_python_interpreter() {
    python_major=$(python3 --version | awk '{print $2}' | awk -F"." '{print $1}')
    python_minor=$(python3 --version | awk '{print $2}' | awk -F"." '{print $2}')
    if [[ "$python_major" -ne 3 || "$python_minor" -lt 11 ]]; then
        return 1
    elif [ "$python_minor" -ge 13 ]; then
        print_13_warning
    fi 
    print_log $python_major $python_minor
    setup_venv "python3"
    return 0
}

check_explicit_python_interpreter() {
    # Check if Python 3.13/3.12/3.11 is explicitly installed (prefer latest)
    for python_minor in 13 12 11; do
        if eval "python3.$python_minor --version &> /dev/null"; then
            if [ "$python_minor" -eq 13 ]; then
                print_13_warning
            fi
            print_log 3 $python_minor
            setup_venv "python3.$python_minor"
            return 0
        fi
    done
    return 1
}

main() {
    if check_default_python_interpreter; then
        return 0
    elif check_explicit_python_interpreter; then
        return 0
    else
        print_failure
        return 1
    fi
}

main
