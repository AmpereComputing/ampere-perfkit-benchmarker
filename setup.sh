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

setup_venv() {
	local python_interpreter=$1

	rm -rf venv
	eval "$python_interpreter -m venv venv"
	source venv/bin/activate
	eval "$python_interpreter -m pip install --upgrade pip"
	pip install -r requirements.txt
	if [ $? -eq 0  ]; then
		echo
		echo "✅ Python virtual environment successfully created, all dependencies resolved."
		echo "✅ To stop virtual environment run 'deactivate'"
		echo "✅ To restart virtual environment run 'source venv/bin/activate'"
	else
		echo
		echo "❌ Command failed with exit code $?"
		echo "❌ Failed to install APKB dependencies in the virtual environment."
	fi

}

main() {
	required_version="3.11"
	python_version=$(python3 --version | awk '{print $2}' | awk -F"." '{print $1"."$2}')
	# Check if python3.11 is explicitly installed
	if eval "command -v python$required_version &> /dev/null"; then
		echo "🐍 Python version $required_version.x detected, proceeding with APKB installation..."
		echo
		setup_venv "python$required_version"
	# Check if current python3 version is 3.11.x
	elif [ "$python_version" = "$required_version"  ]; then
		echo "🐍 Python version $required_version.x detected, proceeding with APKB installation..."
		echo
		setup_venv "python3"
	else
		echo "🐍 APKB requires a Python version == $required_version.x"
		echo "⚠️  Please install this version using your system's package manager (or build from source)"
		echo
	fi
}

main
