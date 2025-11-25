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

import sys
import subprocess

# PyTorch
native_frameworks = list()
try:
    import torch  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch"])
    native_frameworks.append("PyTorch")
try:
    import torchvision  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torchvision"])

# summary
len_native_frameworks = len(native_frameworks)
if len_native_frameworks > 0:
    print(f"\033[1;31m \nCAUTION: Pytorch framework has been installed in their native versions ")
