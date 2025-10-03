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
Module containing function to download the model

"""


import posixpath
from ampere.pkb.common import download_utils
from absl import flags

PACKAGE_NAME = "ampere_llama"
BENCHMARK_NAME = "ampere_llama_benchmark"

INSTALL_DIR = download_utils.INSTALL_DIR
DEPLOY_DIR = posixpath.join(INSTALL_DIR, "models")

FLAGS = flags.FLAGS

flags.DEFINE_string(f"{PACKAGE_NAME}_models_url", None, "")

DATA_FILES = flags.DEFINE_string(
    f"{PACKAGE_NAME}_data",
    "",
    "Must be in ./ampere/pkb/data/",
)


def download_model(vm):
    """
    Downloads or copies model files to the remote virtual machine.

    This function retrieves the model URLs or names from global FLAGS and ensures
    they are available in the target deployment directory on the remote VM. If a
    models URL is specified and uses HTTP(S), it downloads the model files from
    that URL using `download_utils.download`. Otherwise, it falls back to copying
    the first model from the local VM cache.

    Args:
        vm: A virtual machine object that provides the `RemoteCommand` method to
            execute commands remotely.

    """    
    models_url = FLAGS[f"{PACKAGE_NAME}_models_url"].value
    models = FLAGS[f"{BENCHMARK_NAME}_model_names"].value
    vm.RemoteCommand(f"mkdir -p {DEPLOY_DIR}")
    for model in models:
        force = False
        if models_url:
            url = models_url + model
            cache_dir = ""
            if not (url.startswith("http") or url.startswith("https")):
                raise ValueError(f"Unknown src format for {url}! Only HTTP(s) is supported!")
        elif DATA_FILES.value:
            url = model
            cache_dir = DATA_FILES.value
        else:
            url = model
            cache_dir = ""
        download_utils.download(
                url, [vm], cache_dir=cache_dir, dst="", force=force)
        vm.RemoteCommand(f"cp /opt/pkb-cache/{model} {DEPLOY_DIR}/{model}")
