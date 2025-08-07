# Modifications Copyright (c) 2024-2025 Ampere Computing LLC
# Copyright 2021 PerfKitBenchmarker Authors. All rights reserved.
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

"""Runs the vbench transcoding benchmark with h.264 and vp9.

Paper: http://www.cs.columbia.edu/~lottarini/test/data/vbench.pdf
Vbench suite download link: http://arcade.cs.columbia.edu/vbench/
"""

import json
import posixpath
import subprocess

from typing import Any, Dict, List
from absl import flags
import numpy as np
import pandas as pd
from perfkitbenchmarker import benchmark_spec
from perfkitbenchmarker import configs
from perfkitbenchmarker import flag_util
from perfkitbenchmarker import sample
from perfkitbenchmarker.linux_benchmarks import (
    vbench_transcoding_benchmark as pkb_vbench,
)
from perfkitbenchmarker import vm_util
from statistics import stdev
from ampere.pkb.common import download_utils
from ampere.pkb.linux_packages import ffmpeg
from ampere.pkb.utils import bash_template

FLAGS = flags.FLAGS

BENCHMARK_NAME = "ampere_vbench"
BENCHMARK_CONFIG = """
ampere_vbench:
  description: Runs a video transcoding benchmark.
  vm_groups:
    default:
      vm_spec:
        GCP:
          machine_type: n2d-highcpu-8
          zone: us-central1-f
        AWS:
          machine_type: c6g.2xlarge
          zone: us-east-1a
        Azure:
          machine_type: Standard_F8s
          zone: westus2
      disk_spec:
        # Standardize with 250 MB/s bandwidth.
        # The largest video file is ~300 MB; we want to minimize I/O impact.
        GCP:
          disk_size: 521
          disk_type: pd-ssd
          mount_point: /scratch
        AWS:
          disk_size: 521
          disk_type: gp2
          mount_point: /scratch
        Azure:
          disk_size: 2048
          disk_type: Premium_LRS
          mount_point: /scratch
      os_type: ubuntu2004
"""


BENCHMARK_DATA = {
    # Download from http://arcade.cs.columbia.edu/vbench/
    "vbench.zip": "c34b873a18b151322483ca460fcf9ed6a5dbbc2bb74934c57927b88ee1de3472"
}

CODEC_H264 = "h264"
CODEC_VP9 = "vp9"
DEFAULT_H264_THREADS_LIST = [4, 8]
DEFAULT_VP9_THREADS_LIST = [1]

flags.DEFINE_list(
    f"{BENCHMARK_NAME}_ffmpeg_codecs",
    [CODEC_H264],
    "List of the codecs to use for the transcoding benchmark. "
    "For now, this is some combination of h264 and vp9.",
)
flag_util.DEFINE_integerlist(
    f"{BENCHMARK_NAME}_ffmpeg_threads_list",
    None,
    "List of threads to give to each ffmpeg job. Defaults to "
    "[4, 8] for h.264 and [1] for vp9.",
)
flag_util.DEFINE_integerlist(
    f"{BENCHMARK_NAME}_ffmpeg_parallelism_list",
    None,
    "List of ffmpeg-jobs to run in parallel. Defaults to " "[number of logical CPUs].",
)
_FFMPEG_DIR = flags.DEFINE_string(
    f"{BENCHMARK_NAME}_ffmpeg_dir",
    "/usr/bin",
    "Directory where ffmpeg and ffprobe are located.",
)

_VALID_CODECS = [CODEC_H264, CODEC_VP9]
flags.register_validator(
    f"{BENCHMARK_NAME}_ffmpeg_codecs",
    lambda codecs: all([c in _VALID_CODECS for c in codecs]),
)

flags.DEFINE_string(
    f"{BENCHMARK_NAME}_vbench_path",
    None,
    "Defaults to GCS path where vbench.zip is located. "
    "Can be a GCS path e.g. gs://<bucket-name>/<file-name> "
    "Can be an http(s) url where the archive is located.",
)

flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_run_parallel",
    False,
    "Transcode all 15 input files in parallel.",
)

flags.DEFINE_bool(
    f"{BENCHMARK_NAME}_run_vod_2pass",
    False,
    "Transcode all 15 input files with 2 pass.",
)


def GetConfig(user_config: Dict[str, Any]) -> Dict[str, Any]:
    return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def Prepare(spec: benchmark_spec.BenchmarkSpec) -> None:
    """Install FFmpeg and download sample videos on the VM.

    Args:
    spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
    """
    vm = spec.vms[0]

    # Send zip archive of vbench input video files to VM from GCS bucket or http(s) link
    #   - download_utils helper functions download() and gsutil() send files to VM under
    #   - /opt/pkb-cache by default
    vbench_path = FLAGS[f"{BENCHMARK_NAME}_vbench_path"].value
    if vbench_path.startswith("http") or vbench_path.startswith("https"):
        download_utils.download(vbench_path, [vm])
    elif vbench_path.startswith("gs://"):
        download_utils.gsutil(vbench_path, [vm])
    else:
        raise ValueError(
            f"Unknown src format for {vbench_path}! Only HTTP(s) and gs:// are supported!"
        )

    # Unzip archive if hasn't been unzipped
    vm.InstallPackages("unzip")
    _, _, retcode = vm.RemoteCommandWithReturnCode(
        f"test -d {download_utils.CACHE_DIR}/vbench", ignore_failure=True
    )
    file_exists = retcode == 0
    if not file_exists:
        vm.RemoteCommand(
            f"sudo unzip {download_utils.CACHE_DIR}/vbench.zip -d {download_utils.CACHE_DIR}"
        )
    # Catch cases where scratch disk doesn't exist (Baremetal) and create directory
    vm.RemoteCommand("sudo mkdir -p /scratch")
    # Change ownership of scratch disk so PKB can access vbench files
    vm.RemoteCommand("sudo chown -R $USER:$USER /scratch")
    vm.RemoteCommand(f"sudo cp -R {download_utils.CACHE_DIR}/vbench /scratch/")

    # Install ffmpeg and other utils
    vm.Install(ffmpeg.PACKAGE_NAME)
    vm.InstallPackages("parallel")
    vm.InstallPackages("time")
    vm.Install("numactl")


def Run(spec: benchmark_spec.BenchmarkSpec) -> List[sample.Sample]:
    # Run with GNU parallel if desired (from upstream PKB)
    parallel_run = FLAGS[f"{BENCHMARK_NAME}_run_parallel"].value
    return pkb_vbench.RunParallel(spec) if parallel_run else RunPerCore(spec)


def RunPerCore(spec: benchmark_spec.BenchmarkSpec) -> List[sample.Sample]:
    vm = spec.vms[0]
    input_videos_dir = "/scratch/vbench/videos/crf0"
    vod_run = FLAGS[f"{BENCHMARK_NAME}_run_vod_2pass"].value
    if vod_run:
        input_videos_dir = "/scratch/vbench/videos/crf18"
    # Create new folder in home directory to store logs, time files
    log_dir = "/scratch/vbench-logs"
    vm.RemoteCommand(f"mkdir -p {log_dir}")

    # Check online CPUs for numa pinning
    lscpu = vm.CheckLsCpu()
    core_ranges = lscpu.data["On-line CPU(s) list"].split(",")
    bash_sequences = ""
    for core_range in core_ranges:
        core_start, core_end = core_range.split("-")
        bash_sequences += f"$(seq {core_start} {core_end}) "
    render_args = {
        "bash_sequences": bash_sequences,
        "input_videos_dir": input_videos_dir,
        "log_dir": log_dir,
        "ffmpeg_dir": _FFMPEG_DIR.value,
        "scratch_dir": "/scratch",
    }
    deploy_dir = "/scratch"
    template_name = "vod_2pass.sh.j2" if vod_run else "vbench_upload.sh.j2"
    deploy_path = bash_template.render_and_copy_to_vm(
        vm, deploy_dir, template_name, render_args
    )
    vm.RemoteCommand(deploy_path)

    # Copy all ffmpeg log and time files back to PKB system
    #   - /tmp/perfkitbenchmarker/<run_uri>/vbench-logs/<video_name>_<core>.log
    vm.RemoteCommand(f"tar -czf {log_dir}.tar.gz {log_dir}")
    vm.PullFile(vm_util.GetTempDir(), f"{log_dir}.tar.gz")
    local_log_tar = posixpath.join(vm_util.GetTempDir(), "vbench-logs.tar.gz")
    return _parse_results(vm, lscpu, local_log_tar)


def _run_subprocess(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode("utf-8"), stderr.decode("utf-8")
    if stderr:
        raise ValueError(stderr)
    else:
        return stdout


def _prepare_local_log_dir(local_log_tar):
    untar_cmd = f"tar -xf {local_log_tar} -C {vm_util.GetTempDir()}"
    _run_subprocess(untar_cmd)
    local_log_dir = posixpath.join(vm_util.GetTempDir(), "scratch/vbench-logs")
    return local_log_dir


def _get_frames_dict(vm, input_videos_dir):
    """Takes input_videos_dir
    Returns: dictionary where key=input_video, val=frame_count
    """
    file_names, _ = vm.RemoteCommand(
        f"ls {input_videos_dir}  > filename.txt; cat filename.txt;"
    )
    input_videos = file_names.strip().split()
    frames_dict = {}
    for input_video in input_videos:
        frames, _ = vm.RemoteCommand(
            f" ~/bin/ffprobe  -v error -select_streams v:0 -count_packets "
            f"-show_entries stream=nb_read_packets -of csv=p=0 {input_videos_dir}/{input_video}"
        )
        frame_count = frames.strip()
        input_video_key = input_video.replace(".mkv", "")
        frames_dict[input_video_key] = float(frame_count)
    return frames_dict


def _get_transcode_time(local_log_dir, input_video_name, core):
    """By using wildcard match after the input_video_name
    awk can aggregate transcode times across
        - 1 file for 1-pass encoding (upload) or
        - 2 files for 2-pass encoding (VoD)
    """
    parse_cmd = (
        f"awk '{{sum+=$1; sum+=$2}} END {{print sum}}' "
        f"{local_log_dir}/{input_video_name}*_{core}.time"
    )
    return float(_run_subprocess(parse_cmd))


# Create dictionary for transcode time
def _create_transcode_time_dict(input_videos_name_list, local_log_dir, lscpu):
    transcode_time_dict = {}
    core_ranges = lscpu.data["On-line CPU(s) list"].split(",")
    for input_video in input_videos_name_list:
        transcode_time_dict[input_video] = []
        for core_range in core_ranges:
            core_start, core_end = [int(core) for core in core_range.split("-")]
            for core in range(core_start, core_end + 1):
                transcode_time = _get_transcode_time(local_log_dir, input_video, core)
                transcode_time_dict[input_video].append(transcode_time)
    return transcode_time_dict


def _gmean(value_list):
    return np.exp(np.log(value_list).mean()).round(2)


def _parse_results(vm, lscpu, local_log_tar):
    ### Prepare data
    vod_run = FLAGS[f"{BENCHMARK_NAME}_run_vod_2pass"].value
    dir_num = "18" if vod_run else "0"
    input_videos_dir = f"/scratch/vbench/videos/crf{dir_num}"
    frames_dict = _get_frames_dict(vm, input_videos_dir)
    input_videos_name_list = frames_dict.keys()

    local_log_dir = _prepare_local_log_dir(local_log_tar)

    transcode_time_dict = _create_transcode_time_dict(
        input_videos_name_list, local_log_dir, lscpu
    )

    ### Perform calculations
    # generate data frame from dictionary for easier parsing
    df = pd.DataFrame(transcode_time_dict)

    # save sum of transcode times per core
    per_core_transcode_times = df.sum(axis=1)
    per_core_results = per_core_transcode_times.round(2).to_dict()

    # save transcode time geomeans
    #   - per file (column)
    #   - per core (row, transposed data frame)
    gmean_transcode_times_per_file = pd.Series(_gmean(df), index=df.columns)
    gmean_transcode_times_per_core = pd.Series(_gmean(df.T), index=df.index.to_list())
    # get geomean of geomean
    gmean_of_gmean = float(_gmean(gmean_transcode_times_per_core).round(2))

    # convert df of transcode times to fps values, save to dict
    # WARN: this overwrites the existing df
    for input_video in df.columns:
        df[input_video] = frames_dict[input_video] / df[input_video]
    fps_dict = df.round(2).to_dict()

    # save fps geomeans
    #   - per file (column)
    #   - per core (row, transposed data frame)
    gmean_fps_per_file = pd.Series(_gmean(df), index=df.columns)
    gmean_fps_per_core = pd.Series(_gmean(df.T), index=df.index.to_list())

    ### Assemble metadata and create samples
    profile = "vod" if vod_run else "upload"
    metadata = _assemble_metadata(vm, profile, per_core_results)
    geomean_metadata = {
        "Geomean Transcode Times per File": json.dumps(
            gmean_transcode_times_per_file.round(2).to_dict()
        ),
        "Geomean Transcode Times per Core": json.dumps(
            gmean_transcode_times_per_core.round(2).to_dict()
        ),
        "Geomean FPS per File": json.dumps(gmean_fps_per_file.round(2).to_dict()),
        "Geomean FPS per Core": json.dumps(gmean_fps_per_core.round(2).to_dict()),
    }
    samples = _create_samples(gmean_of_gmean, metadata, geomean_metadata)
    return samples


def _get_package_versions(vm, log_dir):
    """Returns a dictionary with:
    -  ffmpeg version
    -  x264 version
    -  gcc version (used by ffmpeg)
    """
    # Get ffmpeg version
    stdout, _ = vm.RemoteCommand(f"{_FFMPEG_DIR.value}/ffmpeg -version | head -1")
    # Remove extra data from version
    ffmpeg_version = stdout.strip("ffmpeg version").split("Copyright")[0].strip()

    # Get gcc version used by ffmpeg
    stdout, _ = vm.RemoteCommand(
        f"{_FFMPEG_DIR.value}/ffmpeg -version | grep gcc | head -1"
    )
    ffmpeg_gcc_version = (
        stdout.strip()
    )  # Remove leading/trailing spaces and newline chars

    # Get x264 version from log
    example_log = "bike_1280x720_29_0.log"
    grep_string = "H.264/MPEG-4 AVC codec"
    stdout, _ = vm.RemoteCommand(
        f"cat ~/{log_dir}/{example_log} | "
        f"grep '{grep_string}' | "
        f"awk -F'-' '{{print $2}}'"
    )
    x264_version = stdout.strip()  # Remove leading/trailing spaces and newline chars

    all_versions = {
        "ffmpeg_version": ffmpeg_version,
        "ffmpeg_gcc_version": ffmpeg_gcc_version,
        "x264_version": x264_version,
    }

    return all_versions


def _assemble_metadata(vm, profile, per_core_results):
    metadata = {
        "profile": profile,
        "codec": "h264",  # TODO: make this dynamic?
        "num_files": 15,
        "parallelism": None,
        "threads": 1,  # hardcoded single-threaded for per core run
        "ffmpeg_compiled_from_source": True,
        "video_copies": 1,
        "per_core_results": json.dumps(per_core_results),
    }

    # Get all package versions (ffmeg, x264, gcc for ffmpeg) and add to metadata
    remote_log_dir = "/scratch/vbench-logs"
    pkg_versions = _get_package_versions(vm, remote_log_dir)
    for pkg, ver in pkg_versions.items():
        metadata[pkg] = ver

    # Get max, min, avg, and stdev of per core runtimes
    # Report longest runtime as high level result
    len_results = len(per_core_results.values())
    max_core, max_time = max(per_core_results.items(), key=lambda k: k[1])
    min_time = min(per_core_results.values())
    avg_time = sum(per_core_results.values()) / len_results
    sum_time = sum(per_core_results.values())
    avg_transcoding_vod = 15 * len_results / sum_time
    stdev_all_cores = stdev(per_core_results.values())

    # Add all calculations to metadata
    metadata["max_time_all_cores"] = max_time
    metadata["min_time_all_cores"] = min_time
    metadata["avg_time_all_cores"] = avg_time
    metadata["stdev_all_cores"] = stdev_all_cores
    return metadata


def _create_samples(gmean_of_gmean, metadata, geomean_metadata):
    # Report average transcode time as high-level metric for both Upload and VoD
    # Report all calculations in metadata
    results = []
    geomean_sample = sample.Sample(
        metric="Geomean of Geomean of Transcode Time",
        value=gmean_of_gmean,
        unit="seconds",
        metadata=metadata | geomean_metadata,
    )
    results.append(geomean_sample)

    avg_time = metadata["avg_time_all_cores"]
    avg_transcode_time_sample = sample.Sample(
        metric="Avg Transcode Time", value=avg_time, unit="seconds", metadata=metadata
    )
    results.append(avg_transcode_time_sample)
    return results


def Cleanup(spec: benchmark_spec.BenchmarkSpec) -> None:
    vm = spec.vms[0]
    vm.RemoteCommand(f"sudo rm -rf {download_utils.CACHE_DIR}/vbench")
    # cleanup logs, time, and out files
    vm.RemoteCommand("sudo rm -rf /scratch/*")
    del spec  # Unused
