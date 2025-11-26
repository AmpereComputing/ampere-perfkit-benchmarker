# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-11-19

### Added
-   `pytorch_dlrm_benchmark`: new benchmark implementation supports deep learning recommendation model performance

## [2.1.1] - 2025-11-07

### Fixed
-   `llm_benchmark`: updated `-fa` flag to pass string in recent versions of llama-batched-bench
-   `docker`: improvements to container cleanup and installation on Oracle Linux

## [2.1.0] - 2025-10-03

### Added
-   `llm_benchmark`: new benchmark implementation supports inference benchmarking of large language models (LLaMA, Qwen, etc.) 
-   `docker`: new package implementation provides support for docker: installation, pulling an existing image, creating an image based on a Dockerfile, managing containers, etc.

## [2.0.1] - 2025-08-08

### Added

-   `bashrc_handler`: ensures modular code and replaces repeated functions throughout the namespace
-   `ampere_find_jdk`: captures jdk version in `*results.json`
-   `setup.sh`: new convenience script helps automate APKB setup for new users
-   `sysbench`: can now pin to specified core range
 
### Changed

-   Upstream PKB upstream removed a `wrk` file, this has been moved back to the ampere namespace and modified to report p95 latency
-   `download_utils`: changed to ensure the same utility functions are used to download files/resources (be it `wget`, `git clone` or from `gsutil`)
-   `provision_disk`: modified such that the upstream .yml files work with APKB in cases when the vm group specified is `default`
-   Changes related to unifying reported metrics and modes: 
    - Throughput modes are now standardized and always called as `{BENCHMARK_NAME}_throughput_mode` across all benchmarks
    - All outputs are now now standardized and called `p95_latency` or `<unit>_performance`
-   `vBench`: new performance metric (geomean of all geomeans) is used.
-   `nginx`: the nginx.conf is now generated from the namespace's code itself, this adds also adds support for brotli compression
-   Firewall changes to support the upcoming deprecation of `IPTABLES` package
    - Newer operating systems are moving to `nft-tables`, this is now supported by `firewall_handler`
-   `memtier`: builds a fixed version of `libevent` which is used across all OS types) 

### Fixed

-   `redis`: updated git repository URL and logic (since the repo has been migrated)
-   `multichase`: correctly installs `glibc-static` on RHEL-based systems

## [2.0.0] - 2025-04-07

### Added

-   Support for a suite of microtools: new workloads include Multiload and SPECint
-   Netperf and FIO from upstream PKB are now functional with the Ampere namespace 
-   New flag added to OCI provider `oci_enable_firewall`

### Changed

-   Rebased from upstream PKB, commit SHA: 3a8ae41e2c162829d628f9d59ce4aa2060e0b236
-   Python version bumped from 3.9 -> 3.11
-   New requirement `immutabledict` added from upstream PKB
-   `BaseOSMixin` definition moved from `virtual_machine.py` -> `os_mixin.py` 
-   Refactored OCI Security List Rules logic

### Fixed

-   Passing `false` to `oci_enable_firewall` allows for functional runs of `netperf` from upstream PKB on OCI 

## [1.0.0] - 2024-09-23

Initial release

