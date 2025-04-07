# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

