# Run Multiload with Thread Scaling to Study Latencies

## Config

Update `./ampere/pkb/configs/mulitload_thread_scaling.yml` with SUT information to run multiload with `memcpy-libc` and thread scaling up to 192 cores, e.g.

## Run

Invoke APT with the following command-line and YAML config

```bash
./pkb.py --benchmarks=ampere_multichase_benchmark --benchmark_config_file=./ampere/pkb/configs/multiload_thread_scaling.yml
```

## Parse 

Use `csv_multiload.sh` to parse APT's JSON results into CSV format

```bash
./ampere/pkb/scripts/multiload_csv.sh /tmp/perfkitbenchmarker/runs/<run_uri>/perfkitbenchmarker_results.json
```

Example output

```bash
LoadThds,ChaseNS,ChaseMibs,LdMaxMibs,Total Load
1,144.948,53.0,0.0,53.0
4,164.365,46.0,49509.0,49555.0
8,178.914,43.0,103312.0,103355.0
12,196.715,39.0,151097.0,151136.0
16,218.71,35.0,194446.0,194481.0
...
```

Plot `Total Load` on the x-axis with `ChaseNS` on the y-axis to see a latency curve as `LoadThds` scales.
