#!/bin/bash
# Copyright (c) 2024-2025, Ampere Computing LLC
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

if [ "$#" -eq 0 ]; then
	echo "Usage: provide a path to an APKB results JSON file from a multiload thread scaling run."
	echo -e "\te.g. ./multiload_csv.sh ./perfkitbenchmarker_results.json"
	exit 1
fi

RESULTS=$1

# Get all thread values used
# e.g. 1 4 8 16 32 ...
all_threads=$(grep -i "chasens" ${RESULTS} | awk '{print $2}' | awk -F"_" '{print $3}' | sed 's/",//g' | xargs)

# Print header
echo "LoadThds,ChaseNS,ChaseMibs,LdMaxMibs,Total Load"

# Print relevant data points for each thread value
for thread in $all_threads; do
	chasens=$(grep -i "chasens_threads_${thread}\"" $RESULTS | awk '{print $4}' | sed 's/,//g')
	chasemibs=$(grep -i "chasemibs_threads_${thread}\"" $RESULTS | awk '{print $4}' | sed 's/,//g')
	ldmaxmibs=$(grep -i "ldmaxmibs_threads_${thread}\"" $RESULTS | awk '{print $4}' | sed 's/,//g')
	totalload=$(echo "${chasemibs} + ${ldmaxmibs}" | bc)
	echo "${thread},${chasens},${chasemibs},${ldmaxmibs},${totalload}"
done

