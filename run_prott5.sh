#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --job-name=prott5_job

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <input_fasta> <output_dir> <prott5_model_id> <dtype> [shard_accession]" >&2
  echo "  dtype: computation dtype passed through to the container (e.g., bf16 or fp32)." >&2
  echo "  shard_accession: optional flag (true/1/yes) to enable accession-based sharding." >&2
  exit 1
fi

input_fasta=$(realpath "$1")
output_dir=$(realpath "$2")
prott5_model_id=$3
dtype=$4
shard_accession=${5:-}

echo "$output_dir"

prott5_models=/shared/archive/cas/common-data/rostlab-models

shard_flag=""
if [[ "${shard_accession}" =~ ^(true|1|yes)$ ]]; then
  shard_flag="--shard-accession"
fi

docker run --user $(id -u):$(id -g) \
           -v "${input_fasta}":/workspace/input-fasta.fa \
           -v "${output_dir}":/workspace/output_dir \
           -v "${prott5_models}":/rostlab-models \
           -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
           --rm --gpus all \
           docker.beverara.biocomp.unibo.it:5000/rostlab --model-dir "/rostlab-models/${prott5_model_id}" --out-dir /workspace/output_dir --save-per-residue --pooling none --dtype "${dtype}" ${shard_flag} /workspace/input-fasta.fa
