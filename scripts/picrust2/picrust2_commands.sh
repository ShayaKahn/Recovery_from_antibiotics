#!/usr/bin/env bash
set -euo pipefail

# Create docker image
docker pull quay.io/biocontainers/picrust2:2.6.2--pyhdfd78af_1

# Create docker container
docker run -it --rm -v "C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results:/work" -w /work quay.io/biocontainers/picrust2:2.6.2--pyhdfd78af_1 bash

# take subset of the samples
biom subset-table \
  -i feature-table.biom \
  -a sample \
  -s filtered_sample_ids.txt \
  -o filtered_feature-table.biom

# summaraize results
biom summarize-table -i filtered_feature-table.biom

# Convert BIOM to TSV
biom convert \
  -i feature-table.biom \
  -o feature-table.tsv \
  --to-tsv

# Compute number of samples and prevalence threshold
NUM_SAMPLES=$(sed -n '2p' feature-table.tsv | awk '{print NF - 1}')
PREVALENCE=$(python3 -c "import math; print(max(1, math.ceil(0.001 * $NUM_SAMPLES)))")

# Step 3: Filter ASVs by abundance and prevalence
tail -n +3 feature-table.tsv | awk -v min_abun=10 -v min_prev=$PREVALENCE '
{
    sum = 0; count = 0;
    for (i = 2; i <= NF; i++) {
        val = $i + 0;
        sum += val;
        if (val > 0) count++;
    }
    if (sum >= min_abun && count >= min_prev) print $1;
}' > asv_ids.txt

# Subset original BIOM using selected ASV IDs
biom subset-table \
  -i feature-table.biom \
  -a observation \
  -s asv_ids.txt \
  -o filtered-feature-table.biom

# Filter the study_seqs correspondingly
awk '
BEGIN {
  while ((getline k < "asv_ids.txt") > 0) keep[k]=1
}
# On a header: grab ID up to first whitespace; decide whether to print this record
/^>/ {
  split($0, a, /[ \t]/); id = substr(a[1], 2)   # remove leading ">"
  printrec = (id in keep)
}
# Print lines for selected records
{ if (printrec) print }
' study_seqs.fasta > filtered-study_seqs.fasta

# Run the picrust2 pipeline on the filtered data with the stratified flag
picrust2_pipeline.py -s filtered-study_seqs.fasta -i filtered-feature-table.biom -o picrust2_out_pipeline_strat -p 4 --stratified --verbose
