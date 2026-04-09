#!/usr/bin/env bash
set -euo pipefail

docker pull quay.io/qiime2/amplicon:2025.4

docker run -it -v C:\Users\USER\OneDrive\Desktop\Antibiotics\DAV132\16S_data:/data quay.io/qiime2/amplicon:2025.4 bash

#!/bin/bash

# --- Configuration ---
# The directory where your FASTQ files are located inside the Docker container
INPUT_DIR="/data"
# The name of the manifest file to be created
MANIFEST_FILE="${INPUT_DIR}/manifest.tsv"

# Define the extensions for forward and reverse reads
FORWARD_EXT="_1.fastq"
REVERSE_EXT="_2.fastq"

echo "--- Starting Manifest File Generation ---"
echo "Output manifest file: ${MANIFEST_FILE}"
echo "Searching for files in: ${INPUT_DIR}"
echo "Expected forward extension: ${FORWARD_EXT}"
echo "Expected reverse extension: ${REVERSE_EXT}"
echo ""

# Create or overwrite the manifest file with the header
echo -e "sample-id\tforward-absolute-filepath\treverse-absolute-filepath" > "${MANIFEST_FILE}"

# Use a temporary file to keep track of accessions for which we found a pair
TEMP_PAIRED_ACCESSION_LIST=$(mktemp)

# Find all forward read files and iterate through them
find "${INPUT_DIR}" -maxdepth 1 -type f -name "*${FORWARD_EXT}" | sort | while read -r forward_filepath; do
    # Extract the base accession ID (e.g., SRR23799210 from /data/SRR23799210_1.fastq)
    # basename extracts SRR23799210_1.fastq, then sed removes _1.fastq
    accession_id=$(basename "$forward_filepath" | sed "s/${FORWARD_EXT}//")

    # Construct the expected path for the corresponding reverse file
    reverse_filepath="${INPUT_DIR}/${accession_id}${REVERSE_EXT}"

    # Check if the reverse file exists
    if [ -f "$reverse_filepath" ]; then
        # Both forward and reverse files exist, add to manifest
        echo -e "${accession_id}\t${forward_filepath}\t${reverse_filepath}" >> "${MANIFEST_FILE}"
        echo "  [FOUND PAIR] Added ${accession_id}"
        echo "${accession_id}" >> "${TEMP_PAIRED_ACCESSION_LIST}" # Mark as processed
    else
        # Forward file found, but reverse is missing
        echo "  [WARNING] Missing reverse file: '${reverse_filepath}' for forward file '${forward_filepath}'"
    fi
done

# This part checks if there are any reverse files that *don't* have a corresponding forward file
echo ""
echo "--- Checking for Orphaned Reverse Files ---"
find "${INPUT_DIR}" -maxdepth 1 -type f -name "*${REVERSE_EXT}" | sort | while read -r reverse_filepath; do
    # Extract the base accession ID
    accession_id=$(basename "$reverse_filepath" | sed "s/${REVERSE_EXT}//")

    # Check if this accession ID was NOT found as a pair in the first loop
    if ! grep -q "^${accession_id}$" "${TEMP_PAIRED_ACCESSION_LIST}"; then
        echo "  [WARNING] Missing forward file: '${INPUT_DIR}/${accession_id}${FORWARD_EXT}' for reverse file '${reverse_filepath}'"
    fi
done

# Clean up the temporary file
rm -f "${TEMP_PAIRED_ACCESSION_LIST}"

echo ""
echo "--- Manifest File Generation Complete ---"
# Count lines in the manifest file to see how many samples were added
NUM_SAMPLES=$(wc -l < "${MANIFEST_FILE}")
# Subtract 1 for the header line
if [ "$NUM_SAMPLES" -gt 1 ]; then
    echo "Successfully added $((NUM_SAMPLES - 1)) sample pairs to ${MANIFEST_FILE}."
else
    echo "WARNING: No sample pairs were added to ${MANIFEST_FILE}. Please check your file naming and the script's 'FORWARD_EXT' and 'REVERSE_EXT' variables."
fi
echo ""
echo "You can inspect the content of your manifest file using: cat ${MANIFEST_FILE}"
echo "Then proceed with 'qiime tools import ...' as per the next steps."

# Run in Docker terminal
sed -i 's/\r$//' generate_manifest.sh
./generate_manifest.sh

qiime tools import \
  --type 'SampleData[PairedEndSequencesWithQuality]' \
  --input-path manifest.tsv \
  --output-path demux-paired-end.qza \
  --input-format PairedEndFastqManifestPhred33V2

qiime demux summarize \
  --i-data demux-paired-end.qza \
  --o-visualization demux-paired-end.qzv

qiime dada2 denoise-paired \
  --i-demultiplexed-seqs demux-paired-end.qza \
  --p-trunc-len-f 232 \
  --p-trunc-len-r 234 \
  --p-trim-left-f 0 \
  --p-trim-left-r 0 \
  --o-table table.qza \
  --o-representative-sequences rep-seqs.qza \
  --o-denoising-stats denoising-stats.qza \
  --p-n-threads 0

qiime metadata tabulate \
  --m-input-file denoising-stats.qza \
  --o-visualization denoising-stats.qzv

qiime feature-classifier classify-sklearn \
  --i-classifier /data/2024.09.backbone.v4.nb.sklearn-1.4.2.qza \
  --i-reads rep-seqs.qza \
  --o-classification taxonomy.qza

qiime alignment mafft \
  --i-sequences rep-seqs.qza \
  --o-alignment aligned-rep-seqs.qza

qiime alignment mask \
  --i-alignment aligned-rep-seqs.qza \
  --o-masked-alignment masked-aligned-rep-seqs.qza

qiime phylogeny fasttree \
  --i-alignment masked-aligned-rep-seqs.qza \
  --o-tree unrooted-tree.qza

qiime phylogeny midpoint-root \
  --i-tree unrooted-tree.qza \
  --o-rooted-tree rooted-tree.qza

qiime tools export \
  --input-path table.qza \
  --output-path exported_feature_table

qiime tools export \
  --input-path taxonomy.qza \
  --output-path exported_taxonomy

qiime tools export \
  --input-path rooted-tree.qza \      
  --output-path exported-tree

cd cd exported_feature_table
biom convert -i feature-table.biom -o feature-table.tsv --to-tsv

#################################################
#qiime rescript filter-seqs-length \
#  --i-sequences rep-seqs.qza \
#  --p-global-min 370 \
#  --p-global-max 430 \
#  --o-filtered-seqs  rep-seqs-370-430.qza \
#  --o-discarded-seqs rep-seqs-outside-range.qza
#################################################
