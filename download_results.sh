#!/bin/bash

# Create output directory
mkdir -p ./results

# File ID from your Google Drive link
FILE_ID="1Db92EcCs1cp9Q5BiQZUfmWwNWS7IwDjD" 
DEST="./results/OffsetOPT_results.zip"

# Download using gdown
# pip install -q gdown
gdown --id $FILE_ID -O $DEST

# Unzip the file
unzip -o $DEST -d ./results

# Optional: Remove the zip after extraction
rm $DEST
