#!/bin/bash

# Create output directory
mkdir -p ./Data

# File ID from your Google Drive link
FILE_ID="1E-VjTC5oByg6-LUYVcGYUTjVKxPJDjaZ"
DEST="./Data/OffsetOPT_data.zip"

# Download using gdown
# pip install -q gdown
gdown --id $FILE_ID -O $DEST

# Unzip the file
unzip -o $DEST -d ./Data

# Optional: Remove the zip after extraction
rm $DEST
