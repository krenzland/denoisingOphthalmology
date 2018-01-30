#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "$#" -ne 2 ]; then
    echo "1st Parameter = Raw-Image-Directory, 2nd Parameter = Target-Directory"
    exit 1
fi

raw_dir="$1"
target_dir="$2"
extension='jpg'

echo "$1"
# Optimal for Messidor!
find ${raw_dir} \( -name "*.jpeg" -or -name "*.tif" \) -type f | parallel "convert '{}' -gravity Center -crop 67x100%+0+0 -bordercolor black -fuzz 10% -trim +repage "${target_dir}/'$(basename '{}')'""

# Fuzz 5% is a bit too eager for very dark images
# Discard them, they aren't that great anyway.
find "$target_dir" -type f -size -100k -delete 
