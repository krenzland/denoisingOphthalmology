#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "$#" -ne 2 ]; then
    echo "1st Parameter = Raw-Image-Directory, 2nd Parameter = Target-Directory"
    exit 1
fi

raw_dir="$1"
target_dir="$2"

echo "$1"
find ${raw_dir} \( -name "*.jpeg" -or -name "*.ppm" \) -type f | while read image; do
    raw_path="$(readlink -f $image)"
    image_name="$(basename ${raw_path})"
    target_name="$(readlink -f ${target_dir}/${image_name})"
    # TODO: Find better value for fuzz
    convert "${raw_path}" -fuzz 20% -bordercolor black -trim +repage "${target_name}"
done

# Fuzz 20% is a bit too eager for very dark images
# Discard them, they aren't that great anyway.
find "$target_dir" -size -100 -delete 
