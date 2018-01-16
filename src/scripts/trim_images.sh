#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "$#" -ne 2 ]; then
    echo "1st Parameter = Raw-Image-Directory, 2nd Parameter = Target-Directory"
    exit 1
fi

raw_dir="$1"
target_dir="$2"
extension='tif'

echo "$1"
find ${raw_dir} \( -name "*.jpeg" -or -name "*.ppm" \) -type f | while read image; do
    raw_path="$(readlink -f $image)"
    image_name="$(basename ${raw_path} | cut -f 1 -d '.')" # Remove extension.
    target_name="$(readlink -f ${target_dir}/${image_name}.${extension})"
    # TODO: Find better value for fuzz
    convert "${raw_path}" -fuzz 20% -bordercolor black -trim +repage -resize 1024x1024 "${target_name}"
done

# Fuzz 20% is a bit too eager for very dark images
# Discard them, they aren't that great anyway.
find "$target_dir" type f -size -100k -delete 
