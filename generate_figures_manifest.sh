#!/usr/bin/env bash
set -euo pipefail

image_dir="$1"
output_dir="$2"

figures_dir="${output_dir}/figures"
images_dir="${figures_dir}/images"

mkdir -p "$images_dir"

IMG_CONTAINER="${IMG_CONTAINER:-docker://brainlife/imagemagick:latest}"

run_convert() {
    if command -v apptainer >/dev/null 2>&1; then
        apptainer exec "$IMG_CONTAINER" convert "$@"
    else
        singularity exec "$IMG_CONTAINER" convert "$@"
    fi
}

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

entries=()

for img in "$image_dir"/*.{png,jpg,jpeg}; do
    [[ -f "$img" ]] || continue

    base=$(basename "$img")
    name="${base%.*}"
    ext="${base##*.}"

    out="${images_dir}/${name}.png"

    if [[ "$ext" == "png" ]]; then
        cp "$img" "$out"
    else
        run_convert "$img" PNG:"$out"
    fi

    desc="QC figure"

    entries+=(
        "$(printf '{"filename":%s,"name":%s,"desc":%s}' \
        "$(json_escape "images/${name}.png")" \
        "$(json_escape "$name")" \
        "$(json_escape "$desc")")"
    )
done

images_json="["
for i in "${!entries[@]}"; do
    [[ "$i" -gt 0 ]] && images_json+=","
    images_json+="${entries[$i]}"
done
images_json+="]"

cat > "${figures_dir}/images.json" <<EOF
{
  "images": ${images_json}
}
EOF
