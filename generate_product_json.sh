#!/usr/bin/env bash
# generate_product_json.sh
#
# Usage:
#   bash generate_product_json.sh [--compress] <image_dir> <output_product_json>
#
# Behavior:
#   - Default: embeds PNGs as full base64
#   - --compress: tries to keep final product.json under 1 MB by
#       resizing PNGs proportionally based on image count, then
#       iteratively shrinking further if needed.
#
# Requirements for --compress:
#   - ImageMagick ("convert" or "magick")
#
set -euo pipefail

MAX_JSON_SIZE=$((1024 * 1024))   # 1 MB
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

compress=false

usage() {
    cat <<EOF
Usage: $0 [--compress] <image_dir> <output_product_json>

Options:
  --compress   Resize images before base64 embedding so product.json
               tries to stay under 1 MB.
EOF
}

# ----------------------------
# Parse args
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compress)
            compress=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

image_dir="${1:?Usage: $0 [--compress] <image_dir> <product_json>}"
product_json="${2:?Usage: $0 [--compress] <image_dir> <product_json>}"

metrics_json="null"
metrics_file="${image_dir}/metrics.json"
if [[ -f "$metrics_file" ]]; then
    metrics_json="$(cat "$metrics_file")"
fi

mapfile -d '' images < <(find "${image_dir}" -type f -name "*.png" -print0 | sort -z)
num_images="${#images[@]}"

if [[ "$num_images" -eq 0 ]]; then
    cat > "${product_json}" <<EOF
{
  "datatype_tags": [],
  "brainlife": [{"type":"error","msg":"No QC images were generated."}],
  "metrics": ${metrics_json}
}
EOF
    echo "product.json written to ${product_json}"
    exit 0
fi

# ----------------------------
# Choose image resize command
# ----------------------------
IMG_CONTAINER="${IMG_CONTAINER:-docker://brainlife/imagemagick:latest}"

img_resize() {
    local input="$1"
    local output="$2"
    local percent="$3"

    if command -v apptainer >/dev/null 2>&1; then
        apptainer exec "$IMG_CONTAINER" convert "$input" -resize "${percent}%" "$output"
    elif command -v singularity >/dev/null 2>&1; then
        singularity exec "$IMG_CONTAINER" convert "$input" -resize "${percent}%" "$output"
    else
        echo "Error: --compress requires apptainer or singularity." >&2
        exit 1
    fi
}
# ----------------------------
# Build product.json at a given scale
# ----------------------------
build_product_json() {
    local scale_percent="$1"
    local out_json="$2"

    local qa_entries=()
    local total_base64_chars=0
    local idx=0

    for image in "${images[@]}"; do
        local base
        base="$(basename "$image" .png)"

        local src_image="$image"
        local working_image="$image"

        if [[ "$compress" == true && "$scale_percent" -lt 100 ]]; then
            working_image="${TMPDIR}/img_${idx}.png"
            img_resize "$src_image" "$working_image" "$scale_percent"
        fi

        local b64
        b64="$(base64 -w 0 "$working_image")"
        total_base64_chars=$((total_base64_chars + ${#b64}))

        local entry
        entry=$(printf '{"type":"image/png","name":"%s","base64":"%s"}' \
            "$base" "$b64")
        qa_entries+=("$entry")

        idx=$((idx + 1))
    done

    local brainlife_array
    brainlife_array="[$(printf '%s,' "${qa_entries[@]}" | sed 's/,$//')]"

    cat > "${out_json}" <<EOF
{
  "datatype_tags": [],
  "brainlife": ${brainlife_array},
  "metrics": ${metrics_json}
}
EOF
}

# ----------------------------
# Compression strategy
# ----------------------------
if [[ "$compress" == false ]]; then
    build_product_json 100 "${product_json}"
    final_size=$(wc -c < "${product_json}")
    echo "product.json written to ${product_json} (${final_size} bytes)"
    exit 0
fi

# Heuristic:
# Reserve some space for JSON/metrics overhead, then estimate a first-pass
# per-image share of the remaining budget. Since base64 expands binary data
# by ~4/3, we estimate a target compressed PNG size, then derive a resize %
# from the average original byte size. This is only an estimate, so we follow
# with iterative retries.
reserved_overhead=16384
available_for_images=$((MAX_JSON_SIZE - reserved_overhead))
if [[ "$available_for_images" -le 0 ]]; then
    echo "Error: overhead budget exceeds max size." >&2
    exit 1
fi

target_base64_per_image=$((available_for_images / num_images))
target_binary_per_image=$((target_base64_per_image * 3 / 4))

total_original_size=0
for image in "${images[@]}"; do
    sz=$(wc -c < "$image")
    total_original_size=$((total_original_size + sz))
done
avg_original_size=$((total_original_size / num_images))

# Initial guess
scale_percent=100
if [[ "$avg_original_size" -gt 0 && "$target_binary_per_image" -lt "$avg_original_size" ]]; then
    # area roughly scales with bytes, so dimension percent is sqrt(byte_ratio)
    scale_percent=$(python3 - <<PY
import math
avg_size = $avg_original_size
target = $target_binary_per_image
ratio = max(target / avg_size, 0.01)
pct = int(max(5, min(100, math.sqrt(ratio) * 100)))
print(pct)
PY
)
fi

candidate_json="${TMPDIR}/product.json"
attempt=1
best_size=0
best_scale=0

while :; do
    rm -f "${TMPDIR}"/img_*.png "$candidate_json" 2>/dev/null || true
    build_product_json "$scale_percent" "$candidate_json"
    current_size=$(wc -c < "$candidate_json")

    best_size="$current_size"
    best_scale="$scale_percent"

    if [[ "$current_size" -le "$MAX_JSON_SIZE" ]]; then
        cp "$candidate_json" "$product_json"
        echo "product.json written to ${product_json} (${current_size} bytes, scale=${scale_percent}%)"
        exit 0
    fi

    if [[ "$scale_percent" -le 5 ]]; then
        cp "$candidate_json" "$product_json"
        echo "Warning: even at very small size, product.json is still above 1 MB." >&2
        echo "product.json written to ${product_json} (${current_size} bytes, scale=${scale_percent}%)"
        exit 0
    fi

    # Compute next scale based on overflow ratio, then add a small safety cut.
    next_scale=$(python3 - <<PY
import math
current_size = $current_size
max_size = $MAX_JSON_SIZE
current_scale = $scale_percent

ratio = max_size / current_size
# size ~ area, so dimensions scale with sqrt(ratio)
next_scale = math.sqrt(ratio) * current_scale
# safety margin to avoid hovering around the threshold
next_scale *= 0.92
next_scale = int(max(5, min(current_scale - 1, next_scale)))
print(next_scale)
PY
)

    if [[ "$next_scale" -ge "$scale_percent" ]]; then
        next_scale=$((scale_percent - 5))
    fi
    if [[ "$next_scale" -lt 5 ]]; then
        next_scale=5
    fi

    scale_percent="$next_scale"
    attempt=$((attempt + 1))

    if [[ "$attempt" -gt 12 ]]; then
        cp "$candidate_json" "$product_json"
        echo "Warning: reached max compression attempts." >&2
        echo "product.json written to ${product_json} (${best_size} bytes, scale=${best_scale}%)"
        exit 0
    fi
done
