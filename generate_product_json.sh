#!/usr/bin/env bash
# generate_product_json.sh
# Usage: bash generate_product_json.sh <image_dir> <output_product_json>
#
# Scans <image_dir> for *.png files and embeds them as base64 images inside
# the brainlife product.json format.
set -euo pipefail

image_dir="${1:?Usage: $0 <image_dir> <product_json>}"
product_json="${2:?Usage: $0 <image_dir> <product_json>}"

qa_entries=()

while IFS= read -r image; do
    base=$(basename "$image" .png)
    entry=$(printf '{"type":"image/png","name":"%s","base64":"%s"}' \
        "$base" \
        "$(base64 -w 0 "$image")")
    qa_entries+=("$entry")
done < <(find "${image_dir}" -name "*.png" | sort)

if [ ${#qa_entries[@]} -eq 0 ]; then
    brainlife_json='{"type":"error","msg":"No QC images were generated."}'
    brainlife_array="[${brainlife_json}]"
else
    brainlife_array="[$(printf '%s,' "${qa_entries[@]}" | sed 's/,$//')]"
fi

# Embed metrics if present
metrics_json="null"
metrics_file="${image_dir}/metrics.json"
if [ -f "$metrics_file" ]; then
    metrics_json=$(cat "$metrics_file")
fi

cat > "${product_json}" << EOF
{
  "datatype_tags": [],
  "brainlife": ${brainlife_array},
  "metrics": ${metrics_json}
}
EOF

echo "product.json written to ${product_json}"
