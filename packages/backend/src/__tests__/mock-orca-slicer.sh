#!/usr/bin/env bash
# Mock orca-slicer: creates expected GLB output file
d=""
prev=""
for a in "$@"; do
  [ "$prev" = "--outputdir" ] && d="$a"
  prev="$a"
done
mkdir -p "$d"
# Minimal valid GLB binary header (12 bytes) + empty JSON chunk
printf 'glTF\x02\x00\x00\x00\x20\x00\x00\x00\x0c\x00\x00\x00JSON{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[]}\x00\x00\x00\x00' > "$d/plate_1_toolpaths.glb"
# Also create a minimal PNG for the async render
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82' > "$d/preview.png"
exit 0
