#!/usr/bin/env bash
# Fetch the CHART-Infographics 2024 training set into the cache and verify it
# against the checksum recorded in data/chartinfo_runs/SOURCE.md.
# (Behind a slow proxy this can take hours; copying the file from another
# machine and running this script with --verify-only is fine.)
set -euo pipefail
DEST=~/.cache/real-chart-bench/chartinfo/CHARTINFO_2024_Train.zip
SHA=3352254c3a4afe5e456096d50cf0c70125ba7f7ac8a8c647a152538f366a89be
URL="https://www.dropbox.com/scl/fi/vy7gyemald8z6rohx9mxx/CHARTINFO_2024_Train.zip?rlkey=ki3q4bb02rzdpdbih17we63gm&dl=1"
mkdir -p "$(dirname "$DEST")"
if [ "${1:-}" != "--verify-only" ] && [ ! -f "$DEST" ]; then
  curl -L --progress-bar "$URL" -o "$DEST"
fi
echo "$SHA  $DEST" | sha256sum -c -
