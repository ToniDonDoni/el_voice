#!/bin/sh
# Record the deployed commit for the on-screen badge. The repository root is not
# mounted into the voice container, so the app reads this file from /data.
set -eu
cd "$(dirname "$0")/.."
commit=$(git rev-parse --short=8 HEAD)
printf '%s\n' "$commit" > data/COMMIT
printf 'Recorded commit %s in data/COMMIT\n' "$commit"
