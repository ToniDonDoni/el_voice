#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p workspace models data work
if [ ! -e .env ]; then
  umask 077
  {
    printf 'EL_VOICE_PORT=8765\n'
    printf 'LOCAL_UID=%s\n' "$(id -u)"
    printf 'LOCAL_GID=%s\n' "$(id -g)"
  } > .env
fi
printf 'Workspace directories and local configuration are ready.\n'
