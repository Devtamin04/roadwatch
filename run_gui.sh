#!/usr/bin/env bash
# Launch the RoadWatch GUI. Optional argument: a video file to start immediately.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m roadwatch.gui "$@"
