#!/bin/bash
set -e
cd "$(dirname "$0")"

# One-shot launcher for Demo 1b (webcam -> RTSP -> Dragonwing device -> live overlay):
#   1. start the RTSP publisher (mediamtx + ffmpeg, see demo1b_publish_webcam.sh)
#   2. point the device's pipeline at it (demo1b_set_rtsp_source.py)
#   3. bring up the local server for demo1b_webcam.html
#   4. open the page in your browser
#
# Ctrl+C stops everything (RTSP publisher, local server) together.
#
# Setup (see the README):
#   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
#   export EYEPOP_PIPELINE_ID=<pipeline id printed by `eyepop instance init`>
#
# Usage: ./demo1b_start.sh [camera_index]
#   camera_index defaults to 0. List devices with:
#     ffmpeg -f avfoundation -list_devices true -i ""

CAMERA_INDEX="${1:-0}"
HTTP_PORT=8935
DEVICE_URL="${EYEPOP_DEVICE_URL:-http://localhost:8080}"

if [ -z "${EYEPOP_PIPELINE_ID:-}" ]; then
  echo "Set EYEPOP_PIPELINE_ID (printed by 'eyepop instance init'). See the README." >&2
  exit 1
fi

cleanup(){
  kill "$MEDIAMTX_PID" "$FFMPEG_PID" "$HTTP_PID" 2>/dev/null
}
trap cleanup EXIT

echo "==> Starting RTSP publisher (mediamtx + ffmpeg, camera index ${CAMERA_INDEX})"
mediamtx > /tmp/demo1b_mediamtx.log 2>&1 &
MEDIAMTX_PID=$!
sleep 1

# -video_size is required: left unset, avfoundation picks this camera's max
# resolution, which it may not sustain at 30fps, causing the device to see
# stale/duplicated frames. 1280x720 is within this camera's real capability.
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "${CAMERA_INDEX}:none" \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/stream \
  > /tmp/demo1b_ffmpeg.log 2>&1 &
FFMPEG_PID=$!
sleep 2

echo "==> Pointing the device at this computer's RTSP stream"
python3 demo1b_set_rtsp_source.py

echo "==> Starting local server on :${HTTP_PORT}"
python3 -m http.server "$HTTP_PORT" --bind 127.0.0.1 > /tmp/demo1b_http.log 2>&1 &
HTTP_PID=$!
sleep 1

echo "==> Opening the page"
open "http://localhost:${HTTP_PORT}/demo1b_webcam.html?device=${DEVICE_URL}&pipeline=${EYEPOP_PIPELINE_ID}"

echo "==> Running. Press Ctrl+C to stop everything."
wait "$FFMPEG_PID"
