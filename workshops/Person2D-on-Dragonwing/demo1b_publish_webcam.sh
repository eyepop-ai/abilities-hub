#!/bin/bash
set -e
cd "$(dirname "$0")"

# Publishes this computer's webcam as an RTSP stream at rtsp://<this-machine>:8554/stream,
# which the device then pulls from (see demo1b_set_rtsp_source.py).
#
# Stock ffmpeg can't act as an RTSP *server* on its own (its rtsp muxer has no
# listen/server mode -- confirmed via `ffmpeg -h muxer=rtsp`, which lists no
# such option). MediaMTX is a small, purpose-built RTSP/RTMP/HLS server: this
# script starts it locally, then has ffmpeg PUBLISH into it (a normal RTSP
# client operation ffmpeg does support), and the device pulls that same URL as
# its own RTSP client.
#
# Usage: ./demo1b_publish_webcam.sh [camera_index]
#   camera_index defaults to 0. List devices with:
#     ffmpeg -f avfoundation -list_devices true -i ""

CAMERA_INDEX="${1:-0}"

mediamtx &
MEDIAMTX_PID=$!
trap 'kill "$MEDIAMTX_PID" 2>/dev/null' EXIT

sleep 1

# -video_size is required here: left unset, avfoundation picked this
# camera's max resolution (1552x1552 square) by default, which it can't
# actually sustain at 30fps -- ffmpeg was duplicating the same 1-2 real
# captures over and over to pad the output to the requested rate (visible
# as constant "Non-monotonic DTS" warnings and "dup" markers), which is why
# the device's detections looked frozen. 1280x720 is well within this
# camera's real full-rate capability.
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "${CAMERA_INDEX}:none" \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/stream
