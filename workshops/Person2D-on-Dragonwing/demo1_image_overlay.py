# FLAKE8: noqa
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import requests

# Person detection + 2D body points on a Dragonwing board, drawn onto an image.
#
# demo1_image.py prints the raw JSON that comes back from the device. This is
# the same call, but the result is drawn back onto the picture: a box around
# each person and the 2D body points joined into a skeleton.
#
# Everything runs on the Dragonwing board. The image goes to the device's HTTP
# API, the person + tracking + 2D body points pipeline runs on the NPU, and
# only the JSON result comes back to this computer (which draws it).
#
# processing=sync makes the request block until the result is ready, so there
# is nothing to poll.
#
# Setup (see the README):
#   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
#   export EYEPOP_PIPELINE_ID=<pipeline id printed by `eyepop instance init`>
#
# Usage: python3 demo1_image_overlay.py <image_path> [--no-open]
#
# Outputs: demo1_image_annotated.jpg and demo1_image.detections.json.

DEVICE_URL = os.environ.get('EYEPOP_DEVICE_URL', 'http://localhost:8080').rstrip('/')
PIPELINE_ID = os.environ.get('EYEPOP_PIPELINE_ID', '')
PIPELINE_BASE = f'{DEVICE_URL}/pipelines/{PIPELINE_ID}'
OUT_IMAGE = 'demo1_image_annotated.jpg'
OUT_JSON = 'demo1_image.detections.json'

SCALE = 2  # draw on a 2x upscaled copy so lines and text stay crisp on stream
LABEL_SCALE = 0.6  # label chip text size (demo4_yoga_gif.py raises it for tall video)

# EyePop's render-2d look (same colors as demo1b_webcam.html), as BGR.
PRIMARY = (215, 167, 47)     # #2fa7d7
SECONDARY = (255, 224, 148)  # #94e0ff

# Which joints to connect. Names match the classLabel values this pipeline
# returns (the model finds 17 body points, but only the ones above the
# confidence threshold come back, so people cut off at the waist have just
# the upper-body ones; a bone is skipped if either end was not detected).
BONES = [
    ('nose', 'left eye'), ('nose', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'), ('right shoulder', 'right elbow'),
    # lower arms and body, present when the person is fully in frame
    ('left elbow', 'left wrist'), ('right elbow', 'right wrist'),
    ('left shoulder', 'left hip'), ('right shoulder', 'right hip'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'), ('right hip', 'right knee'),
    ('left knee', 'left ankle'), ('right knee', 'right ankle'),
]


def detect(image_bytes):
    response = requests.post(
        f'{PIPELINE_BASE}/source?mode=queue&processing=sync',
        data=image_bytes,
        headers={'Content-Type': 'image/jpeg', 'Accept': 'application/jsonl'},
        timeout=30,
    )
    response.raise_for_status()
    return json.loads(response.text.strip().split('\n')[0])


def corner_brackets(img, x1, y1, x2, y2, size, color, thickness):
    """Four L-shaped corners, the EyePop reticle look."""
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * size), color, thickness, cv2.LINE_AA)


def draw_person(img, obj):
    x, y = int(obj['x'] * SCALE), int(obj['y'] * SCALE)
    w, h = int(obj['width'] * SCALE), int(obj['height'] * SCALE)
    x2, y2 = x + w, y + h
    dim = min(w, h)
    thickness = max(2, int(dim * 0.012))

    # translucent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), PRIMARY, -1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.rectangle(img, (x, y), (x2, y2), PRIMARY, 1, cv2.LINE_AA)

    # double corner bracket: outer in primary, smaller inset in secondary
    outer = max(20, int(dim * 0.3))
    corner_brackets(img, x, y, x2, y2, outer, PRIMARY, thickness)
    pad = max(4, int(dim * 0.04))
    corner_brackets(img, x + pad, y + pad, x2 - pad, y2 - pad, max(10, outer - int(pad * 1.33)), SECONDARY, thickness)

    # skeleton: bones first so the joint dots sit on top
    points = {}
    for kp in obj.get('keyPoints', []):
        for pt in kp.get('points', []):
            points[pt['classLabel']] = (int(pt['x'] * SCALE), int(pt['y'] * SCALE))
    for a, b in BONES:
        if a in points and b in points:
            cv2.line(img, points[a], points[b], SECONDARY, max(2, thickness - 1), cv2.LINE_AA)
    radius = max(4, int(dim * 0.025))
    for p in points.values():
        cv2.circle(img, p, radius, PRIMARY, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, SECONDARY, 2, cv2.LINE_AA)

    # label chip above the box: "Person 97%  #id"
    label = f"{obj.get('classLabel', 'person').capitalize()} {obj.get('confidence', 0) * 100:.0f}%"
    track = obj.get('traceId', obj.get('trackId'))
    if track is not None:
        label += f'  #{track}'
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, LABEL_SCALE, (1 if LABEL_SCALE < 1 else 2)
    (tw, text_h), base = cv2.getTextSize(label, font, scale, th)
    cy = max(0, y - text_h - base - 12)
    cv2.rectangle(img, (x, cy), (x + tw + 14, cy + text_h + base + 10), PRIMARY, -1)
    cv2.putText(img, label, (x + 7, cy + text_h + 4), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to a JPEG or PNG image')
    parser.add_argument('--no-open', action='store_true', help='do not open the result when done')
    args = parser.parse_args()

    if not PIPELINE_ID:
        sys.exit('Set EYEPOP_PIPELINE_ID (printed by `eyepop instance init`). See the README.')

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    print(f'Sending {args.image} to {DEVICE_URL}...')
    result = detect(image_bytes)
    objects = result.get('objects', [])
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'{len(objects)} people detected on-device:')
    for i, obj in enumerate(objects, 1):
        n_points = sum(len(kp.get('points', [])) for kp in obj.get('keyPoints', []))
        print(f"  person {i}: confidence {obj['confidence']:.2f}, {n_points} body points")

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for obj in objects:
        draw_person(img, obj)
    cv2.imwrite(OUT_IMAGE, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'Overlay saved to {OUT_IMAGE}')

    if not args.no_open and sys.platform == 'darwin':
        subprocess.run(['open', OUT_IMAGE])


if __name__ == '__main__':
    main()
