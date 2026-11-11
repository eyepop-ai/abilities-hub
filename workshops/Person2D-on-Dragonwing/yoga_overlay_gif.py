# FLAKE8: noqa
import argparse
import copy
import json
import os
import subprocess
import sys

import cv2
import numpy as np

import demo1_image_overlay as overlay

# A video with the EyePop person + 2D body points overlay, turned into a GIF.
# (Written for a yoga clip; any video with one person works.)
#
# Steps:
#   1. ffmpeg: trim the source to its first 10 s, play it at 2x so it
#      becomes a 5 s, 25 fps, 1080x1920 mp4.
#   2. Every frame of that clip goes to the person + tracking + 2D body
#      points pipeline running on the Dragonwing device. Frames are
#      sent at half size (540x960) and the result is drawn on the full-size
#      frame, which is why overlay.SCALE is 2.
#   3. The boxes and skeletons are drawn with the same brand style as
#      demo1_image_overlay.py and encoded back to an mp4.
#   4. ffmpeg turns the overlay mp4 into a GIF (two-pass palette, so the
#      brand blue stays clean).
#
# Setup (see the README):
#   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
#   export EYEPOP_PIPELINE_ID=<pipeline id printed by `eyepop instance init`>
#
# Usage: python3 yoga_overlay_gif.py <source.mp4> [--force]
# Outputs, in out/: yoga_5s.mp4, yoga_5s_results.json, yoga_5s_overlay.mp4,
# yoga_5s_overlay.gif

OUT_DIR = 'out'
CLIP_SECONDS = 10      # how much of the source to use
SPEEDUP = 2            # 10 s at 2x = 5 s
OUT_FPS = 25
OUT_W, OUT_H = 1080, 1920
SEND_W, SEND_H = 540, 960  # what the device sees (SCALE 2 maps it back to OUT size)
GIF_W = 480
GIF_FPS = 15
# Arduino Project Hub cover image: cropped to 4:3, 900x675 px or higher. The clip is
# tall (9:16), so it is cropped to a full-width 4:3 window (1080x810) that follows the
# person from about the waist up, then scaled to 1200x900. Windowing on the person
# fills the frame, where fitting the whole tall clip into 4:3 leaves wide empty bars.
COVER_W, COVER_H = 1200, 900
COVER_GIF_W = 900
# The form's upload limit is 5 MB and a 900x675 photo GIF is big, so this trades frame
# rate and palette size (and a light denoise, which helps frame-to-frame compression).
COVER_GIF_FPS = 8
COVER_GIF_COLORS = 40
CROP_W = OUT_W
CROP_H = OUT_W * 3 // 4          # 810: 4:3 at full width
CROP_TOP_MARGIN = 100            # room above the box for the label chip
CROP_SMOOTH = 9                  # frames of moving average so the window glides


def run(cmd):
    subprocess.run(cmd, check=True)


def make_clip(source, out_path):
    run([
        'ffmpeg', '-v', 'error', '-y',
        '-t', str(CLIP_SECONDS), '-i', source,  # -t before -i: trim the INPUT, before the speed-up
        '-vf', f'setpts=PTS/{SPEEDUP},fps={OUT_FPS},scale={OUT_W}:{OUT_H}:flags=lanczos',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path,
    ])


def detect_frames(clip_path, results_path, force):
    if os.path.exists(results_path) and not force:
        with open(results_path) as f:
            return json.load(f)
    cap = cv2.VideoCapture(clip_path)
    results = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (SEND_W, SEND_H), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result = overlay.detect(jpg.tobytes())
        results.append(result.get('objects', []))
        i += 1
        if i % 25 == 0:
            print(f'  {i} frames sent to the device')
    cap.release()
    with open(results_path, 'w') as f:
        json.dump(results, f)
    return results


def fill_gaps(results, max_gap=2):
    """If the detector drops the person for a frame or two, hold the last result
    instead of letting the overlay flicker off. Longer gaps are left empty."""
    filled, last, gap = [], [], 0
    for objects in results:
        if objects:
            last, gap = objects, 0
            filled.append(objects)
        else:
            gap += 1
            filled.append(last if (last and gap <= max_gap) else objects)
    return filled


def hold_points(results, max_gap=20):
    """The device only returns a body point when its confidence is above the
    pipeline's threshold (0.5), so a point that hovers right at the threshold
    (here, a wrist hidden behind the other hand) blinks in and out and its
    bone flickers. For single-person clips, fill each gap of up to max_gap
    frames (0.8 s) by drawing the point on a straight line between where it
    was last seen and where it next reappears. A point that never reappears
    is held in place for up to 8 frames, then dropped. Gaps before the first
    sighting are left alone."""
    held = copy.deepcopy(results)
    single = [i for i, objs in enumerate(held)
              if len(objs) == 1 and objs[0].get('keyPoints')]
    labels = {pt['classLabel'] for i in single for pt in held[i][0]['keyPoints'][0]['points']}
    for label in labels:
        seen = {}  # frame index -> point
        for i in single:
            for pt in held[i][0]['keyPoints'][0]['points']:
                if pt['classLabel'] == label:
                    seen[i] = pt
        frames = sorted(seen)
        for a_i, b_i in zip(frames, frames[1:]):
            gap = [i for i in single if a_i < i < b_i]
            if not gap or b_i - a_i - 1 > max_gap:
                continue
            for i in gap:
                t = (i - a_i) / (b_i - a_i)
                pa, pb = seen[a_i], seen[b_i]
                pt = dict(pa)
                pt['x'] = pa['x'] + (pb['x'] - pa['x']) * t
                pt['y'] = pa['y'] + (pb['y'] - pa['y']) * t
                held[i][0]['keyPoints'][0]['points'].append(pt)
        if frames:  # trailing gap: hold the last position briefly
            last = frames[-1]
            for i in single:
                if last < i <= last + 8:
                    held[i][0]['keyPoints'][0]['points'].append(dict(seen[last]))
    return held


def render(clip_path, results, out_path):
    overlay.SCALE = OUT_W // SEND_W
    overlay.LABEL_SCALE = 1.5  # the clip is tall and gets shrunk to a 480px GIF
    results = hold_points(fill_gaps(results))
    cap = cv2.VideoCapture(clip_path)
    enc = subprocess.Popen([
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', str(OUT_FPS), '-i', '-',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path,
    ], stdin=subprocess.PIPE)
    for objects in results:
        ok, frame = cap.read()
        if not ok:
            break
        for obj in objects:
            overlay.draw_person(frame, obj)
        enc.stdin.write(frame.tobytes())
    cap.release()
    enc.stdin.close()
    enc.wait()


def make_gif(mp4_path, gif_path):
    vf = f'fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos'
    run([
        'ffmpeg', '-v', 'error', '-y', '-i', mp4_path,
        '-vf', f'{vf},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];'
               f'[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
        '-loop', '0', gif_path,
    ])


def crop_tops(results):
    """Top edge of the 4:3 window for each frame: just above the person's box,
    smoothed, and kept inside the frame."""
    tops, last = [], 0.0
    for objects in results:
        if objects:
            last = min(obj['y'] for obj in objects) * overlay.SCALE - CROP_TOP_MARGIN
        tops.append(last)
    half = CROP_SMOOTH // 2
    smooth = []
    for i in range(len(tops)):
        window = tops[max(0, i - half):i + half + 1]
        smooth.append(sum(window) / len(window))
    return [int(min(max(t, 0), OUT_H - CROP_H)) for t in smooth]


def make_cover(overlay_mp4, results, cover_mp4, cover_gif, cover_jpg):
    tops = crop_tops(hold_points(fill_gaps(results)))
    cap = cv2.VideoCapture(overlay_mp4)
    enc = subprocess.Popen([
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{CROP_W}x{CROP_H}', '-r', str(OUT_FPS), '-i', '-',
        '-vf', f'scale={COVER_W}:{COVER_H}:flags=lanczos',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', cover_mp4,
    ], stdin=subprocess.PIPE)
    for top in tops:
        ok, frame = cap.read()
        if not ok:
            break
        enc.stdin.write(np.ascontiguousarray(frame[top:top + CROP_H, :CROP_W]).tobytes())
    cap.release()
    enc.stdin.close()
    enc.wait()
    vf = f'hqdn3d=4:4:8:8,fps={COVER_GIF_FPS},scale={COVER_GIF_W}:-1:flags=lanczos'
    run([
        'ffmpeg', '-v', 'error', '-y', '-i', cover_mp4,
        '-vf', f'{vf},split[a][b];[a]palettegen=max_colors={COVER_GIF_COLORS}:stats_mode=diff[p];'
               f'[b][p]paletteuse=dither=none:diff_mode=rectangle',
        '-loop', '0', cover_gif,
    ])
    # a still frame from mid-clip, in case the form does not animate the cover
    run(['ffmpeg', '-v', 'error', '-y', '-ss', '1.0', '-i', cover_mp4,
         '-frames:v', '1', '-q:v', '2', cover_jpg])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='path to a portrait (9:16) video with one person in it')
    parser.add_argument('--force', action='store_true', help='re-run detection even if cached')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    clip = f'{OUT_DIR}/yoga_5s.mp4'
    results_path = f'{OUT_DIR}/yoga_5s_results.json'
    overlay_mp4 = f'{OUT_DIR}/yoga_5s_overlay.mp4'
    gif = f'{OUT_DIR}/yoga_5s_overlay.gif'

    print('1. trimming and speeding up the source...')
    make_clip(args.source, clip)
    print('2. running every frame through the device...')
    results = detect_frames(clip, results_path, args.force)
    empty = sum(1 for objs in results if not objs)
    print(f'   {len(results)} frames, {empty} with no person detected')
    print('3. drawing the overlay...')
    render(clip, results, overlay_mp4)
    print('4. making the GIF...')
    make_gif(overlay_mp4, gif)
    print(f'   {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)')
    print('5. making the 4:3 cover image...')
    cover_gif = f'{OUT_DIR}/yoga_cover_4x3.gif'
    make_cover(overlay_mp4, results, f'{OUT_DIR}/yoga_cover_4x3.mp4', cover_gif,
               f'{OUT_DIR}/yoga_cover_4x3.jpg')
    print(f'Done: {cover_gif} ({os.path.getsize(cover_gif) / 1e6:.1f} MB)')


if __name__ == '__main__':
    sys.exit(main())
