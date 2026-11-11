# Person detection, tracking and 2D body points on a Qualcomm Dragonwing board

Build a person detection + tracking + 2D body points pipeline in the cloud with EyePop,
run it on the Dragonwing NPU with the EyePop Edge Runtime (QAIRT), and draw the result on
images, a live webcam feed, and video.

The full walkthrough is in the Arduino Project Hub post (link to be added).

## What is here

| File | What it does |
|---|---|
| `pipeline_1_person2d.json` | The pipeline ("pop"): person detection, then tracking and 2D body points on each person crop |
| `demo1_image.py` | Prints the raw JSON the device returns for an image |
| `demo1_image_overlay.py` | Draws the result on the image: a box and confidence per person, plus a skeleton |
| `demo1b_start.sh` | Live camera demo: publishes your webcam over RTSP, points the device at it, opens the overlay page |
| `demo1b_publish_webcam.sh` | Just the RTSP publisher (mediamtx + ffmpeg) |
| `demo1b_set_rtsp_source.py` | Points the device's pipeline at your RTSP stream |
| `demo1b_webcam.html` | Browser page that shows the camera and draws live detections |
| `mediamtx.yml` | Minimal config for the local RTSP server |
| `yoga_overlay_gif.py` | Runs a video through the device frame by frame and renders an overlay mp4 and GIF |

## Setup

1. Create an EyePop account and an API key at https://dashboard.eyepop.ai/api-keys.
2. On the Dragonwing board, install the EyePop CLI:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/eyepop-ai/homebrew-eyepop/main/install.sh | sh
   ```
3. Copy `pipeline_1_person2d.json` to the board, then export your key and initialize the Edge Runtime:
   ```bash
   export EYEPOP_API_KEY='<your-api-key>'
   export ACCOUNT_UUID='<your-account-uuid>'
   export QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt.slim   # path of the QAIRT install on your board
   eyepop instance init --pop ./pipeline_1_person2d.json --bind 0.0.0.0
   ```
   The output includes the pipeline ID. Check the device with `curl http://<device>:8080/health`.
4. On the computer you will run the demos from, point the scripts at the device:
   ```bash
   pip install -r requirements.txt
   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
   export EYEPOP_PIPELINE_ID=<pipeline id from step 3>
   ```

Never put your API key in a script or commit it. It is only needed on the board, in the shell
where you run `eyepop instance init`. None of the scripts here use it.

## Run

```bash
python3 demo1_image.py photo.jpg               # raw JSON
python3 demo1_image_overlay.py photo.jpg       # overlay image + JSON
./demo1b_start.sh                              # live webcam (needs ffmpeg and mediamtx)
python3 yoga_overlay_gif.py clip.mp4           # overlay video and GIF (needs ffmpeg)
```

Bring your own image and video. Use media you have the right to use, and check that anyone
who appears in it is fine with that.

## Notes

- The overlay uses the pipeline's own confidence threshold (0.5 for body points). A point
  below it is not returned, so people cut off at the waist come back with only upper-body points.
- A single still image has no history, so track IDs only appear on video and live streams.
- If the device is remote (for example over Tailscale), it must be able to open a connection
  back to your computer on port 8554 for the live camera demo.
