# FLAKE8: noqa
import asyncio
import os
import socket
import sys
from urllib.parse import urlparse

import requests
import websockets

# Points the pipeline running on the Dragonwing device at an RTSP stream
# published from THIS computer, instead of pushing video to the device. The
# device dials OUT to this computer as a normal RTSP client, which needs no
# NAT or WebRTC ICE traversal. (WHIP, a WebRTC push, did not work here: the
# device's WebRTC ingress advertised a Docker-internal IP as its ICE
# candidate, which nothing on the network can reach.)
#
# Run demo1b_publish_webcam.sh first (separate terminal) to publish this
# computer's webcam via a local mediamtx instance, then run this script to
# point the device at it. Leave that terminal running: the device connects to
# it as an RTSP client the moment this PATCH lands.
#
# Setup (see the README):
#   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
#   export EYEPOP_PIPELINE_ID=<pipeline id printed by `eyepop instance init`>

DEVICE_URL = os.environ.get('EYEPOP_DEVICE_URL', 'http://localhost:8080').rstrip('/')
PIPELINE_ID = os.environ.get('EYEPOP_PIPELINE_ID', '')
PIPELINE_BASE = f'{DEVICE_URL}/pipelines/{PIPELINE_ID}'
RTSP_PORT = 8554
CONNECT_TIMEOUT_SECONDS = 12


def ip_device_can_reach():
    """This computer's address on the path toward the device.

    Asks the OS which local address it would use to reach the device, rather
    than which one it uses for the internet. Same network: that's the LAN IP.
    Device remote over a VPN such as Tailscale: that's the VPN IP. Advertising
    the LAN IP to a remote device is a silent failure: a private address on
    another network is unroutable, so the device never connects and nothing
    errors.
    """
    device_ip = socket.gethostbyname(urlparse(PIPELINE_BASE).hostname)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((device_ip, 8080))
        return s.getsockname()[0]
    finally:
        s.close()


async def _got_a_prediction():
    token = requests.get(f'{PIPELINE_BASE}/predictions/token', timeout=10).json()['token']
    host = urlparse(PIPELINE_BASE).hostname
    async with websockets.connect(f'ws://{host}:8080/ws/predictions/{token}') as ws:
        try:
            await asyncio.wait_for(ws.recv(), CONNECT_TIMEOUT_SECONDS)
            return True
        except asyncio.TimeoutError:
            return False


def wait_for_device_to_connect():
    # Waits for an actual prediction message, i.e. frames flowing. Neither
    # the PATCH returning 204 nor the pipeline state flipping to BUSY proves
    # that: BUSY only means the device started trying, and it stays BUSY while
    # failing to reach an unroutable source.
    return asyncio.run(_got_a_prediction())


def main():
    if not PIPELINE_ID:
        sys.exit('Set EYEPOP_PIPELINE_ID (printed by `eyepop instance init`). See the README.')
    laptop_ip = ip_device_can_reach()
    rtsp_url = f'rtsp://{laptop_ip}:{RTSP_PORT}/stream'
    print(f'Pointing the device at {rtsp_url}')

    response = requests.patch(
        # videoMode=stream avoids the API's default buffering behavior,
        # which was adding a very noticeable (~5s) delay for a live source.
        f'{PIPELINE_BASE}/source?mode=queue&processing=async&videoMode=stream',
        json={'sourceType': 'URL', 'url': rtsp_url, 'isLive': True},
        timeout=10,
    )
    response.raise_for_status()
    print(f'Source set (HTTP {response.status_code}). Waiting for the device to connect...')

    if wait_for_device_to_connect():
        print('The device is pulling the stream.')
        return

    print()
    print(f'WARNING: the device never connected to {rtsp_url} within {CONNECT_TIMEOUT_SECONDS}s.')
    print('The page will open but show no detections. Most likely the device cannot reach')
    print(f'{laptop_ip}:{RTSP_PORT} on this computer. Check from the device:')
    print(f'  bash -c "</dev/tcp/{laptop_ip}/{RTSP_PORT}" && echo reachable')
    print('If the device is remote over Tailscale, the tailnet ACL must allow it to')
    print('initiate connections to this computer.')


if __name__ == '__main__':
    main()
