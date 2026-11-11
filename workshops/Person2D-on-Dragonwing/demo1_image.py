# FLAKE8: noqa
import asyncio
import json
import os
import sys

from eyepop import EyePopSdk

# Person + Tracking + 2D body points on a Dragonwing board.
#
# Prints the raw JSON the device returns for an image.
#
# Setup (see the README):
#   export EYEPOP_DEVICE_URL=http://<device-hostname-or-ip>:8080
#   export EYEPOP_PIPELINE_ID=<pipeline id printed by `eyepop instance init`>
#
# Usage: python3 demo1_image.py <image URL or path>
#
# This client-side copy of the pop is used only so async_worker can skip
# re-fetching and validating the live pop from the device (that fetch can hit
# a schema mismatch between some SDK versions and the running runtime). It is
# NOT pushed to the device: the device already has this pipeline loaded via
# `eyepop instance init --pop ./pipeline_1_person2d.json`. Two fields differ
# from that file to satisfy the SDK's stricter schema: `reidAbility` isn't a
# recognized field here (only `reidModel`/`reidModelUuid` are), and the crop
# operator's `maxItems` must be nested under a `crop` key rather than flattened.
POP = {
    "components": [
        {
            "type": "inference",
            "ability": "eyepop.person:latest",
            "inferenceTypes": ["object_detection"],
            "confidenceThreshold": 0.9,
            "forward": {
                "targets": [
                    {
                        "type": "tracking",
                        "maxAgeSeconds": 2,
                        "motionModel": "random_walk",
                        "agnostic": False,
                        "classHysteresis": False
                    },
                    {
                        "type": "inference",
                        "ability": "eyepop.person.2d-body-points:latest",
                        "inferenceTypes": ["key_points"],
                        "categoryName": "Person 2D Body Points",
                        "confidenceThreshold": 0.5
                    }
                ],
                "operator": {"type": "crop", "crop": {"maxItems": 4}}
            }
        }
    ]
}


async def main(image):
    async with EyePopSdk.async_worker(
        eyepop_url=os.environ.get('EYEPOP_DEVICE_URL', 'http://localhost:8080'),
        is_local_mode=True,
        pop_id=os.environ['EYEPOP_PIPELINE_ID'],
        pop=POP
    ) as endpoint:
        if image.startswith(('http://', 'https://')):
            job = await endpoint.load_from(image)
        else:
            job = await endpoint.upload(os.path.abspath(image))
        while result := await job.predict():
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2 or 'EYEPOP_PIPELINE_ID' not in os.environ:
        sys.exit('Usage: EYEPOP_PIPELINE_ID=<id> python3 demo1_image.py <image URL or path>\n'
                 'See the README for EYEPOP_DEVICE_URL and EYEPOP_PIPELINE_ID.')
    asyncio.run(main(sys.argv[1]))
