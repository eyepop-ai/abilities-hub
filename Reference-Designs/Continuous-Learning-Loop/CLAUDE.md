Name: Continuous Learning Loop on EyePop.ai (Reference Design)
Goal: Teach users how to do continuous learning for vision models from an initially trained model and additional production data

Intial Setup:
    - EyePop.ai Trained a model (See: https://www.youtube.com/watch?v=XCR7vyT44fk)
    - “Production” video group #1
    - A pop configuration that uses the model
    - .env file with 
    api key
    account uuid
    original dataset uuid
    original model uuid

Steps:
    Python Script - Post process, ingest, autolabel
    - Process all data through Pop with original model
    - Look for low conf etc
    - Use tracking (trackId) to catch flicker: a track that drops below the confidence threshold intermittently (present in frame N-1 and N+1, missing/low-conf in N) is a stronger uncertainty signal than a single low-conf frame, and also surfaces near-miss false negatives that a flat confidence threshold misses
    - don’t sample too many from same video
    - don’t sample too many from same time in video/livestream
    - Ingest samples into original dataset uuid (EyePop.ai will version automatically)
    - programatically kick off auto label in script
    - review as human on EyePop's human review process -> have the code direct the user to the webpage/ format -> https://dashboard.eyepop.ai/wizardModel?type=object&step=autoLabel&accountUUID=2c97ab0b556742dbbdb7af34cc6f3b6a&modelUUID=069d7d51de87749f8000d9e3f477af3c&datasetUUID=06738f1e89d276078000cb5247fdc3c0


    Users actions:
    - User will review on EyePop.ai and hit train
    - User will preview newly trained model
    - User could update and retrain again


    Talk about production vs side car
    - recordings
    - tee off rtsp etc


Status (2026-07-21):
    - continuous_learning_loop.py, .env, .gitignore, requirements.txt exist in this folder — inference/tracking/flicker-detection/sampling/ingest/auto-label steps are implemented.
    - Not yet validated end-to-end: production compute-api is currently misreporting real pipeline/pop-config failures for this account's model as "HTTP 503: no available server" (confirmed via a bare abilityUuid probe with no forward/tracking — same error). This matches eyepop-wiki postmortem 2026-07-21-node-sdk-pipeline-error-hotfix (SESS_007 mapping fix is live in staging v1.32.0/1.32.1, not yet promoted to production as of this date).
    - Before re-running the dry run: check whether that compute-api fix has since reached production, or reproduce against staging, to see the real pipeline error instead of the misleading 503.