# Face Familiarity

A local reference design for scoring **how familiar a face is** against a
library of people you've previously labeled — from a recorded video, a
single photo, or a live webcam feed. Built on EyePop.ai's
Person → Face → Face-Embedding pipeline.

Everything runs locally: a small stdlib Python HTTP server, SQLite for the
face library, your filesystem for thumbnails/video cache. No database
server, no build step, no framework. One EyePop API key is all the external
dependency this needs.

## Quickstart

```bash
python3.12 -m pip install -r requirements.txt
cp .env.example .env        # then fill in EYEPOP_API_KEY (dashboard.eyepop.ai)
python3.12 server.py        # -> http://localhost:8080
```

Open `http://localhost:8080`, drop in a video or a few photos, group and
name the faces that come out, then hit "Score a video" or "Watch webcam" to
see it recognize (or not) the people you just labeled.

Requires Python 3.12 (scripts are `#!/usr/bin/env python3.12`; other 3.x
versions will likely work but aren't tested). `yt-dlp` (a dependency) also
needs a JS runtime on your PATH to get past YouTube's token challenge —
Node.js satisfies this; see `common.py:download_youtube`'s comment if
YouTube ingest 403s.

## The three pages

| Route | What it's for |
|---|---|
| `/` | **Label** — ingest a video/photo, group the faces that come out into named people. Drag-and-drop or the Browse button both work; a YouTube URL or a local path (if you're running this on the same machine as your browser, which is the default assumption everywhere in this project) work too. |
| `/review` | **Score a video** — scan a new video against your labeled library; scrubs through with bounding boxes and familiar/unknown labels, lets you confirm detections straight into the library. |
| `/watch` | **Live webcam** — real-time recognition over WebRTC. Tracks faces across frames, smooths the recognized/unknown decision so it doesn't flicker, and can add confidently-unknown-but-probably-someone-you-know faces to that person's collection automatically. See "How `/watch` talks to EyePop" below — it's the one page doing something architecturally unusual. |

Each dataset (see below) has its own independent library — switch datasets
from any page's header without losing anything.

## How this actually works

1. **Ingest**: sample frames from a video (or treat a single photo as a
   one-frame video — same code path, no special case needed) through a Pop
   chain of `eyepop.person` → crop → `eyepop.person.face.short-range` → crop
   → `eyepop.face-id.large`. Every detected face becomes an unlabeled row
   (thumbnail + 512-float embedding) in SQLite.
2. **Label**: group unlabeled faces (by similarity-based clustering
   suggestions, or by hand) into named people.
3. **Score**: for a new video/photo/webcam frame, find each detected face's
   nearest stored embedding by cosine similarity, *per individual stored
   face, not a per-person average* — a centroid gets washed out once someone
   has enough faces spanning different looks (glasses on/off, angle,
   lighting). Below `FAMILIARITY_THRESHOLD`, report "Unknown" rather than
   forcing the nearest label.

`CALIBRATION.md` covers every tunable threshold in the pipeline above and how
to re-derive it for your own data — the shipped defaults were tuned against
one specific test library and won't necessarily transfer to yours.

## Datasets

Everything you label lives under `data/<dataset-name>/` (SQLite db,
thumbnails, score reports) — completely independent per dataset. Switch or
create one from the dropdown in any page's header; `downloads/` and `cache/`
(raw video files, cached EyePop results) are shared across all datasets
since they're content-addressed and re-downloading/re-inferring the same
video for a second dataset would be wasteful.

```bash
python3.12 ingest.py <youtube_url_or_path> [--interval 0.5]   # CLI, same as the UI's ingest
python3.12 score.py <youtube_url_or_path> [--threshold 0.45]  # CLI, same as the UI's /review
python3.12 reset.py [--yes] [--include-shared-cache]          # wipe the CURRENT dataset only
```

## How `/watch` talks to EyePop

`/watch` is the one page where the *browser* talks to EyePop directly over
WebRTC (`EyePopSdk.EyePop.workerEndpoint(...).process({source:
{mediaStream}})`) instead of going through this app's own Python backend —
necessary for real-time video, since round-tripping every frame through our
server first would add a full extra hop of latency for no benefit.

That means client-side JS needs *some* form of EyePop credential. **It never
gets the raw `EYEPOP_API_KEY`.** Instead:

1. The browser asks this server for a token: `GET /api/eyepop-token`.
2. The server exchanges `EYEPOP_API_KEY` for a short-lived bearer token via
   EyePop's own `POST /v1/auth/authenticate` (the same exchange the SDKs do
   internally for API-key auth) and hands back *only* that token — see
   `common.py:mint_eyepop_token`.
3. The browser connects with `{accessToken: <token>, ...}`. The token is
   good for the observed lifetime (~22h on staging) — comfortably longer
   than any real `/watch` session, so there's no client-side refresh logic.

If you're adapting this pattern elsewhere: this only works because a
compute-api access token is *itself* meant to be short-lived and scoped the
same as the key it came from — it's not a capability restriction, just a
blast-radius one (a leaked token expires; a leaked API key doesn't, until you
rotate it). Don't skip this step and ship the raw key to a browser instead —
that key doesn't expire on its own.

## Updating the JS SDK bundle

`static/eyepop.min.js` is a vendored copy of `@eyepop.ai/eyepop`'s browser
build (currently `3.17.3`) — no npm/webpack step in this project, so it's
committed as a plain file and loaded via `<script src="/eyepop.min.js">`.
To refresh it (e.g. after a new SDK release):

```bash
npm pack @eyepop.ai/eyepop@<version>
tar xzf eyepop.ai-eyepop-*.tgz
cp package/dist/eyepop.min.js static/eyepop.min.js
rm -rf package eyepop.ai-eyepop-*.tgz
```

Then update `requirements.txt`'s `eyepop` pin to match and re-verify
`/watch` end to end — the JS SDK jumped major versions (1.x → 3.x) once
already during this project's development and broke both auth (the
`secretKey`/`liveIngress` API this page originally used no longer exists)
and the live-streaming call shape. Don't assume a version bump is a no-op.

## Known limitations

This is a **single-user local tool**, not a hosted service — there's no
authentication on the app itself (anyone who can reach `localhost:8080` has
full control: add/rename/delete people, wipe a dataset, etc.), and several
things explicitly assume the server and the browser are the same machine
(e.g. a local-path video source in the ingest field). That's fine for what
this is; it is **not** safe to expose this server's port on a network
without adding your own access control first.

Other things worth knowing before you build on this:

- No automated tests — changes were verified by hand against real EyePop
  staging data throughout development (see `CLAUDE.md`'s dated log).
- SQLite + local disk storage — fine for one person's use, not built for
  concurrent multi-user writes or horizontal scaling.
- The upstream face-embedding duplicate-tensor bug documented in
  `CLAUDE.md` was confirmed fixed on EyePop's side, but the client-side
  mitigation (`common.py:extract_faces`) is left in place defensively.

## Deep dive / design log

`CLAUDE.md` is a dated running log of every design decision, bug, and
calibration measurement made while building this — useful if you want the
*why* behind something, or you're debugging a regression and want to know
if a similar issue came up before.

## License

MIT — see `LICENSE`.
