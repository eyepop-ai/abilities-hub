# Calibration

Every threshold below was tuned against one specific face library (a handful
of test videos, a few dozen people). They are reasonable **starting points**,
not universal constants — face embeddings, lighting, camera quality, and how
many people you're distinguishing between all shift where these should sit.
Re-derive them against your own data before trusting this in front of anyone
else. `CLAUDE.md` has the full dated history of how each number below was
reached, including the false starts; this doc is just the current values and
the knob to turn for each.

## Matching: is this face someone I know?

**`FAMILIARITY_THRESHOLD`** (`.env`, default `0.45`) — the cosine-similarity
cutoff between a detected face's embedding and a labeled person's nearest
stored embedding. Below it, the face is reported as "Unknown" no matter how
close the nearest match is.

- Matching is nearest-neighbor (best similarity against any ONE of a
  person's stored faces), not a centroid/average — a centroid gets washed
  out once someone has many faces spanning different looks (glasses on/off,
  angles, lighting). Measured on real data: a threshold that gave 80%
  recall / 0% false-match with a centroid gave 94% recall / 0.2% false-match
  with nearest-neighbor at the same threshold.
- **How to re-derive it**: label a reasonable-sized set of faces for 2-3
  people, then score a fresh clip of those same people against the library
  at a few threshold values (0.35, 0.40, 0.45, 0.50). Pick the value where
  false-matches (a stranger, or the wrong person, scored as familiar) are
  rare enough for your use case — going lower always trades some
  false-match rate for recall, there's no free win.
- This value is tied to the exact similarity formula in `common.py:
  cosine_similarity`/`normalize_rows`. If that formula ever changes (it has,
  more than once, during this project's development — see `CLAUDE.md`),
  re-derive the threshold again. Don't assume an old value still holds.

## Ingest: what counts as a usable face shot?

Both live in `common.py`, applied by `is_good_face_shot()`:

- **`MIN_LANDMARK_CONFIDENCE`** (default `0.65`) — the face detector emits
  eye/nose/mouth/ear landmarks with a single confidence for the whole group.
  Below ~0.50 it doesn't emit landmarks at all (a hard floor, not a quality
  signal); presence alone still isn't reliable — a manual sample across
  confidence bands found 0.50-0.60 was garbage 3/3 times (hand, black frame,
  shoulder), 0.60-0.70 was a coin flip, 0.70-0.80 was legitimate profile
  faces 3/3 times. Hence 0.65.
- **`MIN_FACE_SIZE_PX`** (default `80`) — `min(width, height)` of the face
  box, in source pixels. Below this puts a face in roughly the smallest 4%
  seen across two real test videos — likely a distant/background face, not
  a usable shot for identity matching. Face size and eye-visibility are
  measured as uncorrelated (~95-97% both-eyes-visible at every size
  bucket), so this is a genuinely separate filter, not redundant with the
  landmark-confidence one.
- **How to re-derive them**: ingest a video without either filter, sort the
  resulting faces by landmark confidence / box size, and eyeball where
  "clearly not a usable face" starts. Both were derived exactly this way —
  by looking at real flagged examples, not by guessing a round number.

## `/watch`: live webcam tracking and auto-labeling

All in `static/watch.html`. These are newer and less battle-tested than the
matching/ingest constants above — watch `#auto-add-log` and the raw-vs-smoothed
scores in the status line against reality before trusting them unattended.

- **`TRACK_WINDOW_SIZE`** (`7`) / **`TRACK_MIN_AGREEMENT`** (`0.5`) — identity
  smoothing ("inertia"). A per-frame match on its own flickers between a name
  and Unknown whenever the raw score hovers near `FAMILIARITY_THRESHOLD`; the
  displayed name only changes once this fraction of the last `N` raw matches
  for a tracked face agree. Window too short → still flickery. Window too
  long → slow to recognize someone who just walked up, and slow to notice
  they walked away.
- **`AUTO_ADD_MIN_SCORE`** (`0.35`) / **`AUTO_ADD_MIN_AGREEMENT`** (`0.7`) — a
  face can be confidently "not familiar enough to display as recognized" by
  `FAMILIARITY_THRESHOLD` while still consistently pointing at the same known
  person underneath. If the window's best-guess agreement clears
  `AUTO_ADD_MIN_AGREEMENT` and its average score clears `AUTO_ADD_MIN_SCORE`
  (deliberately below `FAMILIARITY_THRESHOLD` — that's the point: catch the
  "almost recognized" cases), the face gets added to that person's collection
  with no human review. Set `AUTO_ADD_MIN_SCORE` too low and you'll silently
  contaminate someone's collection with the wrong face — watch the log.
- **`SUGGEST_MIN_SCORE`** (`0.15`) — below the auto-add bar but above this,
  a manual capture card still defaults its person picker to the best guess
  instead of "Unassigned." Below this floor there's no real signal, so the
  card leaves it unassigned rather than confidently suggesting a coin-flip.
- **`MAX_PENDING_CAPTURES`** (`8`) — hard cap on manual capture cards kept on
  screen at once (oldest dropped first). Not a calibration knob so much as a
  UI-doesn't-take-over-the-page guard.
- **`STALE_MS`** (`1500`) — how long without a fresh result before the
  overlay clears a box / a track is considered gone. UX responsiveness, not
  a matching-quality knob.

## Sampling rate

**`SAMPLE_INTERVAL_SECONDS`** (`.env`, default `2`) — seconds between sampled
frames during `ingest.py`/`score.py`. 2s is far too sparse for a short phone
clip (a 10s video gets ~5 samples total) — pass `--interval 0.25` or smaller
for anything under a minute or so. This is a coverage/cost tradeoff, not
something to calibrate against accuracy data: smaller interval = more EyePop
calls = more faces captured = slower and (if metered) more expensive.
