Name: Face Familiarity Scoring on EyePop.ai (Reference Design)
Goal: Score how familiar a face in a video is against a dataset of previously seen people.

Pipeline (see plan.md for the original spec):
    - Pop: eyepop.person -> eyepop.person.face.short-range -> eyepop.face-id.large
      (person crop -> face crop -> 512-dim face embedding). Defined once in common.py.
      Was eyepop.face-id.base until 2026-08-07 — swapped for the larger embedding
      model; still 512-dim, no downstream schema changes needed.
    - ingest.py samples a video every SAMPLE_INTERVAL_SECONDS, runs the Pop on each
      sampled frame, and stores one unlabeled face (thumbnail + embedding) per
      detection in the CURRENT dataset's library.db (SQLite; see dataset.py and
      the "Datasets" design note) — skipping any face that fails
      common.py:is_good_face_shot (no eyes visible, or too small; see design
      notes). Appends per-frame timing (video-decode ms vs. EyePop round-trip ms)
      and a run summary to logs/ingest.log (shared across datasets, dataset name
      included per-line).
    - The label UI (`/` in server.py, static/label.html) lets a human group
      unlabeled faces (via a per-face "similar faces" nearest-neighbor panel, or
      auto-clustered "Suggested groups") and assign them to a named person —
      rename/delete people, drag-and-drop faces onto a person to assign, and a
      per-person "unassigned faces similar to this person" panel to keep growing
      an existing person's set. A dataset switcher in the header (also in
      static/review.html) picks which face library you're looking at.
    - score.py samples a new video the same way, then matches each detected face's
      embedding by nearest-neighbor against every labeled person's individual
      stored embeddings (not an average — see design notes) via
      common.py:similarity_matrix. Below FAMILIARITY_THRESHOLD it's reported
      "Unknown" rather than forced into the closest label. Streams progress via an
      `on_update` callback (see design notes) and writes the final JSON report
      under the current dataset's reports/ dir, including each detection's
      person_bbox/face_bbox and the video's source_width/height (for the review
      UI's overlay) and video_path (for /video streaming).
    - The review UI (`/review` in server.py, static/review.html) is a 3-step flow:
      (1) enter a URL/path, (2) a brief progress panel only until the video itself
      is downloaded and playable, (3) the video playing immediately with
      canvas-drawn person/face bounding boxes and a name-or-"Unknown" label synced
      to playback time, plus a confirm/correct grid — both fill in progressively
      as the background scan continues rather than waiting for it to finish (see
      design notes). Confirming a detection inserts that face into library.db,
      closing the loop so the library improves over time.

Setup:
    - .env: EYEPOP_API_KEY (required). SAMPLE_INTERVAL_SECONDS, FAMILIARITY_THRESHOLD,
      PORT are optional overrides (see common.py / server.py for defaults). DATASET
      overrides which dataset a script starts on (see dataset.py) — otherwise it's
      whatever was last selected via the label UI's switcher (persisted to
      data/.current).
    - pip install -r requirements.txt (needs ffmpeg on PATH for some YouTube formats
      via yt-dlp, and requires the eyepop package's own dependencies).

Usage:
    python3.12 server.py                      # http://localhost:8080 (label) and /review (score)
    python3.12 ingest.py <youtube_url_or_path> # CLI equivalent of the ingest job
    python3.12 score.py <youtube_url_or_path>  # CLI equivalent of the score job
    python3.12 reset.py [--yes]                # wipe the CURRENT dataset's library.db/
                                                # thumbnails/reports only — other datasets,
                                                # and the shared downloads/cache/logs, are
                                                # untouched unless --include-shared-cache

Design notes:
    - Configurable sample interval (2026-08-10): SAMPLE_INTERVAL_SECONDS (default
      2s) was a fixed global — fine for a 30min TV episode, way too sparse for a
      ~15s phone clip (only ~8 samples total). common.py:scan_video,
      ingest.py:ingest_video, and score.py:score_video all gained an `interval`
      param (falls back to SAMPLE_INTERVAL_SECONDS if omitted); ingest.py/
      score.py expose `--interval`, server.py's /api/ingest and /api/score read
      an optional `interval` from the POST body, and both static/label.html and
      static/review.html have an interval input next to the source field.
      _results_cache_path now takes the interval as a parameter (was reading the
      global directly) so different intervals for the same video get separate
      cache files rather than colliding or reusing a mismatched sample set.
      Verified: a ~14.6s clip at interval=0.5 produced 28 faces through the real
      /api/ingest endpoint, versus the handful the 2s default would give.
    - Native file picker, server.py:_pick_file (2026-08-10): label.html's
      ingest bar has a "Browse..." button for local files. A browser
      `<input type="file">` deliberately never exposes the real filesystem
      path (only the bare filename) — no amount of frontend cleverness gets
      around that, it's a deliberate browser security boundary — and
      ingest.py (which needs an actual path to open with cv2) hits a dead
      end there. Works around it with a server-side native macOS "choose file"
      dialog (`osascript -e 'choose file ...'`) instead, which only works
      because server and browser are the same machine for this tool (see
      CLAUDE.md's single-user-local framing elsewhere) — POST /api/pick-file
      runs it and returns the POSIX path, or null if the user cancels
      (osascript exits non-zero with "User canceled (-128)", not a real
      error). Verified live: the dialog opens and a cancel is handled cleanly.
    - Video rotation, common.py:_open_video (2026-08-10): cv2.VideoCapture does
      NOT apply a video container's rotation metadata by default — phone videos
      (confirmed on 4 real iPhone .MOV files, all `rotation=-90` per
      `ffprobe -show_entries stream_side_data=rotation`) get read as raw
      1920x1080 landscape sensor frames even though the video is actually
      1080x1920 portrait and plays upright in every normal player. Every
      downstream step (face-id inference, bbox coordinates, thumbnails, the
      review UI's overlay) would be working against sideways frames without
      this. Fixed by setting `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)`
      (OpenCV 4.5+; installed version here is 4.11) on every VideoCapture — a
      single flag, no manual ffprobe-then-cv2.rotate() step needed. Both
      sample_frame_seconds and grab_frame_jpeg route through the new
      _open_video() helper so this can't be forgotten on a future call site.
      Verified against real footage, not just dimensions: extracted frames
      with and without the flag, compared side by side — without it the
      subject and on-shirt text are rotated 90°; with it, upright and correctly
      oriented.
    - Datasets (dataset.py, 2026-08-10): multiple independent face libraries
      coexist under data/<slug>/{library.db,thumbnails,reports} — added so a
      second labeling project (an "employee familiarity" test) could start
      completely clean without touching or losing the existing "30-rock"
      library (27 people, 2,855 faces at the time of the split). db.py's
      connect() and common.py's thumbnail_dir()/report_dir() all resolve
      dataset.get_current() fresh on every call — never cache the Path they
      return past a single call, a dataset switch must take effect on the very
      next db.connect() without a server restart. This is WHY THUMBNAIL_DIR/
      REPORT_DIR/DB_PATH used to be plain module-level Path constants and now
      aren't: `from common import THUMBNAIL_DIR` binds the OLD Path object
      forever at import time, so reassigning common.THUMBNAIL_DIR later
      wouldn't be seen by anything that already did that import — they had to
      become functions (thumbnail_dir(), report_dir()) that every caller
      invokes, not values imported once. Deliberately NOT dataset-scoped:
      common.py's DOWNLOAD_DIR/FRAME_CACHE_DIR/RESULTS_CACHE_DIR (downloads/,
      cache/) — content-addressed by video id/stem/POP_FINGERPRINT already, so
      sharing them across datasets is safe and avoids redundant
      downloads/EyePop calls if the same video is ever used in two datasets;
      reset.py leaves them alone by default for exactly this reason (wiping
      them for one dataset would force every OTHER dataset to redo EyePop
      calls too — see reset.py's --include-shared-cache).
      Current dataset is a process-wide global (dataset._current), persisted
      to data/.current so it survives a server restart (DATASET env var
      overrides the persisted choice, e.g. for scripts) — NOT per-request/
      session, so two browser tabs switching different datasets at once would
      race each other. Fine for this tool's single-user local use; would need
      real session/request-scoped state (e.g. a query param or cookie
      threaded through every db.connect() call) to support concurrent users on
      different datasets, which is a much bigger change than was warranted
      here.
      Migration performed when this landed: the pre-existing top-level
      library.db/thumbnails/reports (the "30 Rock" work) were moved to
      data/30-rock/ verbatim (`mv`, not through the dataset API, to avoid its
      own directory-creation racing the move) and verified intact (27 people,
      2,855 faces) before creating data/employees/ as a fresh dataset and
      switching to it.
    - Face quality filter, common.py:is_good_face_shot (2026-08-08): drops a face
      before it's ever stored/matched if it fails any of three checks —
      _eye_landmarks(face_obj) both-eyes-present, its landmark confidence
      >= MIN_LANDMARK_CONFIDENCE (0.65), or MIN_FACE_SIZE_PX (80px,
      min(width,height)). Data behind these, measured across 3,749 real face
      detections from both plan.md videos:
        - keyPoints landmarks (right/left eye, nose, mouth, right/left ear) are
          all-or-nothing per face — every detection has all 6 or none, never a
          partial set — so "both eyes visible" is equivalent to "landmarks
          present at all." 95.7% of faces have them; the other 4.3% get zero
          keyPoints even when the face bbox itself is detected with high
          confidence (up to 0.97) — this looks like genuine occlusion/extreme
          profile angle, not just a low-quality detection.
        - Presence alone is NOT a reliable quality signal, confirmed by real
          user-flagged examples: a user manually grouped two faces they'd
          spotted with no eyes actually visible (one was the back of a head,
          one was an unrecognizable dark blur) — both had all 6 landmark labels
          present, at confidence 0.538 and 0.541. Landmark confidence (one
          shared value for the whole keyPoints group, not per-point — right
          eye/left eye/nose/mouth/ears always carry the identical confidence
          within one face) has a hard floor at exactly 0.500 — below that the
          model omits points entirely rather than emitting a low-confidence
          guess — but that floor is an emission threshold, not a quality bar.
          A follow-up manual sample (9 real faces spread across the 0.50-0.60/
          0.60-0.70/0.70-0.80 confidence bands, viewed as actual image crops)
          found 0.50-0.60 was garbage 3/3 times (a hand, a black frame, a
          shoulder — no face at all), 0.60-0.70 was a coin flip, and 0.70-0.80
          was legitimate (if often profile-angle) faces 3/3 times — hence
          MIN_LANDMARK_CONFIDENCE=0.65, roughly the middle of that transition.
        - Face size (min(width,height) in source pixels) and eye-visibility are
          UNCORRELATED — 95.5-96.6% both-eyes-visible at every size bucket
          checked, including the smallest faces. So size is a genuinely separate
          quality signal, not redundant with the landmark checks.
        - MIN_FACE_SIZE_PX=80 sits just above the p5 percentile (~85px) of real
          face sizes — excludes roughly the smallest ~4%, which read as
          distant/background faces, not primary subjects.
      Applied in both ingest.py (skip before storing) and score.py (skip before
      matching) via the same is_good_face_shot() call — a face that's an
      unreliable embedding shouldn't be trusted for library-building OR
      scoring. Adding MIN_LANDMARK_CONFIDENCE increased the drop rate a lot:
      89/1,472 (6.0%) with presence-only, 347/1,472 (23.6%) once confidence was
      added — most of what presence-only let through in the 0.50-0.70 band was
      exactly the kind of garbage the user's two examples represented. Re-ran
      with a random 5-face visual spot-check after the confidence filter landed
      — all 5 were genuinely clear, usable face shots.
    - FIXED on EyePop staging (2026-08-08, confirmed via test_duplicate_embeddings.py
      against both known reproduction frames — 2-face and 3-face, both now return
      distinct embeddings per face). common.py:extract_faces's duplicate-drop
      filter is now a no-op in the common case but left in place as a safety net
      (costs nothing, and guards against a regression or the same class of bug
      resurfacing). The rest of this note is kept for history/context:
    - Upstream bug, now fixed (was CRITICAL, confirmed 2026-08-07 night): when a sampled frame has
      2+ people, EyePop can return the SAME embedding tensor for every face in that
      frame instead of each face getting its own — reproduced directly against a
      fresh raw API call (not a parsing bug on our side): two faces at completely
      different, non-overlapping bounding boxes came back byte-for-byte identical,
      and a 3-face frame had all three identical. Looks like the first face's
      face-id result gets broadcast to every face slot in a multi-person frame
      rather than each crop getting its own inference. Not documented in
      eyepop-wiki as of this writing — worth reporting upstream; the person/face
      CropForward nesting with multiple person branches per frame is the
      reproduction case. common.py:extract_faces now defensively drops every face
      in a frame if 2+ share a byte-identical embedding (we can't tell which one,
      if any, is genuinely correct, so trusting any of them risks a confident wrong
      match) — this only prevents NEW contamination. Measured against the live
      library the same night: 940 of 1,472 faces (64%!) already in library.db were
      built before this filter existed and share a same-frame duplicate embedding
      — every single labeled person is affected. Centroids built from a library
      this contaminated are unreliable; this was likely a real contributor to
      "Unknown" rates being high and to person-to-person separation quality
      varying so much between people (see the FAMILIARITY_THRESHOLD note below).
      Decision (2026-08-07, night): reset.py + full re-ingest once the fix and
      everything else pending was confirmed in place, rather than surgically
      unassigning the 940 rows — simpler than a cleanup pass, and the fresh
      ingest can't add new instances of this contamination since extract_faces
      filters it at the source now. The (source, seconds)-grouped duplicate-blob
      query used to find these is still worth keeping in mind as a "resolve exact
      duplicates"-style cleanup action if this ever needs doing again without a
      full reset.
    - No tracking: unlike Continuous-Learning-Loop, this Pop has no TrackingComponent,
      so both ingest and score sample the video at a fixed wall-clock interval
      (cv2, not EyePop-side sampling) rather than processing every frame — a face
      that lingers on screen for 10s at a 2s interval yields ~5 near-duplicate
      embeddings, which is fine for a nearest-neighbor library but means the same
      person will show up as several separate face rows.
    - "Familiar" is a per-person centroid (mean of that person's assigned
      embeddings), not nearest-single-neighbor — more stable as more faces get
      labeled, but a person's first face defines their centroid alone until more
      are added.
    - common.py:cosine_similarity is a raw dot product (`np.sum(a * b)`), not
      normalized cosine similarity — changed 2026-08-07 (later same day), and kept
      because it measurably improved matching precision on real data (see Status).
      Embedding norms in this library run ~0.65-1.93, so scores are NOT bounded to
      [-1, 1] the way true cosine similarity is — pairwise dot products up to ~3.3
      have been observed. Every threshold in this codebase (FAMILIARITY_THRESHOLD,
      the clustering default) is calibrated against THIS metric's actual value
      distribution, not a textbook cosine-similarity range — don't reuse "cosine
      similarity" intuition (e.g. "0.8 means basically identical") when tuning them.
    - FAMILIARITY_THRESHOLD (default 0.30, was 0.45 under the old normalized metric)
      is centroid-based: retuned by measuring every labeled face's dot product
      against its own person's centroid ("own", n=196) vs. every other person's
      centroid ("cross", n=1176). At 0.30: recall (own >= t) = 47.4%, false-match
      rate (cross >= t) = 0.9%. Separation quality varies a lot by person — one
      person's faces never overlap with cross-person scores at all, another's
      same-person mean (0.15) barely clears the cross-person mean (0.11) — so no
      single threshold is clean for every person; 0.30 favors precision (rather
      say "Unknown" than misattribute) at a real recall cost. Re-derive this
      whenever the embedding model, image resolution, or similarity function
      changes — the numbers above are specific to eyepop.face-id.large at 1080p
      with this raw-dot-product metric, not a general constant.
    - The label UI's "Suggested groups" (server.py:_cluster_unassigned) auto-clusters
      unassigned faces by this same metric so many faces can be assigned to a
      person at once. Uses complete-linkage (a candidate must be above threshold
      against EVERY current member, not just the seed) — star clustering (vs. seed
      only) let one ambiguous face bridge two different people into one group, which
      is what "too disparate of people clumped together" (2026-08-07 user feedback)
      turned out to be. Default threshold is 0.40 (was 0.58 under the old metric):
      retuned from real pairwise face-to-face scores (same-person mean 0.46 n=4664;
      cross-person mean 0.089 max 2.10(!) n=14446 — that one cross-person outlier
      means no pairwise threshold is airtight alone, complete-linkage's
      all-pairs-must-clear requirement is what actually keeps false merges rare in
      practice, not the threshold in isolation). At 0.40: same-person recall 28.8%,
      cross-person false-positive rate 1.4%.
      Known side-effect, not yet addressed: the biggest suggested groups are often
      near-duplicate detections from a single video frame (the person detector
      emitting several overlapping boxes on one face in a busy shot) rather than
      genuine cross-time recurrence — correct to assign, but low-value, and it
      crowds out real multi-timestamp clusters since groups sort by size. Consider
      deduping same-frame near-duplicates at ingest time, or ranking clusters by
      distinct-timestamp count instead of raw size.
    - Same-frame duplicate detections are not just cosmetic: measured directly
      against the live library (2026-08-07, night), 112 of 209 unassigned faces
      (54%!) were exact bit-for-bit duplicate embeddings of a face already assigned
      to someone — same video timestamp, sibling overlapping box. ingest.py stores
      every detection with no dedup, so a person with many labeled faces
      accumulates plenty of these, and they drown out real candidates in any
      similarity ranking unless explicitly excluded (see _person_suggestions below).
      Root fix would be deduping exact/near-exact same-frame embeddings at ingest
      time; not done — every ranking function that touches the unassigned pool
      needs to keep this in mind until it is.
      "Resolve exact duplicates" (label.html button -> POST
      /api/faces/resolve-duplicates -> server.py:_resolve_exact_duplicates) is a
      mitigation, not the root fix: since an exact bit-for-bit embedding match to
      an already-labeled face is unambiguous (not a similarity guess), it
      auto-assigns those to the same person in one click. First real run
      (2026-08-07, night) resolved all 101 remaining unassigned faces at once —
      after enough manual labeling, same-frame duplicates can end up being 100% of
      the unassigned pool, which is what made _person_suggestions look completely
      broken for "Tracy" (0 suggestions) right before this was added: every
      unassigned face was a duplicate of someone already labeled, correctly
      excluded from suggestions, leaving nothing fresh to show.
    - _person_suggestions (the label UI's per-person "unassigned faces similar to
      this person" panel) was ranking by a centroid — cosine similarity to the
      mean/sum of the person's stored embeddings. That degrades badly once a
      person has many faces spanning different angles/expressions: verified
      directly on "Jenna" (116 labeled faces) that a real, high-confidence,
      visually-confirmed Jenna face sitting unassigned ranked #105 of 209 under
      the centroid — invisible in a panel that only shows the top 60. Switched to
      nearest-neighbor (max similarity to ANY of the person's individual stored
      faces, not their average) plus excluding exact-duplicate embeddings of any
      already-labeled face (see note above) — the same real face jumped to #2.
      Re-verify this holds for other people too; Jenna was the one reported
      problematic, not an exhaustive check.
    - cosine_similarity's exact formula has been edited back and forth during this
      project (normalized dot/|a||b| vs. raw dot product) — whatever it currently
      is, every hardcoded threshold in this file (FAMILIARITY_THRESHOLD, the
      clustering default) was derived against a SPECIFIC prior version of it and
      may no longer be valid. If matching/clustering results look off, check
      common.py:cosine_similarity's current formula first and re-run the
      threshold-derivation approach in the notes above against real library data
      before assuming the threshold itself is wrong.
    - Server is stdlib `http.server` (ThreadingHTTPServer), no web framework —
      matches the brainpowerstudio demo's local-UI pattern. Long-running
      ingest/score jobs run in a background thread with in-memory job-status
      polling (JOBS dict in server.py); state is lost on server restart, but the
      underlying library.db and reports/*.json are not.
    - yt-dlp format string matters: the original `"mp4/bestvideo[ext=mp4]+..."`
      picked the bare `"mp4"` alternative first — YouTube's combined-stream mp4
      tops out around 360p, so every ingested/scored video was silently 640x360
      even though 1080p was available. Fixed to
      `"bestvideo[height<=1080]+bestaudio/best[height<=1080]"` (no bare-mp4
      fallback ahead of the adaptive streams) — verify with `cv2.VideoCapture(...).
      get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` after any further format-string changes,
      the failure mode is silent.
    - yt-dlp extraction can 403 even on the latest version — recent YouTube anti-bot
      changes require solving a JS token challenge, and yt-dlp's default runtime
      (deno) usually isn't installed. Fixed by passing `"js_runtimes": {"node": {}}`
      in download_youtube's YoutubeDL options (node was already present via nvm/
      Homebrew). Two API gotchas if this needs revisiting: the CLI flag
      `--js-runtimes RUNTIME[:PATH]` maps to a `js_runtimes` dict option, NOT a
      list — `{"node": {}}`, not `{"node": None}` (None crashes with
      `AttributeError: 'NoneType' object has no attribute 'get'` despite passing
      the dict-shape validator) and not `["node"]` (raises `ValueError: Invalid
      js_runtimes format`). A "Remote component challenge solver script (node) was
      skipped" warning can still print even when this works — extraction/download
      succeeding is what to check, not warning-free output.
    - /video/<path> in server.py streams whatever score.py's report.video_path
      points at (Range-request support for seeking, copied from the
      brainpowerstudio demo's pattern) — same trust boundary as running score.py
      with that path directly from the CLI, not scoped to a single directory.
      review.html's canvas overlay scales report person_bbox/face_bbox (in source
      pixel coordinates) by canvas.width/report.source_width — recompute the
      canvas size on the video's loadedmetadata and on window resize, or boxes
      drift off the rendered video. Boxes are drawn from a FLOOR lookup — the
      largest sampled timestamp <= currentTime (review.html:floorKey, binary
      search over sortedSeconds) — not nearest-in-either-direction; the original
      nearest-neighbor version showed a sample's boxes as soon as playback crossed
      the MIDPOINT before it, i.e. up to 1s early at a 2s sample interval (user-
      reported "boxes come in too early", 2026-08-07 night). Boxes go stale
      (disappear) `staleAfter` seconds past the last matching sample — computed
      from the actual gap between sorted sample timestamps, not a hardcoded
      constant, so it adapts if SAMPLE_INTERVAL_SECONDS changes.
    - EyePop results are cached (common.py:RESULTS_CACHE_DIR, keyed by video stem +
      SAMPLE_INTERVAL_SECONDS + POP_FINGERPRINT) — re-scanning/re-ingesting the same
      video needs zero EyePop connection if every sampled frame's result is already
      cached; partial coverage still uses the cache for what it has and only calls
      EyePop for the missing timestamps. Bump POP_FINGERPRINT whenever POP's
      abilities change (there's no automatic invalidation otherwise). Frame jpegs
      are always re-grabbed locally from the video file regardless of cache
      state — cheap, and needed for thumbnail cropping either way. Video downloads
      also short-circuit before ever calling yt-dlp if `downloads/<id>.mp4`
      already exists (common.py:_youtube_id parses the id from the URL with no
      network call) — belt-and-suspenders on top of yt-dlp's own skip-if-exists
      behavior, and avoids the metadata-lookup network round trip entirely for an
      already-downloaded video.

Status (2026-08-07):
    - Initial build from plan.md: common.py, db.py, ingest.py, score.py, server.py,
      static/label.html, static/review.html, requirements.txt, .gitignore exist.
    - Ran end-to-end against EyePop staging (EYEPOP_URL=https://compute.staging.eyepop.xyz)
      using both plan.md YouTube links:
        - Ingested the ~30min "30 Rock Cold Opens" video: 894 sampled frames ->
          1,403 unlabeled faces in library.db.
        - Labeled 4 recurring people (168 of the 1,403 faces) via a greedy
          similarity-cluster script rather than clicking through label.html by
          hand — same cosine-similarity mechanism the UI's "similar faces" panel
          uses, just scripted. label.html/review.html themselves are unexercised
          by a real browser; only their backing API routes are confirmed live.
        - Scored the second ~42min "30 Rock" video: 1,266 sampled frames ->
          2,154 detected faces, 355 matched a labeled person, 1,799 Unknown.
          Visually spot-checked several matches and near-miss Unknowns against
          their thumbnails — see the FAMILIARITY_THRESHOLD design note above.
    - Confirms the common.py:_extract_embedding parsing (512-dim eyepop.face-id.base
      tensor) and the full ingest -> label -> score -> report loop work against
      real staging data.
    - Still unverified: the /review confirm-into-library flow and label.html's
      assign/unassign calls, both only exercised via curl in an earlier session,
      not through the actual browser UI.

Status (2026-08-07, later same day):
    - Reset everything (reset.py, new) and re-ingested the same ~30min video after
      two fixes: 1080p instead of 360p (yt-dlp format-string bug, see design notes),
      and eyepop.face-id.large instead of .base. Fresh run: 894 sampled frames ->
      1,472 unlabeled faces in 285.5s wall time (229.5s/80% in EyePop calls, avg
      257ms/frame, 36.1s local decode) — logged to logs/ingest.log, which every
      ingest run now appends to for before/after comparison.
    - User has been labeling live in label.html while this work happened (Jack: 17
      faces, Liz: 10 faces, confirmed via db query) — label.html's core assign flow
      is therefore now confirmed working through the real browser UI, not just curl.
    - Rebuilt the /review UI (static/review.html) from a single results-grid page
      into an explicit 3-step flow (see Pipeline above): input -> live progress bar
      -> video playback with canvas-drawn person/face boxes + name-or-Unknown
      label, confirm grid still below it. score.py's report now carries
      person_bbox/face_bbox/video_path/source_width/source_height to support this;
      server.py gained a Range-aware /video/<path> route (see design notes).
    - Mechanically validated the new report shape and /video route (small manual
      script + curl), including confirming face-id.large embeddings are still
      512-dim so nothing downstream needed to change. Not yet validated: the
      overlay actually rendered correctly in a real browser against a full score
      run — the report-shape/plumbing is confirmed, the pixel-perfect box
      alignment on screen is not.

Status (2026-08-07, evening):
    - cosine_similarity switched from normalized cosine similarity to a raw dot
      product. Retuned FAMILIARITY_THRESHOLD (0.45 -> 0.30) and the clustering
      default (0.58 -> 0.40) against real face-vs-centroid and face-vs-face scores
      from the current library (7 labeled people: _hair, Jack, Jenna, Jonathan,
      Liz, Pete, Tracy). See the design notes above for the precision/recall
      numbers behind each value — both were chosen to favor precision (fewer wrong
      matches/merges) over recall, consistent with the earlier "too disparate of
      people clumped together" feedback.

Status (2026-08-08):
    - Multi-person duplicate-embedding bug fixed on EyePop staging. Verified with
      test_duplicate_embeddings.py against both prior reproduction cases (22.0s:
      2 faces, was byte-identical, now distinct with different norms; 12.0s: 3
      faces, was all-identical, now all distinct) — bug did not reproduce either
      time. This unblocks the planned reset.py + full re-ingest (the library's
      940 contaminated faces are pre-fix; a fresh ingest now won't reproduce that
      contamination). FAMILIARITY_THRESHOLD/clustering-default retuning above was
      done against contaminated data with a since-changed similarity formula —
      re-derive both again once the library is rebuilt clean, don't assume 0.30/
      0.40 still hold.
    - Reset + re-ingested the ~30min library video same day: 894 sampled frames ->
      1,472 faces, 287.9s wall time, 78% in EyePop calls — same face count as the
      pre-fix run (expected: the bug never changed how many faces were detected,
      only corrupted some of their embedding values, so total count doesn't
      signal anything either way). Confirmed zero same-frame duplicate-embedding
      contamination this time (every one of 855 multi/single-face frames checked
      clean), versus 940/1,472 (64%) before. Library is back to 0 people, 1,472
      unassigned faces — labeling starts over from scratch.
    - Added the is_good_face_shot quality filter (see design notes) and re-ran
      reset + re-ingest again the same day — by this point real labeling work
      existed again (16 people, 1,116 assigned faces) so a full reset.py wipe
      would have destroyed it; did a narrower reset instead (library.db,
      thumbnails/, reports/ only — left cache/ and downloads/ alone) so the
      results cache (see caching design note) could be reused. Re-ingest needed
      ZERO EyePop calls (confirmed: 0.0s / 0% in the run summary) — every sampled
      frame's raw result was already cached from the prior run, so applying a
      new client-side filter to already-cached data is free. 1,383 of 1,472 raw
      detections survived the filter (89 dropped, 6.0%) in 35.2s wall time,
      entirely local. Library is back to 0 people, 1,383 unassigned faces.
    - score.py switched from centroid to nearest-neighbor matching (2026-08-08,
      later same day) — reported symptom: scoring a video with a labeled "Liz"
      (308 faces) missed her real appearances in the first ~11s. Root cause
      confirmed visually: 5 of 6 random Liz library faces wear glasses; the
      no-glasses frames in question correlated much better with other individual
      no-glasses Liz faces (nearest-neighbor sim ~0.45-1.00, one was an exact
      duplicate of an already-confirmed detection from a prior /review session)
      than with the glasses-dominated centroid (~0.42-0.44) — a centroid gets
      washed out by whichever look is overrepresented once a person has enough
      faces. Measured on the current library at the SAME threshold (0.45):
      centroid recall 80.0%/false-match 0.0% vs nearest-neighbor recall
      93.9%/false-match 0.2% — switched, and moved FAMILIARITY_THRESHOLD's
      default from 0.30 to 0.45 to match (0.30 was tuned for the old
      centroid+since-reverted-metric combination, doubly stale). score.py's
      load_person_centroids -> load_person_embeddings now returns every
      embedding per person instead of one averaged vector; match_person takes
      max(cosine_similarity(...)) across them. Re-scored the same video (cache
      reused, zero EyePop calls): familiar detections 774 -> 884. The two
      hardest early frames (4.0s, 6.0s — no glasses, no prior library match to
      lean on) still read Unknown at 0.4487/0.4253, just under 0.45; both would
      clear a 0.40 threshold (96.2% recall/0.9% false-match per the same sweep)
      — a real precision/recall tradeoff at the margin, not a bug, left at 0.45
      pending user preference.
    - Also that day: forgot to restart server.py after the match_person edit
      above — the running process kept executing the OLD centroid code in
      memory (Python doesn't hot-reload), so the review UI still showed the bug
      even after the fix landed on disk. If a fix "isn't working" after editing
      score.py/server.py/common.py, check whether the server process predates
      the edit (`ls -la <file>` vs the server's start time) before assuming the
      fix itself is wrong.
    - Nearest-neighbor got slow once people had hundreds of faces (reported:
      "Liz Suggestions" — 809 faces, 464 unassigned — taking over a second).
      NOT a fundamental NN-vs-clustering tradeoff (asked about vector
      quantization / IVF-style clustering-then-search, which is the right name
      for that technique, but not needed here) — the actual bug was
      server.py:_person_suggestions doing a Python-level double loop calling
      cosine_similarity() per pair (~375k individual calls) instead of one
      batched matmul. Added common.py:normalize_rows/similarity_matrix
      (L2-normalize once, one matmul for all pairs) and used them in
      _person_suggestions, _similar_faces, and score.py's match_person
      (pre-normalizes each person's matrix once in load_person_embeddings
      instead of per detected face). _cluster_unassigned already did this
      correctly — only these three were unvectorized. Measured: same exact
      results (not approximate), 809x464 pairs 1170ms -> 4.5ms in isolation,
      1170ms -> 24ms end-to-end over real HTTP including DB I/O. If this ever
      needs to scale further (a person with tens of thousands of faces), THEN
      revisit clustering/vector-quantization — reduce each person's gallery to
      K k-means centroids and search those instead of every stored face — but
      that trades exactness for speed and wasn't the right first move here.
    - /review streams progressively instead of blocking on the whole scan
      (2026-08-08). Key insight: score_video's resolve_video() already fully
      downloads the video BEFORE any frame is scanned, so the file is completely
      seekable/playable via /video/ from the very start — only the detection
      data (overlay boxes, confirm grid) is genuinely progressive. score.py's
      score_video gained an `on_update(report_snapshot)` callback, fired once
      immediately after resolve_video() (video_path known, zero detections yet)
      and again after every processed frame (a fresh shallow copy each time —
      matters because JOBS is read from a different thread than the one running
      the job, so mutating the same list in place while another thread
      json-serializes it would race). server.py:_start_job's `work` signature
      changed from `work(progress)` to `work(progress, set_partial)`; JOBS
      gained a `"partial"` key alongside `"result"` — `"result"` only lands at
      completion (unchanged), `"partial"` updates throughout. ingest.py's work
      function ignores the new set_partial param (nothing progressive to show
      for the label UI's ingest flow). review.html's poll loop reads
      `job.status === 'done' ? job.result : job.partial`, and the moment that
      has a video_path it calls setupPlayer() ONCE (sets video.src — calling
      this again on a later poll would reset playback position) and switches to
      step 3; the confirm grid uses appendNewCards() (append-only, since
      detections only grow) rather than a full re-render, specifically so it
      doesn't wipe out a "new person name" the user might be mid-typing in an
      already-rendered card when the next poll tick lands.

Status (2026-08-10):
    - label.html's ingest bar accepts dropped image files directly, no video
      or Browse… picker needed. Works because cv2.VideoCapture already reads
      a still image as a 1-frame capture (confirmed: fps=25, frame_count=1,
      .read() succeeds) — so ingest_video()/scan_video() needed zero changes;
      an image just becomes a "video" with one sampled frame at 0.00s.
      Added common.py:save_uploaded_image (content-addressed by sha1 into
      new shared, gitignored UPLOAD_DIR = uploads/, same cache-by-hash idea as
      download_youtube) and server.py's POST /api/upload-image (body is JSON
      {filename, data} with data base64-encoded — reuses the existing
      _read_json_body plumbing instead of adding multipart parsing to the
      stdlib server). label.html's startIngest() job-polling was factored out
      into runIngestJob(source, interval, statusEl) so both the existing
      Ingest button and the new drop handler share one code path;
      ingestDroppedImages() uploads each dropped image (filtered to
      file.type.startsWith('image/')) then runs it through the same
      /api/ingest job as any other source, sequentially, updating
      #ingest-status per image. Verified end-to-end against the running
      server: uploaded a real thumbnail jpg via /api/upload-image, then
      ingested the returned uploads/<hash>.jpg path through /api/ingest —
      pipeline ran (single frame @ 0.00s, ~1.2s EyePop call) with no errors;
      0 faces landed because the test image was itself a tight face-only crop
      (fails the person-detector-first stage of POP), not a bug in the new
      code path — a real dropped photo has a full person in frame like any
      ingest source already handles.
    - New /watch page: realtime webcam identification, running the same
      Person -> Face -> Face-Embedding pipeline directly in the browser via
      genuine WebRTC (liveIngress/process({ingressId}), per the reference
      pattern in eyepop-react-shell's text_live.js — NOT snapshot-upload
      polling, which was the wrong pattern initially assumed from
      person_pose_live.js and corrected mid-session). This is the first place
      in the project where the browser talks to EyePop directly instead of
      through server.py, and the first place the API key reaches client-side
      JS — done by copying the JS SDK's self-contained browser bundle
      (node_modules/@eyepop.ai/eyepop/dist/eyepop.min.js, confirmed UMD,
      exposes global `EyePop`, bundles its own browser-safe polyfills) into
      static/eyepop.min.js and loading it via a plain <script src> tag, no
      build step. server.py's new _serve_watch() reads static/watch.html and
      string-replaces __EYEPOP_SECRET_KEY__/__EYEPOP_URL__/
      __FAMILIARITY_THRESHOLD__ with real values from common.py at serve
      time (chosen over a separate /api/eyepop-config endpoint, per explicit
      user preference) — the key never sits in a file on disk under static/,
      only in a response body, same trust boundary as everything else here
      (server and browser are the same machine). Added common.py:EYEPOP_URL
      (env var, defaults to https://compute.eyepop.ai — the Python SDK reads
      this implicitly today, this is the first explicit read of it).
      watch.html's Pop definition mirrors common.py's POP exactly but in the
      JS SDK's inline dict shape (`model` field, not `ability`; PopComponentType/
      ForwardOperatorType are just string enums under the hood — "inference"/
      "crop" — so plain string literals work without importing anything from
      the bundle beyond the `EyePop` global). Client-side face
      extraction/quality-filter/dedup logic
      (extractEmbedding/eyeLandmarks/isGoodFaceShot/extractFaces) is a
      line-for-line JS port of common.py's equivalents, including the
      duplicate-embedding-in-a-frame mitigation for the (now-fixed-upstream)
      EyePop bug — kept out of caution since a live camera feed can still hit
      multi-person frames. Matching is a JS port of score.py's
      match_person/load_person_embeddings: new server.py:_people_embeddings()
      and GET /api/people/embeddings return the current dataset's labeled
      people with raw embeddings (no numpy client-side, so this ships plain
      JSON arrays); watch.html normalizes each person's vectors once on load
      (not per frame) and does a manual dot-product loop over Float32Arrays —
      deliberately avoiding the exact unvectorized-per-pair mistake fixed
      earlier in server.py/score.py, even though there's no batched-matmul
      equivalent to reach for in plain browser JS. Overlay drawing
      (resizeOverlay/drawOverlay) is a straight port of review.html's
      pattern — canvas sized to the <video> element's rendered CSS size,
      boxes scaled from result.source_width/height, a staleness window
      (1.5s) clears boxes if results stop arriving. No <video>/canvas
      mirroring (a selfie-style flip was considered and dropped — real
      complexity for a cosmetic touch not asked for; box/label math is
      simplest unmirrored). Verified end-to-end except the actual
      getUserMedia+WebRTC round trip: server routes and template injection
      confirmed live (GET /watch has real key/url substituted, zero leftover
      __EYEPOP_*__ tokens; GET /eyepop.min.js serves the bundle; GET
      /api/people/embeddings returns real per-person embedding arrays from
      the eyepop-team dataset); the served page's inline JS passes
      `node --check`. The camera-permission/live-inference path itself needs
      a real browser and hasn't been click-tested — flag this to the user
      before calling it done.
    - /watch auth was actually broken as first written — confirmed by the user
      pasting the browser's real request: POST
      https://compute.staging.eyepop.xyz/authentication/token with
      {"secret_key": "..."} 404ing. Root cause: the JS SDK copied in
      (@eyepop.ai/eyepop 1.15.3, from eyepop-react-shell's node_modules — an
      old demo repo the user flagged as stale for auth specifically) used a
      now-legacy nested `auth: {secretKey}` option whose /authentication/token
      exchange belongs to a different flow (web-api, long-format secret key
      tied to a persistent/named pop — that's what text_live.js actually uses:
      NEXT_PUBLIC_TEXT_AD_POP_API_URL=web-api.eyepop.ai + a long base64-ish
      key + a real popId, not our short eyp_ transient-pop setup). Checked npm:
      latest published SDK is 3.17.3, a major-version jump from 1.15.3.
      Upgraded static/eyepop.min.js to it. That version's Options gained a
      top-level `apiKey` field ("Defaults to process.env['EYEPOP_API_KEY']" —
      exactly our existing key), and its real exchange
      (confirmed by reading eyepop.index.mjs) is `POST
      {eyepopUrl}/v1/auth/authenticate` with `Authorization: Bearer
      <apiKey>` — matching the eyepop-agent-skills auth skill's documented
      compute-api flow exactly, unlike the old secretKey path. Also: 3.17.3
      dropped liveIngress()/process({ingressId}) entirely — Source grew a
      MediaStreamSource variant (`{ mediaStream }`), so a live webcam is now
      just `endpoint.process({ source: { mediaStream: stream } })`, no
      separate ingress step. Verified for real with a Node script (temp,
      not committed) using the officially published npm package against our
      real EYEPOP_API_KEY and compute.staging.eyepop.xyz: apiKey auth
      succeeded, changePop-equivalent (`pop:` in workerEndpoint options)
      succeeded, and a real frame through our exact Pop returned the SAME
      wire shape our Python code already parses (raw[].tensors[] with
      name="embedding", 512 floats; keyPoints[0].points[].classLabel for eye
      landmarks) — confirming extractEmbedding/eyeLandmarks in watch.html
      needed no changes, only the auth options object and the
      live-ingress-to-process call. watch.html updated: `EYEPOP_SECRET_KEY`
      -> `EYEPOP_API_KEY` (template placeholder renamed to match,
      server.py:_serve_watch updated), `EyePop.workerEndpoint({auth:
      {secretKey}})` -> `EyePopSdk.EyePop.workerEndpoint({apiKey, pop:
      POP})` (note the UMD global itself is also renamed, `EyePop` ->
      `EyePopSdk`, in the new bundle), and liveIngress+process({ingressId})
      collapsed into one `process({source: {mediaStream: stream}})` call.
      The actual getUserMedia+process({mediaStream}) round trip in a real
      browser is STILL untested — Node can exercise the CJS build's network
      calls but not a live MediaStream from a camera; needs a real
      click-test before trusting it end-to-end.
    - /watch identity flickered rapidly between a name and Unknown once the
      user actually click-tested it — expected: every frame was matched
      independently, so any score hovering near FAMILIARITY_THRESHOLD flips
      the label every frame with zero memory across frames. Fixed with two
      changes:
      1. Added a TrackingComponent as a SIBLING of the face detector inside
         the person's CropForward.targets (not nested — see
         eyepop-wiki:tracking-component-gives-stable-trackids-across-frames),
         giving each person a stable trackId across frames
         (maxAgeSeconds=5.0, motionModel="constant_velocity", matching the
         wiki's recommended settings for person tracking). Deliberately
         added ONLY to watch.html's JS POP, not common.py's shared Python
         POP — tracking state lives inside the worker endpoint and is only
         valid across frames delivered sequentially to the SAME session,
         which is true for /watch's one continuous
         process({source:{mediaStream}}) call but NOT true for
         ingest.py/score.py, which upload each sampled frame as a separate
         one-off call (per the wiki's explicit incompatible-case warning) —
         adding it there would add latency for zero benefit.
      2. Added per-trackId identity smoothing: a rolling window (7 frames) of
         raw per-frame matches per track, only switching the DISPLAYED name
         once a majority (>=50%) of the window agrees — a single noisy frame
         gets outvoted instead of flipping the label (common.py/watch.html
         functions: updateTrack/pruneStaleTracks). `extractFaces` now carries
         `person.trackId` (checking `trackingId` too — the wiki flags both
         names appearing across SDK versions) through onto each face dict;
         faces with no trackId yet (e.g. the first frame or two before the
         worker's tracker assigns one) fall back to the raw per-frame match.
      Also surfaced the raw (unsmoothed) score alongside the smoothed one in
      #watch-status (e.g. "Andy 82% (raw Andy 61%)") and via
      console.debug('[watch] track ...') for every update, specifically so
      threshold/window-size tuning can be done from real numbers instead of
      guessing — this was in response to the user asking "should we log to
      figure out more?" before committing to a fix. TRACK_WINDOW_SIZE (7) and
      TRACK_MIN_AGREEMENT (0.5) are the two knobs if the smoothing itself
      needs tuning later (window too short = still flickery, too long = slow
      to recognize someone who just walked up). Not yet re-tested live by the
      user after this change.
    - /watch left its EyePop connection open when the user navigated away via
      the header nav links (to / or /review) — stopWatching() only ran on the
      explicit Stop button, so clicking "Face Library" mid-watch left the
      webcam MediaStream, the live process({mediaStream}) stream, and the
      worker endpoint session dangling until the browser's own page-unload
      teardown, if any. Fixed two ways: (1) both nav <a> tags now call
      stopWatching() via onclick — an unawaited async call in an onclick
      handler still runs its synchronous work (stream.getTracks().stop(),
      resultStream.cancel()) immediately, before the browser follows the
      link, so the webcam is released and the stream cancelled before
      navigation even starts; endpoint.disconnect()'s network round-trip is
      best-effort past that point, same as any page-unload teardown. (2)
      added a `pagehide` listener calling stopWatching() as a safety net for
      every OTHER way of leaving (back button, closing the tab, typing a new
      URL) — pagehide over beforeunload since it fires more reliably across
      browsers, notably iOS Safari. stopWatching() was already idempotent
      (guards every teardown step on the variable being non-null before
      acting), so calling it twice (nav onclick, then pagehide during the
      same navigation) is safe.
    - /watch can now capture a confidently-unknown tracked face straight into
      the current dataset's library, and separately identify a dropped photo
      one-off, both building on the trackId/smoothing work above.
      - Capture: updateTrack() now also returns `captureReady` — true at most
        once per continuous "unknown" streak on a given trackId, and only
        once a FULL window (not just the first frame or two of a brand-new
        track, which would look identical to a real stranger) has settled on
        unknown with majority agreement. Recognizing the track again
        (displayed.uuid becomes non-null) re-arms it, so a later
        still-unknown streak on the same long-lived trackId gets offered
        again rather than only ever once. On captureReady, queueCapture()
        draws the CURRENT <video> frame into an offscreen canvas (sized to
        source_width/height) and crops it to the face bbox — same idea as
        common.py:crop_jpeg, client-side, and done immediately since the
        frame the detection came from is already gone by the time a human
        could click anything. Renders as a card (thumbnail + person dropdown
        + new-name input + Add/Dismiss, same shape as review.html's confirm
        cards) in a new #captures-panel. "Add to library" posts to a new
        POST /api/faces/capture (server.py:_capture_face) — sibling of
        _confirm_detection, but the thumbnail arrives as fresh base64 JPEG
        bytes (written to thumbnail_dir() here) rather than referencing a
        file review.html's scan already wrote to disk. Verified live against
        the real server: posted a real thumbnail + a synthetic 512-float
        embedding, got back a real face/person uuid, confirmed it via
        GET /api/people, then cleaned up the test person/face/thumbnail.
        After a successful add, loadPeople() re-fetches so the newly labeled
        face is recognized by the SAME live session from then on — no
        restart needed. pendingCaptures resets on stopWatching() (paired
        with the existing tracks.clear()).
      - Identify: separate from capture — a one-off "drop a photo here to
        identify" zone (#identify-dropzone, same drag-over pattern as
        label.html's ingest bar) for checking a single still image against
        the library without opening the webcam. identifyPhoto() reuses
        extractEmbedding/eyeLandmarks/isGoodFaceShot/extractFaces/matchPerson
        unchanged (a still image needs zero tracking/smoothing — there's
        only one frame). Sends the dropped File directly via the JS SDK's
        FileSource (`process({source: {file}})` — confirmed present in the
        3.17.3 Source union: FileSource | StreamSource | PathSource |
        UrlSource | AssetUuidSource | MediaStreamSource) rather than
        base64-uploading it through our own server first. Reuses the live
        `endpoint` if watching is already running; otherwise opens a
        short-lived one just for this call and disconnects it afterward.
        Extracted drawOverlay's per-detection box+label drawing into a
        shared drawDetections(ctx, detections, scaleX, scaleY) so the live
        loop and this one-shot result render through the same code instead
        of duplicating the styling. Draws onto its own #identify-img +
        #identify-overlay pair (not the live <video>/#overlay pair, which
        stays dedicated to the webcam). The FileSource upload path itself is
        UNTESTED beyond type-checking — same caveat as the rest of this
        session's browser-only pieces, needs a real click-test.
    - Capture UX was too slow (one dropdown pick per card) and the cards
      piled up enough to crowd the video — user feedback: "we have a degree
      of confidence on who the unknown faces are, default the names" and "the
      faces overtake the webcam viewer", then escalated to "auto add if we
      have enough signal." Landed as a three-tier response per confidently-
      unknown-tracked-face, computed once per unknown streak in updateTrack:
      1. **Auto-add, no human step at all**: bestMatch(embedding) (new — same
         nearest-neighbor loop as matchPerson but WITHOUT the threshold
         cutoff, since a below-threshold nearest match is still real evidence
         for this purpose even though it doesn't clear the bar for live
         display) is tallied across the window same as the display vote.
         If one person wins with agreement >= AUTO_ADD_MIN_AGREEMENT (0.7 —
         stricter than the 0.5 used for the display vote, since silently
         writing into someone's collection deserves more consistency
         evidence than just changing an on-screen label) AND their
         window-avg score >= AUTO_ADD_MIN_SCORE (0.35 — deliberately below
         FAMILIARITY_THRESHOLD's 0.45, since that's the whole point: catch
         the "almost recognized" cases live-display correctly hides but
         which are still good enough to add as another exemplar, and any
         mislabel is easy to fix later in label.html), autoAddFace() crops
         the frame and POSTs straight to /api/faces/capture with that
         person_uuid — no card. Logs to console.debug AND a small persistent
         #auto-add-log (last 5, "Auto-added to X (39%) — 3:41:02 PM") so
         silent decisions stay visible, continuing the "log so we can see
         real numbers" thread from the smoothing work above.
      2. **Manual card, pre-filled**: below the auto-add bar but there's
         still a below-SUGGEST_MIN_SCORE (0.15) usable nearest guess — the
         capture card's person <select> defaults to that guess (not
         "Unassigned"), with "Best guess: NAME (23%)" shown above it, so
         confirming is one click on "Add to library" instead of hunting
         through the dropdown every time.
      3. **Manual card, unassigned**: no usable guess at all (score below
         SUGGEST_MIN_SCORE, or an empty library) — falls back to the
         original "Unassigned" default.
      "Overtaking the webcam viewer" fixed three ways, independent of the
      auto-add tier split (which already reduces card VOLUME by silently
      absorbing the common case): capped #captures-list to max-height:280px
      with its own scroll (the panel itself can never grow the page or push
      the video around, no matter how many cards accumulate); a
      cosineSim() dedup check in queueCapture skips creating a new card if
      an existing pending one is >90% similar (the same unresolved physical
      face reappearing after a brief track reset shouldn't spawn a second
      card); and a hard MAX_PENDING_CAPTURES=8 cap (oldest dropped first).
      matchPerson is now a thin wrapper around bestMatch (threshold check
      only); added matchAndGuess for the live path, which needs both the
      thresholded display decision AND the raw guess in one call rather than
      two separate passes over `people` per face per frame. Extracted
      cropCurrentFrame(faceBbox) out of the old queueCapture body so
      autoAddFace and queueCapture share the identical frame-grab/crop code.
      Not yet re-tested live by the user after this change — in particular
      AUTO_ADD_MIN_SCORE/AUTO_ADD_MIN_AGREEMENT are first-guess numbers, not
      calibrated against real data the way FAMILIARITY_THRESHOLD was; watch
      the #auto-add-log against reality and adjust if it's adding wrong
      matches or missing obvious ones.

Status (2026-08-11):
    - Productionized this into a clone-and-run reference design, per explicit
      user request ("what's needed to turn this into a public facing
      Reference design other devs can use") followed by a numbered list of
      answers once the two possible bars (clone-and-run vs. actually-hosted-
      for-strangers) were disambiguated — user confirmed clone-and-run.
    - Deleted plan.md (the original spec, stale vs. what got built) and the
      bug-repro harness (test_duplicate_embeddings.py, bug_two_faces_*) — the
      upstream duplicate-embedding bug is confirmed fixed, so this was dead
      weight for a new reader; removed their now-unneeded .gitignore entries
      too.
    - /watch no longer ships the raw EYEPOP_API_KEY to the browser AT ALL —
      user pushback: "there is a way to use the auth key server side to get a
      temp token for the browser." Added common.py:mint_eyepop_token(),
      confirmed live against real staging: POST
      {EYEPOP_URL}/v1/auth/authenticate with Authorization: Bearer <API_KEY>
      (same exchange the SDKs use internally for apiKey auth — read straight
      out of eyepop.index.mjs to get the exact request shape right) returns
      {access_token, expires_in, ...}; observed expires_in ~78,478s (~21.8h)
      on staging — comfortably longer than any real /watch session, so no
      client-side refresh loop was built, just a >5-minutes-from-expiry server
      cache to avoid re-minting on every page load. New GET
      /api/eyepop-token; server.py:_serve_watch no longer injects
      __EYEPOP_API_KEY__ at all (only __EYEPOP_URL__ and
      __FAMILIARITY_THRESHOLD__ remain template-injected). watch.html's two
      workerEndpoint() call sites now pass `accessToken: await
      getAccessToken()` instead of `apiKey`. Verified live: GET /watch has
      zero occurrences of the "eyp_" key prefix; GET /api/eyepop-token
      returns a real usable JWT.
    - Replaced the macOS-only native file picker (osascript "choose file",
      /api/pick-file, server.py:_pick_file) with a cross-platform upload —
      user asked directly: "Is there a better option that is cross machine
      compatible?" The Browse… button is now a standard `<input
      type="file">` that uploads through the SAME base64-JSON path already
      built for drag-and-drop images (generalized: common.py's
      save_uploaded_image -> save_uploaded_file, /api/upload-image ->
      /api/upload-file, both now filename/content-agnostic — cv2.VideoCapture
      already handles a still image as a 1-frame video and a real video
      identically, so there was never a need for two code paths). Drag-and-
      drop's file filter widened from images-only to
      images-or-videos (label.html: ingestDroppedImages -> ingestDroppedFiles).
      Verified live: uploaded a real (truncated, for speed) .mp4 through
      /api/upload-file, got back a valid uploads/<hash>.mp4 path.
    - Pinned eyepop>=3.17.0,<4 in requirements.txt (was unpinned) — direct
      consequence of this session's earlier 1.15.3->3.17.3 JS SDK break;
      capped below the NEXT major bump on purpose, with a comment pointing
      at this file and static/eyepop.min.js as the two things to check
      together before raising the cap. Also pinned python-dotenv,
      opencv-python, numpy, yt-dlp to the versions actually verified working
      in this environment.
    - Added README.md (quickstart, the three pages, how /watch's token
      exchange works and why, the JS-SDK-bundle-refresh command, known
      limitations), .env.example, LICENSE (MIT — no repo-wide license
      exists yet in abilities_hub, and the root README already frames every
      example as "copy-paste deployable," so MIT per-example is the
      pragmatic default), and CALIBRATION.md (every tunable threshold
      pulled out of CLAUDE.md's narrative into one skimmable place: what it
      controls, its current value, and how to re-derive it — CLAUDE.md
      remains the full dated design log, linked from README as the deep-dive
      rather than replaced).
    - Deliberately NOT changed: SQLite, the stdlib HTTP server, no auth on
      the app itself, single-user/single-machine trust model. These are
      fine for "clone and run on your own machine with your own key" and
      explicitly called out as NOT fine for actually hosting this on a
      network in README's "Known limitations" — that's a materially
      different (and not-yet-scoped) bar, per the disambiguating question
      asked before starting this work.
    - Full syntax/smoke-test pass after all of the above: `ast.parse` on
      every .py file, `node --check` on every page's extracted inline
      script, and a live server hit against /, /review, /watch,
      /eyepop.min.js, /api/datasets, /api/people, /api/eyepop-token, and
      /api/people/embeddings — all 200, zero errors in the server log.
      Nothing has been committed to the face-fam branch yet (one prior
      commit, "Add Face-Familiarity reference design," predates everything
      from this session) — commit on explicit request per this project's
      standing git-safety convention.
