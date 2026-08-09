Name: Face Familiarity Scoring on EyePop.ai (Reference Design)
Goal: Score how familiar a face in a video is against a dataset of previously seen people.

Pipeline (see plan.md for the original spec):
    - Pop: eyepop.person -> eyepop.person.face.short-range -> eyepop.face-id.large
      (person crop -> face crop -> 512-dim face embedding). Defined once in common.py.
      Was eyepop.face-id.base until 2026-08-07 — swapped for the larger embedding
      model; still 512-dim, no downstream schema changes needed.
    - ingest.py samples a video every SAMPLE_INTERVAL_SECONDS, runs the Pop on each
      sampled frame, and stores one unlabeled face (thumbnail + embedding) per
      detection in library.db (SQLite) — skipping any face that fails
      common.py:is_good_face_shot (no eyes visible, or too small; see design
      notes). Appends per-frame timing (video-decode ms vs. EyePop round-trip ms)
      and a run summary to logs/ingest.log.
    - The label UI (`/` in server.py, static/label.html) lets a human group
      unlabeled faces (via a per-face "similar faces" nearest-neighbor panel, or
      auto-clustered "Suggested groups") and assign them to a named person —
      rename/delete people, drag-and-drop faces onto a person to assign, and a
      per-person "unassigned faces similar to this person" panel to keep growing
      an existing person's set.
    - score.py samples a new video the same way, then matches each detected face's
      embedding by nearest-neighbor against every labeled person's individual
      stored embeddings (not an average — see design notes) via
      common.py:similarity_matrix. Below FAMILIARITY_THRESHOLD it's reported
      "Unknown" rather than forced into the closest label. Streams progress via an
      `on_update` callback (see design notes) and writes the final JSON report to
      ./reports/, including each detection's person_bbox/face_bbox and the video's
      source_width/height (for the review UI's overlay) and video_path (for
      /video streaming).
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
      PORT are optional overrides (see common.py / server.py for defaults).
    - pip install -r requirements.txt (needs ffmpeg on PATH for some YouTube formats
      via yt-dlp, and requires the eyepop package's own dependencies).

Usage:
    python3.12 server.py                      # http://localhost:8080 (label) and /review (score)
    python3.12 ingest.py <youtube_url_or_path> # CLI equivalent of the ingest job
    python3.12 score.py <youtube_url_or_path>  # CLI equivalent of the score job
    python3.12 reset.py [--yes]                # wipe library.db/thumbnails/downloads/cache/reports
                                                # (logs/ survives on purpose — kept for before/after timing comparisons)

Design notes:
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
