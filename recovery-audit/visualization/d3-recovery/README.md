# Data recovery and research reproduction animation

`recovery.mp4` is a 15-second, 1920×1080, 30 fps H.264/yuv420p video without audio. `initial.png` and `final.png` are stills. All 12 study notebooks, 43 file nodes and 75 baseline dependency edges are included. Same-date notebooks overlap intentionally on the fixed date axis.

## Repeat the export

From this directory:

```sh
npm ci
python3 build.py
node export.cjs
```

Chromium uses `/usr/bin/google-chrome`. FFmpeg is a pinned project-local binary. The export starts its own loopback server, renders explicit frame times, and closes the browser/server afterward. Dependencies and intermediate frames are ignored. To preview, run `python3 -m http.server 8000 --bind 127.0.0.1` here and open http://127.0.0.1:8000. The preview initially displays the baseline; browser console `renderAt(seconds)` selects a video time.

## Interpretation and evidence

A left-aligned step label appears at the timeline start: “Step n: data recovery” or “Step n: notebook reproduction.” Each event is shown for about 0.7 seconds. Changed nodes get a yellow halo that appears at 25% larger than the node diameter, then expands and fades over 0.7 seconds; its inner diameter remains masked at the node’s outer edge. File nodes change on documented availability observations; notebook nodes change on recorded completed executions. Dependency arrows point from source to target and stop at node boundaries. Arrowhead and edge colors follow their source. The bottom caption uses five circle symbols: red outline/white fill for missing; green outline/white fill for recovered; green outline/green fill for reproduced; gray outline/white fill for initially present; and amber outline/white fill for unknown. Unresolved identities remain amber. No file contents or filenames appear in the video.

`data.json` supplies identities and 20 ordered event batches with evidence references and timing qualifications. `evidence.json` contains input hashes. `layout.json` contains the seeded, settled force positions. Notebook positions use filename dates between September 2023 and January 2025; input/output positions are constrained above/below the axis. Colors animate; positions do not.

Recovery event timestamps are often first recorded observations, not exact actions. Where a completed execution manifest was unavailable in a recognizable format, the commit of a promotion report establishes an upper bound for completion. Thus this is an evidence-backed observed sequence, not a claim of exact wall-clock recovery chronology. Output availability observations are distinguished from input data recovery events. Unresolved output paths are never assumed regenerated merely because a notebook ran. Green notebook nodes indicate completed execution; accepted failures caught internally, convergence warnings and target differences remain documented in the underlying reports.

Existing Graphviz files and immutable baseline evidence are preserved. Work is local on visualization/d3-recovery; no commit, hosting or posting is performed.
