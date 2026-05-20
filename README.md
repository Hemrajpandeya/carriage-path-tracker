# Carriage Path Tracker

Tracks one or more carriages in top-view tank videos and writes an annotated
video + overlay image. Designed for the seal-whisker RL rig, where:

- The **front** carriage carries a cylinder that generates wakes.
- The **back** carriage carries the whisker array.
- Full reward is given when the back carriage is within ±80 mm of the
  cylinder's path; reward ramps linearly to 0 between 80 mm and 180 mm.

The script visualises both the trails and the reward zone.

## Install

```powershell
python -m pip install opencv-contrib-python numpy
```

`opencv-contrib-python` is required for the CSRT tracker — the plain
`opencv-python` package does **not** include it. They conflict, so uninstall
the plain one first if it's already installed:

```powershell
python -m pip uninstall -y opencv-python
python -m pip install opencv-contrib-python numpy
```

## Portability — running on a new machine

`track_carriage.py` is **fully self-contained**. The only imports are
`argparse`, `os`, `sys` (Python standard library) plus `cv2` and `numpy`.
To get going on a fresh machine:

```powershell
# 1) Make sure Python 3.x is installed
python --version

# 2) One-time pip install
python -m pip install opencv-contrib-python numpy

# 3) Copy track_carriage.py to the new machine and run it
python track_carriage.py "path/to/video_or_folder"
```

No DLLs, no helper scripts, no other files in this folder are needed.

**Caveat:** the default `--px-per-mm 0.4644` was measured for **this** rig.
If you move to a different camera position, tank, or lens, that number will
be wrong. Re-derive it once with `--calibrate` (click two reference points
of known distance), then pass the value explicitly via `--px-per-mm N` for
future runs.

## Quick start

### One video

```powershell
python track_carriage.py "pathtracking_using_openCV/IMG_5884-Seg03-down_good.MOV"
```

What happens:

1. A window opens on the first frame (rotated to landscape).
   Drag a tight box around the **front carriage** → press **Space/Enter**.
   Drag a box around the **back carriage** → **Space/Enter**.
   Press **Esc** to start tracking.
2. The script tracks every frame and writes two files next to the input:
   - `<stem>_tracked.mp4` — annotated video
   - `<stem>_overlay.png` — first frame with both full trails drawn on it

### A folder of videos (batch mode)

```powershell
python track_carriage.py "path/to/folder"
```

- Finds every `.mp4`/`.mov`/`.avi`/`.mkv` in the folder (top level only,
  case-insensitive, alphabetical order).
- Creates `path/to/folder/tracked_videos/`.
- Draws ROIs **once** on the first video (or use `--bboxes` to skip that
  step entirely — see below). Reuses the same boxes on every video.
- Writes `<stem>_tracked.mp4` and `<stem>_overlay.png` for each input into
  `tracked_videos/`. Existing files are overwritten.

### Skip the ROI picker

If you already know the boxes, pass them on the command line. Front first,
back second, semicolon between them, four comma-separated ints each
(`X,Y,W,H`):

```powershell
python track_carriage.py "video.MOV" --bboxes "1550,522,58,20;1765,465,30,134"
```

This skips the picker entirely. Works the same way in batch mode — the
boxes apply to every video in the folder.

## What you'll see in the output

- **Green trail** — cylinder (front carriage) path.
- **Magenta trail** — whisker (back carriage) path.
- **Yellow tinted band** — ±80 mm full-reward zone around the cylinder path.
- **Dashed yellow outline** — 180 mm ramp boundary (reward fades from 1 to 0).
- **Top-left status text** — `IN ZONE` / `PARTIAL` / `OUT` based on where
  the back carriage is right now.

## All CLI flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `video` (positional) | — | File path **or** folder path |
| `--bboxes "X,Y,W,H;X,Y,W,H"` | (interactive) | Skip ROI picker; front first |
| `--labels front,back` | `front,back` | Label names matching ROI order |
| `--start N` | `0` | Start tracking from frame N |
| `--trail N` | `5` | Trail line thickness in pixels |
| `--crop-right N` | `80` | Pixels to remove from the right edge of the **original portrait** frame (before rotation) |
| `--reward-mm 80` | `80.0` | Full-reward half-width in mm |
| `--ramp-mm 180` | `180.0` | Distance at which reward decays to 0 |
| `--ref-mm 1600.2` | `1600.2` | Known real-world distance between calibration points (63 in default) |
| `--px-per-mm 0.4644` | `0.4644` | Pixels per mm (measured for this rig) |
| `--calibrate` | off | Pop a window to re-derive px/mm by clicking two known points |
| `--no-reward` | off | Disable the reward overlay (tube + dashed outline) |
| `--rotate cw\|ccw\|180\|none` | `cw` | Rotate the output (default 90° clockwise so portrait input becomes landscape) |
| `--fps N` | `24` | Output video fps (capped at source). Tracker still processes every input frame; only the write cadence changes. Useful for PowerPoint embedding. |
| `--scale F` | `0.333…` | Downscale the output by this factor. Default 1/3 sizes a 1920×1080 working frame to 640×360, which is a perfect cell in a 3×3 grid on a 16:9 PowerPoint slide. |

## Common workflows

### "I just want to track one video"

```powershell
python track_carriage.py "video.MOV"
```

Draw two boxes, hit Esc.

### "I want to batch a whole folder using the same boxes"

```powershell
python track_carriage.py "C:/path/to/folder" --bboxes "1550,522,58,20;1765,465,30,134"
```

### "The calibration is slightly off for my recording"

```powershell
python track_carriage.py "video.MOV" --calibrate
```

In the calibration window, click two reference points whose real-world
distance is 63 in (1600.2 mm). Press **Enter** to confirm, **R** to reset
your clicks, **Esc** to skip.

### "My camera was set up differently — change the calibration distance"

```powershell
python track_carriage.py "video.MOV" --calibrate --ref-mm 1000
```

The two points you click will be treated as 1000 mm apart.

### "I want thicker trails / a different reward radius"

```powershell
python track_carriage.py "video.MOV" --trail 8 --reward-mm 100 --ramp-mm 200
```

### "I want to skip the first few seconds"

```powershell
python track_carriage.py "video.MOV" --start 120
```

Starts at frame 120. The ROI picker opens on **that** frame.

### "I just want trails, no reward zone"

```powershell
python track_carriage.py "video.MOV" --no-reward
```

## Tips for good tracking

- Draw **tight** boxes. A loose box around the carriage usually pulls the
  tracker onto neighbouring rails or shadow.
- The ROI window is **resizable**. Drag the corners or hit maximize to zoom
  in for pixel-precise box placement.
- If a tracker drifts mid-video, smaller boxes around a distinctive feature
  (a screw head, a bright corner) usually beat large generic ones.
- In batch mode, if the carriage starting positions vary a lot between
  videos, you may need to drop `--bboxes` and pick per video — or use
  larger initial boxes so CSRT's search window catches the carriage even
  if it's slightly off the seeded position.

## Coordinate system note

`--bboxes` coordinates are interpreted in the **working frame**, which is
what you see in the ROI window — i.e. after `--crop-right` is applied to
the original portrait frame and after `--rotate` is applied. If you change
either flag between runs, the same coordinates will no longer line up.

## Outputs

For a video named `IMG_5884-Seg03-down_good.MOV`, you get:

```
IMG_5884-Seg03-down_good_tracked.mp4     annotated video, landscape
IMG_5884-Seg03-down_good_overlay.png     first frame with both full paths drawn
```

In batch mode, both files land in `<folder>/tracked_videos/`.
