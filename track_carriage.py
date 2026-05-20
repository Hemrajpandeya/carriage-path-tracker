"""
Multi-carriage path tracker with a capsule reward-tube overlay.

The FIRST carriage is the reference / cylinder. Its swept path is tinted as a
solid capsule of radius --reward-mm (full reward zone). Subsequent carriages
get their own trail in a distinct colour.

Usage:
    python track_carriage.py <video>
        -> interactive: draw one box per carriage on frame 1.
           SPACE/ENTER per box, ESC when done. Draw FRONT first.

    python track_carriage.py <video> --bboxes "X,Y,W,H;X,Y,W,H"

Flags:
    --labels front,back     custom label names
    --start N               start tracking from frame N (default 0)
    --trail N               trail thickness in pixels (default 2)
    --crop-right N          remove N px from the right edge (default 80)
    --reward-mm 80          tube half-width in mm (default 80)
    --ref-mm 1600.2         calibration reference distance in mm (63 in default)
    --px-per-mm 0.4644      px-per-mm calibration (default measured for rig)
    --calibrate             click two reference points to recompute px/mm
    --no-reward             disable the tube overlay
"""

import argparse, os, sys
import cv2
import numpy as np


COLORS = [
    (0, 255, 0),     # green   - target 0 (front / cylinder)
    (255, 0, 255),   # magenta - target 1 (back  / whisker)
    (0, 255, 255),
    (255, 255, 0),
]


def make_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT not available. Try: pip install opencv-contrib-python")


def pick_calibration_points(frame, ref_mm):
    """Resizable/maximizable window. Mouse clicks are reported in image space
    by OpenCV regardless of window scaling."""
    pts = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    win = f"Click TWO reference points (known distance = {ref_mm:.1f} mm). ENTER=ok ESC=skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    h, w = frame.shape[:2]
    init_max = 1280
    if max(h, w) > init_max:
        s = init_max / max(h, w)
        cv2.resizeWindow(win, int(w * s), int(h * s))
    else:
        cv2.resizeWindow(win, w, h)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        for p in pts:
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0, 255, 255), 2)
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
            d = (dx * dx + dy * dy) ** 0.5
            ppmm = d / ref_mm
            cv2.putText(disp, f"{d:.1f}px = {ref_mm:.1f}mm  ({ppmm:.4f} px/mm)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(disp, f"clicks: {len(pts)}/2", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(pts) == 2:
            break
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k in (ord('r'), ord('R')):
            pts.clear()
    cv2.destroyWindow(win)
    dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    return ((dx * dx + dy * dy) ** 0.5) / ref_mm


def pick_rois_interactive(frame):
    """Resizable/maximizable window so you can zoom in for precise selection.
    Coordinates returned by selectROIs are in image space regardless of window size."""
    win = "Draw FRONT carriage first, then BACK. SPACE/ENTER per box. ESC when done."
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Initial size: roughly fit a 1280-px-wide screen; the user can still resize/maximize.
    h, w = frame.shape[:2]
    init_max = 1280
    if max(h, w) > init_max:
        s = init_max / max(h, w)
        cv2.resizeWindow(win, int(w * s), int(h * s))
    else:
        cv2.resizeWindow(win, w, h)
    rois = cv2.selectROIs(win, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    if len(rois) == 0:
        sys.exit("No ROIs selected.")
    return [(int(x), int(y), int(bw), int(bh)) for x, y, bw, bh in rois]


def parse_bboxes(s):
    out = []
    for chunk in s.split(";"):
        parts = [int(v) for v in chunk.split(",")]
        if len(parts) != 4:
            sys.exit(f"bad --bboxes chunk: {chunk}")
        out.append(tuple(parts))
    return out


def center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def find_videos(folder):
    """Top-level video files in folder, sorted alphabetically."""
    out = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and name.lower().endswith(VIDEO_EXTS):
            out.append(full)
    return out


def draw_dashed_polyline(img, pts, color, thickness=2, dash=14, gap=10):
    if len(pts) < 2:
        return
    on = True
    remaining = dash
    for i in range(len(pts) - 1):
        ax, ay = float(pts[i][0]), float(pts[i][1])
        bx, by = float(pts[i + 1][0]), float(pts[i + 1][1])
        dx, dy = bx - ax, by - ay
        seg = (dx * dx + dy * dy) ** 0.5
        if seg == 0:
            continue
        t0 = 0.0
        while t0 < seg:
            t1 = min(seg, t0 + remaining)
            if on:
                p0 = (int(ax + dx * t0 / seg), int(ay + dy * t0 / seg))
                p1 = (int(ax + dx * t1 / seg), int(ay + dy * t1 / seg))
                cv2.line(img, p0, p1, color, thickness)
            consumed = t1 - t0
            t0 = t1
            remaining -= consumed
            if remaining <= 0:
                on = not on
                remaining = dash if on else gap


def process_video(video_path, args, out_dir, bboxes=None, px_per_mm=None):
    """Track one video and write outputs into out_dir.
    bboxes/px_per_mm: if None, picked interactively / from args (first video).
    Returns (bboxes_used, px_per_mm_used) so a batch caller can reuse them."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  cannot open {video_path}; skipping")
        return bboxes, px_per_mm
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {n_total} frames @ {fps:.2f} fps, size {W0}x{H0}"
          + (f" -> output @ {min(args.fps, fps):.2f} fps" if args.fps and args.fps > 0 else ""))

    crop_right = max(0, int(args.crop_right))
    if crop_right >= W0:
        print(f"  --crop-right {crop_right} >= width {W0}; skipping")
        cap.release()
        return bboxes, px_per_mm
    post_crop_W = W0 - crop_right

    ROTATE_CODES = {
        "none": None,
        "cw":  cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rot_code = ROTATE_CODES[args.rotate]

    def prep_frame(img):
        """Crop right edge, then rotate. Produces the working-coordinate frame."""
        if crop_right:
            img = img[:, :post_crop_W]
        if rot_code is not None:
            img = cv2.rotate(img, rot_code)
        return img

    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, first = cap.read()
    if not ok:
        print("  could not read first frame; skipping")
        cap.release()
        return bboxes, px_per_mm
    first = prep_frame(first)
    H, W = first.shape[:2]

    # Calibration (only when not already set; happens once for batch on first video)
    if px_per_mm is None:
        px_per_mm = args.px_per_mm
        if args.calibrate and args.show_reward:
            print(f"  calibration: click two points {args.ref_mm:.1f} mm apart...")
            new_val = pick_calibration_points(first, args.ref_mm)
            if new_val:
                px_per_mm = new_val

    # Bboxes (only when not already set)
    if bboxes is None:
        if args.bboxes:
            bboxes = parse_bboxes(args.bboxes)
        else:
            bboxes = pick_rois_interactive(first)

    labels = [s.strip() for s in args.labels.split(",")]
    while len(labels) < len(bboxes):
        labels.append(f"target{len(labels)}")
    labels = labels[: len(bboxes)]

    trackers = []
    for b in bboxes:
        t = make_tracker()
        t.init(first, b)
        trackers.append(t)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    vid_out = os.path.join(out_dir, f"{stem}_tracked.mp4")
    overlay = os.path.join(out_dir, f"{stem}_overlay.png")

    # Output fps: capped at the source rate (don't try to upsample).
    out_fps = min(args.fps, fps) if args.fps and args.fps > 0 else fps
    in_dt = 1.0 / fps if fps > 0 else 0.0
    out_dt = 1.0 / out_fps if out_fps > 0 else 0.0

    # This OpenCV build's avc1 (H.264) path is broken — produces files ~5x
    # larger than mp4v despite still being labelled H.264. Use mp4v directly.
    scale = max(0.05, float(args.scale))
    out_W = max(2, int(round(W * scale)))
    out_H = max(2, int(round(H * scale)))
    # mp4 codecs are happier with even dimensions
    out_W -= out_W % 2
    out_H -= out_H % 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_out, fourcc, out_fps, (out_W, out_H))
    if not writer.isOpened():
        print(f"  could not open writer for {vid_out}; skipping")
        cap.release()
        return bboxes, px_per_mm
    print(f"  codec: mp4v  output: {out_W}x{out_H} @ {out_fps:.2f} fps "
          f"(scale {scale:.2f})")

    def write_frame(img):
        if scale == 1.0 and img.shape[1] == out_W and img.shape[0] == out_H:
            writer.write(img)
        else:
            writer.write(cv2.resize(img, (out_W, out_H),
                                    interpolation=cv2.INTER_AREA))

    reward_on = bool(args.show_reward and px_per_mm and len(trackers) >= 1)
    tube_thick = max(1, int(round(2 * args.reward_mm * px_per_mm))) if reward_on else 0
    ramp_thick = max(1, int(round(2 * args.ramp_mm * px_per_mm))) if reward_on else 0

    # Persistent layers, allocated once
    trail_layer = np.zeros((H, W, 3), dtype=np.uint8)
    trail_present = np.zeros((H, W), dtype=np.uint8)
    tube_mask = np.zeros((H, W), dtype=np.uint8) if reward_on else None
    ramp_mask = np.zeros((H, W), dtype=np.uint8) if reward_on else None

    TUBE_TINT = np.array([60, 220, 220], dtype=np.float32)  # warm yellow (BGR)
    RAMP_OUTLINE = (60, 220, 220)
    ALPHA = 0.45

    trails = [[] for _ in trackers]
    last_drawn = [None for _ in trackers]
    lost_counts = [0] * len(trackers)
    tube_bbox = None  # (x0, y0, x1, y1) — extents that ever had tube content
    ramp_drawn = False  # set True once first cylinder segment is added

    def expand_tube_bbox(p, q, pad):
        nonlocal tube_bbox
        x0 = max(0, min(p[0], q[0]) - pad)
        y0 = max(0, min(p[1], q[1]) - pad)
        x1 = min(W, max(p[0], q[0]) + pad + 1)
        y1 = min(H, max(p[1], q[1]) + pad + 1)
        if tube_bbox is None:
            tube_bbox = (x0, y0, x1, y1)
        else:
            tube_bbox = (min(tube_bbox[0], x0), min(tube_bbox[1], y0),
                         max(tube_bbox[2], x1), max(tube_bbox[3], y1))

    # Seed frame 1
    f0 = first.copy()
    for i, b in enumerate(bboxes):
        cx, cy = center(b)
        pt = (int(cx), int(cy))
        trails[i].append(pt)
        last_drawn[i] = pt
    write_frame(f0)

    pad = max(args.trail, tube_thick // 2 + 1)
    frame_idx = args.start + 1
    input_time = in_dt          # next read frame's timestamp (seed was at 0)
    next_out_time = out_dt      # next time to write an output frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = prep_frame(frame)

        # 1) Update each tracker, append trail point, extend persistent layers
        for i, tr in enumerate(trackers):
            ok2, box = tr.update(frame)
            color = COLORS[i % len(COLORS)]
            if ok2:
                cx, cy = center(box)
                pt = (int(cx), int(cy))
                trails[i].append(pt)
                prev = last_drawn[i]
                if prev is not None:
                    cv2.line(trail_layer, prev, pt, color, args.trail)
                    cv2.line(trail_present, prev, pt, 255, args.trail)
                    if i == 0 and reward_on:
                        cv2.line(tube_mask, prev, pt, 255, tube_thick)
                        cv2.line(ramp_mask, prev, pt, 255, ramp_thick)
                        expand_tube_bbox(prev, pt, pad)
                        ramp_drawn = True
                last_drawn[i] = pt
            else:
                lost_counts[i] += 1
                trails[i].append(None)
                # keep last_drawn so the trail bridges short losses

        # Render + write only at the chosen output cadence. The tracker
        # updates and persistent-layer extensions above happen every frame.
        if input_time >= next_out_time:
            # Tube tint — alpha-blend inside the tracked bbox only
            if reward_on and tube_bbox is not None:
                x0, y0, x1, y1 = tube_bbox
                sub_mask = tube_mask[y0:y1, x0:x1]
                roi = frame[y0:y1, x0:x1]
                m = sub_mask > 0
                if m.any():
                    region = roi[m].astype(np.float32)
                    roi[m] = (region * (1.0 - ALPHA) + TUBE_TINT * ALPHA).astype(np.uint8)

            # Dashed ramp boundary
            if reward_on and ramp_drawn:
                contours, _ = cv2.findContours(ramp_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                for c in contours:
                    pts = c.reshape(-1, 2)
                    if len(pts) > 4:
                        pts = pts[::3]
                    draw_dashed_polyline(frame, pts, RAMP_OUTLINE,
                                         thickness=2, dash=14, gap=10)

            # Composite the persistent trail layer
            tm = trail_present > 0
            if tm.any():
                frame[tm] = trail_layer[tm]

            # Current-position markers
            for i in range(len(trackers)):
                last = last_drawn[i]
                if last is not None:
                    cv2.circle(frame, last, 6, COLORS[i % len(COLORS)], 2)

            # Status text (post-rotation orientation since frame is already rotated)
            if reward_on and len(trackers) >= 2 and last_drawn[1] is not None:
                x, y = last_drawn[1]
                if 0 <= x < W and 0 <= y < H:
                    if tube_mask[y, x] > 0:
                        txt, col = "IN ZONE", (0, 255, 0)
                    elif ramp_mask is not None and ramp_mask[y, x] > 0:
                        txt, col = "PARTIAL", RAMP_OUTLINE
                    else:
                        txt, col = "OUT", (0, 0, 255)
                    cv2.putText(frame, txt, (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            write_frame(frame)
            next_out_time += out_dt

        if frame_idx % 120 == 0:
            print(f"    ...frame {frame_idx}/{n_total}")
        frame_idx += 1
        input_time += in_dt

    writer.release()
    cap.release()

    # Overlay PNG: first frame + final tube tint + final ramp outline + final trails
    canvas = first.copy()
    if reward_on and tube_bbox is not None:
        x0, y0, x1, y1 = tube_bbox
        sub_mask = tube_mask[y0:y1, x0:x1]
        roi = canvas[y0:y1, x0:x1]
        m = sub_mask > 0
        if m.any():
            region = roi[m].astype(np.float32)
            roi[m] = (region * (1.0 - ALPHA) + TUBE_TINT * ALPHA).astype(np.uint8)
    if reward_on and ramp_drawn:
        contours, _ = cv2.findContours(ramp_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) > 4:
                pts = pts[::3]
            draw_dashed_polyline(canvas, pts, RAMP_OUTLINE,
                                 thickness=2, dash=14, gap=10)
    tm = trail_present > 0
    canvas[tm] = trail_layer[tm]
    for i, trail in enumerate(trails):
        clean = [p for p in trail if p is not None]
        if clean:
            cv2.circle(canvas, clean[0], 7, (0, 255, 0), -1)
            cv2.circle(canvas, clean[-1], 7, (0, 0, 255), -1)
            cv2.putText(canvas, labels[i], (clean[0][0] + 8, clean[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[i % len(COLORS)], 2)
    cv2.imwrite(overlay, canvas)

    for lbl, trail, lost in zip(labels, trails, lost_counts):
        tracked = sum(1 for p in trail if p is not None)
        print(f"    {lbl}: tracked {tracked}/{len(trail)} ({lost} lost)")
    print(f"    -> {os.path.basename(vid_out)}")

    return bboxes, px_per_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a video file OR a folder of videos")
    ap.add_argument("--bboxes")
    ap.add_argument("--labels", default="front,back")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--trail", type=int, default=5)
    ap.add_argument("--crop-right", type=int, default=80)
    ap.add_argument("--reward-mm", type=float, default=80.0)
    ap.add_argument("--ramp-mm", type=float, default=180.0,
                    help="outer ramp boundary distance in mm (dashed outline)")
    ap.add_argument("--ref-mm", type=float, default=1600.2)
    ap.add_argument("--px-per-mm", type=float, default=0.4644)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-reward", dest="show_reward",
                    action="store_false", default=True)
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="cw",
                    help="rotate the output (default cw = 90° clockwise)")
    ap.add_argument("--fps", type=float, default=24.0,
                    help="output video fps (default 24; capped at source fps). "
                         "The tracker still processes every input frame; this only "
                         "controls how often a frame is written to the output.")
    ap.add_argument("--scale", type=float, default=1/3,
                    help="downscale the output by this factor (default 1/3 = "
                         "640x360 from 1920x1080, sized for a 3x3 grid on a 16:9 PPT "
                         "slide). Tracking happens at full working resolution; only "
                         "the final write is downscaled.")
    args = ap.parse_args()

    target = args.video
    if os.path.isdir(target):
        videos = find_videos(target)
        if not videos:
            sys.exit(f"no videos found in {target}")
        out_dir = os.path.join(target, "tracked_videos")
        os.makedirs(out_dir, exist_ok=True)
        print(f"batch: {len(videos)} video(s) in {target}")
        print(f"output: {out_dir}")
        bboxes = None
        px_per_mm = None
        for i, v in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {os.path.basename(v)}")
            bboxes, px_per_mm = process_video(v, args, out_dir, bboxes, px_per_mm)
        print("\nbatch complete.")
    else:
        if not os.path.isfile(target):
            sys.exit(f"not found: {target}")
        out_dir = os.path.dirname(os.path.abspath(target))
        print(f"processing {os.path.basename(target)}")
        process_video(target, args, out_dir)


if __name__ == "__main__":
    main()
