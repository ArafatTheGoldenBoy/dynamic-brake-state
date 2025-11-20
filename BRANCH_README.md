# Branch-Specific Overview

## What
This branch layers a two-camera traffic-light perception stack onto `dynamic_brake_state.py`, sharing a single YOLO model between a wide primary RGB camera and an optional narrow-FOV telephoto rig. The wide view crops the upper 20% of each frame to focus on traffic lights within roughly 80% of the image, while the telephoto path supports digitally zoomed crops plus depth-backed distance estimation so far TLs can still be detected.

## Why
The default CARLA single-camera setup struggles with distant lights and wastes GPU memory when running multiple YOLO instances. By coordinating a wide/telephoto pair, we prioritize trustworthy <50 m detections without starving longer-range context, leading to smoother braking decisions and fewer false positives.

## How
1. **Camera orchestration** – `dynamic_brake_state.py` subscribes to the wide RGB stream and, unless `--no-telephoto` is set, the telephoto RGB/depth pair. The wide feed is always processed; the telephoto detector is throttled via `--telephoto-stride` to every N frames.
2. **YOLO sharing** – Both camera helpers reuse a single YOLO model instance, preventing redundant GPU allocations.
3. **Primary-first logic** – The perception loop crops the wide view (top 20%), runs YOLO, and only trusts detections with estimated distances <50 m. If such a candidate exists, we record its source as `wide` and skip telephoto processing for that frame.
4. **Telephoto fallback** – When the wide camera lacks a qualified TL, the telephoto helper crops/optionally digit-zooms the upper-center region, remaps YOLO boxes back into the original telephoto coordinates, and estimates distance using depth pixels or pinhole geometry. Telephoto inference can be digitally magnified via CLI flags or config constants.
5. **HSV/Tiny-CNN color classification** – Whichever camera supplies the final TL bbox, an HSV-based classifier inspects the ROI to label it red/yellow/green/unknown. The output feeds a small temporal smoother so sudden flickers do not jerk the braking state machine.
6. **Braking integration** – The chosen TL state (color + smoothed confidence + source camera + distance) propagates into the braking logic for adaptive stopping behavior.
7. **Dual in-game views** – The pygame window now shows the primary wide RGB feed on the left and the telephoto feed (resized to match) on the right; the old top-down camera view has been removed to focus on the two relevant perspectives.
8. **HUD logging inside pygame** – The distance readout that previously lived in OpenCV windows now renders directly on the main pygame HUD overlay, keeping the depth camera logic intact while eliminating extra pop-up windows.
9. **Explicit shutdown reasons** – The console now prints why the loop exited (e.g., ESC, pygame window close, sensor queue timeout, or CARLA tick failure) before sensors/world cleanup, making surprise stops easier to diagnose.

## Configuration checklist
- `--no-telephoto`: disable the telephoto branch entirely.
- `--telephoto-stride N`: only run telephoto YOLO every N frames (default 3).
- `--telephoto-zoom FACTOR`: enable digital zoom for the telephoto helper; FACTOR > 1 crops tighter before resizing back to the detector resolution.
- `--telephoto-compute-log PATH`: write a CSV summary containing total YOLO compute time for the primary stream, the telephoto helper, plus cache hits/stride skips so you can compare "with telephoto" vs "without telephoto" runtime cost after the drive.
- `dynamic_brake_state.py` constants (search for `TELEPHOTO_DIGITAL_ZOOM_*`) fine-tune crop ratios and zoom placement.

Each compute-log row captures the telephoto toggle state, stride/zoom, total call counts, cumulative milliseconds for both camera paths, cache hits, and skipped frames. The last two columns already report the rolled-up milliseconds with and without the telephoto assist applied; check those to quantify the savings from throttling or disabling the helper entirely.

## Testing
Run `python -m compileall dynamic_brake_state.py` or your preferred CARLA scenario playback. For live validation, launch CARLA with both cameras enabled and observe the logged source/ distance entries whenever traffic lights enter/leave the <50 m window.
