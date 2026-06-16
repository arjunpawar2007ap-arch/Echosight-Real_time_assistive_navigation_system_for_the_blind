# EchoSight - Real-Time Assistive Navigation for the Visually Impaired

A camera-based navigation aid that detects nearby obstacles and announces them through prioritized voice alerts. Designed to provide supplementary environmental awareness for visually impaired users in indoor settings.

> **Not a medical device.** EchoSight is supplementary awareness only - not a replacement for a cane, guide dog, or any primary mobility aid.

---

## How it works

Each video frame goes through a five-stage pipeline:

```
Camera frame → YOLOv8 detection → Distance estimation → Priority scoring → Throttled TTS output
```

1. **Detect** - YOLOv8n identifies objects in the live camera feed (80 COCO classes)
2. **Estimate distance** - bounding-box height + calibrated focal length + per-class real-world height priors → approximate distance in meters
3. **Classify zone** - bounding-box x-center maps to left / center / right
4. **Score urgency** - `(1 / distance) × zone_weight × hazard_weight`; center zone and moving hazards (people, cyclists, animals) weighted up
5. **Announce** - top-scoring object only, with throttling (3s cooldown per class+zone), flicker tolerance (5-frame miss buffer), and danger-zone override for immediate center hazards

---

## Stack

| Component | Library |
|---|---|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Computer vision | OpenCV |
| Text-to-speech | pyttsx3 (fully offline) |
| Language | Python 3.9+ |

---

## Configuration

All thresholds live at the top of `nav.py`:

| Variable | Default | What it controls |
|---|---|---|
| `FOCAL_LENGTH_PX` | 800.0 | Camera focal length from calibration |
| `DANGER_DIST_M` | 1.0 | Below this (center zone) → urgent alert, short cooldown |
| `NEAR_DIST_M` | 3.0 | Beyond this → object ignored entirely |
| `COOLDOWN_S` | 3.0 | Min seconds between repeated announcements of same object+zone |
| `MISS_FRAMES_BEFORE_FORGET` | 5 | Frames an object can be missing before state resets |

Object real-world heights and hazard weights are also editable dicts in `nav.py` - extend them for any class you encounter that isn't covered well.

---

## Project structure

```
echosight/
├── nav.py           # main detection + audio pipeline
└── README.md
```

---

## Known limitations

These are real tradeoffs, documented honestly:

- **Angle sensitivity** - objects viewed from non-frontal angles have different apparent heights, shifting distance estimates
- **Partial occlusion** - if only part of an object is visible (person behind a table), the bounding-box height is wrong and distance is overestimated
- **Class-internal size variance** - COCO's "bottle" spans 200ml to 2L; distance error scales with the size difference from the assumed height
- **No persistent object tracking** - two chairs in the same zone are treated as one; a detection flickering in/out still occasionally re-triggers
- **Indoor focus** - outdoor use (streets, crossings, traffic) is explicitly out of scope and unsafe for navigation

---

## Distance estimation - how the math works

```
distance_m = (real_height_m × focal_length_px) / pixel_height
```

**Getting focal length via calibration:**

```
focal_length_px = (pixel_height × known_distance_m) / real_height_m
```

Place a known object at a known distance, measure its pixel height, and solve. The result is specific to your camera and resolution - recalibrate if you change either.

Distance accuracy is sufficient for three actionable buckets (immediate / near / far). Exact meter values aren't announced to the user - only natural-language buckets ("close", "ahead") - which absorbs most estimation error gracefully.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection backbone
- [COCO dataset](https://cocodataset.org/) for pretrained weights covering 80 indoor/outdoor classes
