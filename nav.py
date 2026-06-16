import time
import threading
import cv2
import pyttsx3
from ultralytics import YOLO


FOCAL_LENGTH_PX = 800.0    

DANGER_DIST_M = 1.0         
NEAR_DIST_M = 3.0           
COOLDOWN_S = 3.0           
MISS_FRAMES_BEFORE_FORGET = 5   


REAL_HEIGHTS = {
    "person": 1.65, "chair": 0.9, "couch": 0.85, "bed": 0.6,
    "dining table": 0.75, "tv": 0.6, "laptop": 0.02, "book": 0.22,
    "bottle": 0.22, "cup": 0.10, "backpack": 0.45, "handbag": 0.30,
    "suitcase": 0.55, "potted plant": 0.45, "refrigerator": 1.7,
    "oven": 0.85, "microwave": 0.30, "toilet": 0.75, "sink": 0.20,
    "dog": 0.55, "cat": 0.25, "bicycle": 1.05,
}

HAZARD_WEIGHTS = {
    "person": 1.5,   
    "dog": 1.3, "cat": 1.3,
    "bicycle": 1.4,
    
}

ZONE_WEIGHTS = {"center": 1.0, "left": 0.5, "right": 0.5}


class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 200)
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self):
        return self._busy

    def say(self, text):
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()
        self._busy = False


def estimate_distance(class_name, pixel_h):
    real_h = REAL_HEIGHTS.get(class_name)
    if real_h is None or pixel_h <= 0:
        return None
    dist = (real_h * FOCAL_LENGTH_PX) / pixel_h
    if dist < 0.2 or dist > 15.0:
        return None
    return dist


def get_zone(x_center, frame_width):
    third = frame_width / 3
    if x_center < third:
        return "left"
    if x_center < 2 * third:
        return "center"
    return "right"


def distance_bucket(d):
    if d < 1.2:
        return "very close"
    if d < 2.0:
        return "close"
    return "ahead"


def urgency_score(distance, zone, class_name):
    return (1.0 / distance) * ZONE_WEIGHTS[zone] * HAZARD_WEIGHTS.get(class_name, 1.0)


def main():
    model = YOLO("yolov8n.pt")
    speaker = Speaker()
    cap = cv2.VideoCapture(0)

    last_announced = {}   
    miss_count = {}       
    print("Running. Press Q in the video window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        results = model(frame, verbose=False)[0]

        candidates = []
        seen_keys = set()
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pixel_h = y2 - y1
            x_center = (x1 + x2) / 2

            distance = estimate_distance(cls, pixel_h)
            if distance is None or distance > NEAR_DIST_M:
                continue

            zone = get_zone(x_center, w)
            score = urgency_score(distance, zone, cls)
            candidates.append({
                "class": cls, "zone": zone, "distance": distance,
                "score": score, "box": (int(x1), int(y1), int(x2), int(y2)),
            })
            seen_keys.add((cls, zone))

        for k in seen_keys:
            miss_count[k] = 0
        for k in list(miss_count.keys()):
            if k not in seen_keys:
                miss_count[k] += 1
                if miss_count[k] > MISS_FRAMES_BEFORE_FORGET:
                    last_announced.pop(k, None)
                    miss_count.pop(k, None)

        for c in candidates:
            x1, y1, x2, y2 = c["box"]
            color = (0, 0, 255) if c["distance"] < DANGER_DIST_M else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{c['class']} {c['distance']:.1f}m {c['zone']}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[0]

            if top["distance"] < DANGER_DIST_M and top["zone"] == "center":
                key = (top["class"], top["zone"])
                if time.time() - last_announced.get(key, 0) > 1.5:  
                    speaker.say(f"{top['class']} immediately ahead")
                    last_announced[key] = time.time()
            else:
                key = (top["class"], top["zone"])
                if (not speaker.is_busy()
                        and time.time() - last_announced.get(key, 0) > COOLDOWN_S):
                    bucket = distance_bucket(top["distance"])
                    if top["zone"] == "center":
                        msg = f"{top['class']} {bucket}"
                    else:
                        msg = f"{top['class']} {bucket}, on your {top['zone']}"
                    speaker.say(msg)
                    last_announced[key] = time.time()

        cv2.imshow("Assistive Nav (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
