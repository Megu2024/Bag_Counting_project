# Bag Counting – Loading Bay CV Prototype

This repository contains a **computer vision prototype** that counts how many **sacks are loaded into a lorry** from CCTV-style video.  
The system is designed to match the “bag counting” analytics shown in the Aivilon Tech demo videos.

The core idea:

- Detect workers and bags in each frame using **YOLOv8**.
- Approximate the **sack on the worker’s head** from the upper part of the person bounding box.
- Track each worker/sack over time using YOLO’s built‑in tracking.
- Define a **virtual counting line** in the image that represents the loading boundary near the truck.
- Count each sack **once**, exactly when its centre crosses this line **in the correct direction** (towards the lorry).
- Generate an **annotated output video** with:
  - The counting line drawn in green.
  - A box only around the **sack region** (not the full person) once the sack has crossed the line.
  - A text overlay showing `Bags counted (crossed line): N`.

---

## 1. Repository structure

- `bag_counter.py` – main script with detection, tracking, line‑crossing logic and video writing.
- `requirements.txt` – Python dependencies (`ultralytics`, `opencv-python`).
- `outputs/` – folder where annotated result videos are written (created automatically).
- *(Input videos are not committed to the repo; reviewer should place them locally and point `--source` to them.)*

---

## 2. Approach in brief

1. **Detection & tracking**

   - Uses `ultralytics` YOLOv8 (`yolov8s.pt`, COCO weights).
   - Runs `model.track(..., persist=True)` so each physical worker/sack pair receives a **stable track ID** across frames.
   - Relevant classes:
     - COCO bag classes: `backpack`, `handbag`, `suitcase`.
     - `person` class: used as a proxy when sacks are carried on workers’ heads.

2. **Sack region approximation**

   - In the assignment clips, sacks are typically carried on the workers’ heads.
   - For detections of class `person`, the code takes only the **upper ~45% of the person bounding box** and treats that region as the “bag”:
     - This avoids drawing a box around the whole body.
     - Works reasonably well for top‑carried sacks without needing a custom “sack” detector.

3. **Virtual counting line**

   - A configurable line models the **boundary of the lorry** (e.g. just at the edge of the truck).
   - The line is defined in **normalised coordinates** between 0 and 1:
     - `(line_x1, line_y1)` – start point
     - `(line_x2, line_y2)` – end point  
       where `0` is left/top and `1` is right/bottom of the frame.
   - The script automatically decides if the line is **vertical** or **horizontal** and applies directional logic accordingly.

4. **Directional crossing logic**

   For each tracked sack centre:

   - If the line is **vertical**:
     - Only counts when the centre moves from **right of the line** to **left of the line** (i.e. towards the truck).
     - Movement left → right is ignored, so workers walking back are not double‑counted.
   - If the line is **horizontal**:
     - Only counts when the centre moves from **above** the line to **below** the line.

   Each track ID is added to a `crossed_ids` set once; the **global counter increments only once per track**.

5. **Output visualisation**

   - Green line = loading boundary.
   - Yellow box = the **sack/head region** of any track that has crossed the line (or all sacks if `only_draw_crossed` is set `False` in code).
   - Top‑left overlay text shows `Bags counted (crossed line): N`.
   - Output video is written as `<input_name>_bag_counted.mp4` into the `outputs/` folder.

---

## 3. Setup

### 3.1. Environment

Requires **Python 3.9+**.

git clone <this-repo-url>
cd <this-repo-folder>

pip install -r requirements.txt
