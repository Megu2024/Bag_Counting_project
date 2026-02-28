# Bag Counting – Loading Bay Computer Vision Prototype

This repository contains a small computer vision project where I try to **count how many sacks are loaded into a lorry** just by looking at video.  
The idea is to get close to the kind of “bag counting” analytics used in real warehouse and logistics setups.

---

## 1. What this project does

In the videos, workers carry sacks towards a truck.  
I place a **virtual line** near the truck and count how many sacks cross that line in the **loading direction**.

Behind the scenes the script:

- Uses **YOLOv8** to detect people and bag‑like objects.
- Uses YOLO’s built‑in **tracking** so each worker/sack pair gets a stable ID over time.
- Estimates where the **sack on the head** is by taking the upper part of the person’s bounding box.
- Defines a configurable **line** that represents the loading boundary.
- Increments the count exactly once when a worker+sack crosses that line towards the truck.
- Produces an annotated video that shows:
  - The green counting line.
  - A box focused on the **sack/head region**, not the whole person.
  - A running text overlay with the current bag count.

---

## 2. Files in this repo

- `bag_counter.py` – main script with:
  - YOLO detection and tracking
  - line‑crossing and direction logic
  - visualisation and counting
- `requirements.txt` – Python dependencies (`ultralytics`, `opencv-python`).
- An output directory is created automatically when you run the script to store annotated clips.

---

## 3. How the method works

### 3.1 Detection and tracking

- I use `ultralytics` YOLOv8 with standard COCO weights.
- I call `model.track(..., persist=True)` so that the same worker/sack keeps the **same track ID** across frames.
- I look at:
  - COCO “bag‑like” classes (`backpack`, `handbag`, `suitcase`).
  - The `person` class, which I treat as a proxy for “person carrying a sack”.

### 3.2 Approximating the sack

In the given clips, sacks are usually carried on the workers’ heads or upper body.  
Instead of training a new detector just for sacks, I do a simple approximation:

- Take the YOLO `person` box.
- Crop only the **top part** of that box (roughly the head and upper torso).
- Use that cropped region as the “sack area”.
- Draw bounding boxes only over this region so the visuals focus on the sack, not the legs.

It’s a simple heuristic, but it works surprisingly well for this particular scenario.

### 3.3 Virtual counting line

To model “loading into the truck”, I draw a **virtual line** in the image:

- The line is defined by two points in **normalised coordinates** between 0 and 1:
  - `(line_x1, line_y1)` – start
  - `(line_x2, line_y2)` – end
- `0` means left/top of the frame, `1` means right/bottom.
- From these values the script decides whether the line is more **vertical** or **horizontal**.

This line plays the role of an invisible gate: only sacks that cross it in the right direction are counted.

### 3.4 Directional counting

For each tracked sack centre:

- If the line is **vertical**:
  - I only count if the centre moves from the **right side** of the line to the **left side** (towards the truck).
  - Movements from left to right are ignored, so walking back does not increase the count.
- If the line is **horizontal**:
  - I only count if the centre moves from **above** the line to **below** it.

Once a track has been counted, its ID is stored in a set and is **never counted again**.

### 3.5 Visual output

The resulting visualisation is meant to be easy to understand for non‑CV people:

- Green line – where “loading” is considered to happen.
- Yellow box – approximate sack/head region for each tracked worker.
- Text overlay – `Bags counted (crossed line): N`.

---

## 4. Setup

### 4.1 Environment

- Python **3.9+** recommended.

git clone <this-repo-url>
cd <this-repo-folder>

pip install -r requirements.txt
