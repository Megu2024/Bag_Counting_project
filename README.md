# 🧺 Sack / Bag Counter using YOLOv8 + Tracking

A computer vision prototype that counts loaded sack bags crossing a virtual boundary line in a video.

This project uses YOLOv8 object detection with tracking to count how many workers carrying sacks cross a defined loading line.

---

## 🚀 Problem Statement

In industrial loading scenarios (e.g., rice sack loading into lorries), manual counting is inefficient and error-prone.

This project provides an automated video analytics solution that:

- Detects workers and bag-like objects
- Tracks each worker using persistent tracking IDs
- Draws a virtual counting line
- Counts sacks when they cross the loading boundary
- Prevents double counting
- Saves an annotated output video

---

## 🧠 Approach

### 🔍 Detection
Uses YOLOv8 (pre-trained on COCO dataset).

Since COCO does not contain a dedicated "sack on head" class, the system:

- Detects `person`
- Detects `backpack`, `handbag`, `suitcase`
- Approximates the sack location as the upper portion of the person bounding box

### 🎯 Tracking
YOLO tracking is enabled (`persist=True`) so that:
- Each worker has a unique track ID
- Each sack is counted only once

### 📏 Line Crossing Logic

- A virtual line (vertical or horizontal) is defined using normalized coordinates (0–1).
- For vertical lines:
  - Count when sack center moves from right → left (loading direction).
- For horizontal lines:
  - Count when sack center moves from top → bottom.

Each track ID is counted only once.

---

## 📦 Features

- ✅ Works on single video or folder of videos
- ✅ Direction-aware counting
- ✅ Prevents double counting
- ✅ Customizable counting line
- ✅ Saves annotated output
- ✅ Clean CLI interface
- ✅ Modular structure

---

## 🛠 Installation

```bash
pip install ultralytics opencv-python
