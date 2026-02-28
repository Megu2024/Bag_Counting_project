# Bag Counting – CV Assignment

This repository contains a small computer vision script (`bag_counter.py`) that counts how many **sacks are loaded into a lorry** from video.

The script uses **YOLOv8** to detect people and bag‑like objects, tracks them across frames, and counts a sack when it crosses a **virtual line** near the truck in the loading direction.  
For workers who carry sacks on their heads, the code focuses on the **upper part of the person box** so the visual box looks like it is drawn around the sack rather than the whole body.

## Requirements

- Python 3.9+
- Install dependencies:

pip install -r requirements.txt
