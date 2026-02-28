import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO



BAG_CLASS_NAMES = {"backpack", "handbag", "suitcase"}
# YOLO does not have an explicit "sack on head" class, so we also rely on the
# detected person box and treat the upper part of the body as an approximate
# location of the sack that the worker is carrying.
PERSON_CLASS_NAME = "person"


def process_video(
  model: YOLO,
  video_path: Path,
  output_dir: Path,
  line: tuple[float, float, float, float] | None = None,
  only_draw_crossed: bool = True,
) -> tuple[int, Path]:
  """
  Process a single video and count how many loaded sacks cross a user‑defined line.

  The function runs YOLO with tracking so that each physical worker/sack pair
  has a stable track ID over time. For every track we follow the centre of the
  "bag" region and increment the global counter once, exactly at the moment
  when that centre crosses the virtual line in the desired direction.

  Args:
    model: Loaded YOLO model.
    video_path: Path to the input video.
    output_dir: Directory where the annotated video will be written.
    line: Optional (x1, y1, x2, y2) expressed in normalised [0, 1] image
          coordinates. If it is not provided the code falls back to a simple
          horizontal line through the middle of the frame.
    only_draw_crossed: If True, the output video only shows boxes for sacks
                       that have already crossed the line. If False, all
                       detected sacks are drawn, with crossed ones highlighted.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"{video_path.stem}_bag_counted.mp4"

  total_count = 0

  writer = None
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  fps = 25.0
  frame_size = None
  pixel_line: tuple[int, int, int, int] | None = None

  # For each track we remember the last "bag centre" point and which track
  # IDs have already been counted as having crossed the line.
  track_centres: dict[int, tuple[float, float]] = {}
  crossed_ids: set[int] = set()

  def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

  def compute_pixel_line(width: int, height: int) -> tuple[int, int, int, int]:
    if line is None:
      # When the user does not specify a line, use a default horizontal line
      # across the middle of the frame so that the script is still runnable.
      x1_n, y1_n, x2_n, y2_n = 0.1, 0.5, 0.9, 0.5
    else:
      x1_n, y1_n, x2_n, y2_n = line

    x1 = int(clamp01(x1_n) * width)
    y1 = int(clamp01(y1_n) * height)
    x2 = int(clamp01(x2_n) * width)
    y2 = int(clamp01(y2_n) * height)
    return x1, y1, x2, y2

  results = model.track(
    source=str(video_path),
    stream=True,
    conf=0.3,
    iou=0.5,
    persist=True,
    verbose=False,
  )

  frame_idx = 0

  for result in results:
    frame_idx += 1
    h, w = result.orig_shape

    if frame_size is None:
      frame_size = (w, h)
      writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
      pixel_line = compute_pixel_line(w, h)

    assert frame_size is not None
    if pixel_line is None:
      pixel_line = compute_pixel_line(frame_size[0], frame_size[1])

    # Always start drawing from the raw frame that comes from YOLO and then
    # overlay our own line and sack bounding boxes on top of it.
    frame = result.orig_img.copy()

    # Draw the counting line that represents the loading boundary.
    lx1, ly1, lx2, ly2 = pixel_line
    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

    # Decide whether the line is closer to vertical or horizontal so that we
    # know which direction (right→left or top→bottom) should be treated as
    # the "loading" motion for counting.
    vertical_line = abs(lx2 - lx1) < abs(ly2 - ly1)
    x_line = (lx1 + lx2) / 2.0
    y_line = (ly1 + ly2) / 2.0

    boxes = result.boxes
    if boxes is not None and boxes.id is not None and boxes.cls is not None:
      ids = boxes.id.int().tolist()
      cls_indices = boxes.cls.int().tolist()
      xyxys = boxes.xyxy

      for box, track_id_t, cls_idx_t in zip(xyxys, ids, cls_indices):
        track_id = int(track_id_t)
        cls_idx = int(cls_idx_t)
        name = model.names[int(cls_idx)]

        if name not in BAG_CLASS_NAMES and name != PERSON_CLASS_NAME:
          continue

        x1_f, y1_f, x2_f, y2_f = box.tolist()

        # In these clips the sack is usually carried on the worker's head.
        # To isolate the sack region we approximate it by taking only the
        # upper part of the person bounding box instead of the whole body.
        if name == PERSON_CLASS_NAME:
          height = y2_f - y1_f
          bag_y1 = y1_f
          bag_y2 = y1_f + 0.45 * height  # top 45% of the body
          bag_x1, bag_x2 = x1_f, x2_f
        else:
          bag_x1, bag_y1, bag_x2, bag_y2 = x1_f, y1_f, x2_f, y2_f

        cx = (bag_x1 + bag_x2) / 2.0
        cy = (bag_y1 + bag_y2) / 2.0

        prev_centre = track_centres.get(track_id)
        track_centres[track_id] = (cx, cy)

        # Directional crossing logic:
        # - For a vertical line, we only count when the bag centre moves from
        #   the right side of the line to the left side (into the lorry area).
        # - For a horizontal line, we only count when the bag moves from above
        #   the line to below it.
        if prev_centre is not None and track_id not in crossed_ids:
          px, py = prev_centre
          crossed = False
          if vertical_line:
            # Worker moving from the right of the line to the loading side.
            if px > x_line and cx <= x_line:
              crossed = True
          else:
            # Worker moving from above the line down towards the loading area.
            if py < y_line and cy >= y_line:
              crossed = True

          if crossed:
            crossed_ids.add(track_id)
            total_count += 1

        should_draw = track_id in crossed_ids or not only_draw_crossed
        if should_draw:
          color = (0, 255, 255) if track_id in crossed_ids else (128, 128, 128)
          cv2.rectangle(
            frame,
            (int(bag_x1), int(bag_y1)),
            (int(bag_x2), int(bag_y2)),
            color,
            2,
          )
          cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
          label = f"id {track_id}"
          cv2.putText(
            frame,
            label,
            (int(bag_x1), max(0, int(bag_y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
          )

    cv2.putText(
      frame,
      f"Bags counted (crossed line): {total_count}",
      (16, 32),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 255),
      2,
      cv2.LINE_AA,
    )

    if writer is not None:
      writer.write(frame)

    # Lightweight progress print so that long videos still give feedback.
    if frame_idx % 50 == 0:
      print(f"  processed {frame_idx} frames...", end="\r", flush=True)

  if writer is not None:
    writer.release()

  return total_count, out_path


def collect_videos(source: Path) -> list[Path]:
  exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
  if source.is_file():
    return [source]
  if source.is_dir():
    videos: list[Path] = []
    for ext in exts:
      videos.extend(source.glob(f"*{ext}"))
    return sorted(videos)
  return []


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Bag counting prototype using YOLO detection + tracking and a virtual "
      "line that represents the loading boundary."
    ),
  )
  parser.add_argument(
    "--source",
    required=True,
    help="Path to a video file or a folder of videos.",
  )
  parser.add_argument(
    "--weights",
    default="yolov8s.pt",
    help="YOLO weights file (default: yolov8s.pt pre-trained on COCO).",
  )
  parser.add_argument(
    "--output-dir",
    default="outputs",
    help="Directory to save annotated videos.",
  )
  parser.add_argument(
    "--line-x1",
    type=float,
    default=0.1,
    help="Normalised X (0-1) for start of counting line (default: 0.1).",
  )
  parser.add_argument(
    "--line-y1",
    type=float,
    default=0.5,
    help="Normalised Y (0-1) for start of counting line (default: 0.5).",
  )
  parser.add_argument(
    "--line-x2",
    type=float,
    default=0.9,
    help="Normalised X (0-1) for end of counting line (default: 0.9).",
  )
  parser.add_argument(
    "--line-y2",
    type=float,
    default=0.5,
    help="Normalised Y (0-1) for end of counting line (default: 0.5).",
  )
  return parser.parse_args(argv)


def main(argv: list[str]) -> None:
  args = parse_args(argv)

  source = Path(args.source)
  output_dir = Path(args.output_dir)

  # Normalised (0-1) line coordinates.
  line = (args.line_x1, args.line_y1, args.line_x2, args.line_y2)

  videos = collect_videos(source)
  if not videos:
    print(f"No video files found in: {source}")
    raise SystemExit(1)

  print("Loading YOLO model...")
  model = YOLO(args.weights)
  print(f"Model loaded. Classes: {model.names}")
  print(f"Counting bags for {len(videos)} video(s)...")

  summary: list[tuple[str, int]] = []

  for video_path in videos:
    print(f"\nProcessing: {video_path}")
    count, out_path = process_video(model, video_path, output_dir, line=line)
    summary.append((video_path.name, count))
    print(f"  -> Bags counted: {count}")
    print(f"  -> Annotated video: {out_path}")

  print("\n=== Summary ===")
  for name, count in summary:
    print(f"{name}: {count} bags")


if __name__ == "__main__":
  main(sys.argv[1:])

