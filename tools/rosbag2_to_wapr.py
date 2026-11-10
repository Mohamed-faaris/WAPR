#!/usr/bin/env python3
"""Extract a ROS 2 SQLite bag into WAPR's RGB-D folder format.

Run this with the ROS 2 environment sourced. The output contains RGB PNGs,
float32 depth .npy files in meters, cam_K.txt, and conversion metadata.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np


def stamp_ns(message, fallback_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_array(message: object) -> np.ndarray:
    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    endian = ">" if bool(message.is_bigendian) else "<"
    formats = {
        "mono8": ("u1", 1), "8uc1": ("u1", 1),
        "16uc1": ("u2", 1), "mono16": ("u2", 1),
        "32fc1": ("f4", 1),
        "rgb8": ("u1", 3), "bgr8": ("u1", 3),
        "rgba8": ("u1", 4), "bgra8": ("u1", 4),
    }
    if encoding not in formats:
        raise ValueError(f"Unsupported Image encoding: {message.encoding}")
    kind, channels = formats[encoding]
    dtype = np.dtype(kind) if kind == "u1" else np.dtype(endian + kind)
    row = np.frombuffer(message.data, dtype=dtype)
    expected = step // dtype.itemsize
    if row.size < height * expected:
        raise ValueError("Image data is shorter than height * step")
    row = row[:height * expected].reshape(height, expected)
    values_per_row = width * channels
    if row.shape[1] < values_per_row:
        raise ValueError("Image step is smaller than its encoded width")
    row = row[:, :values_per_row]
    if channels > 1:
        row = row.reshape(height, width, channels)
    else:
        row = row.reshape(height, width)
    if dtype.itemsize > 1 and row.dtype.byteorder not in ("=", "|"):
        row = row.astype(row.dtype.newbyteorder("="), copy=False)
    return row


def write_rgb(message: object, path: Path) -> None:
    image = image_array(message)
    encoding = str(message.encoding).lower()
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif encoding in {"mono8", "8uc1"}:
        pass
    elif encoding not in {"bgr8", "bgra8"}:
        raise ValueError(f"RGB topic must be color/mono, got {message.encoding}")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def write_depth(message: object, path: Path, depth_scale: float) -> None:
    image = image_array(message)
    encoding = str(message.encoding).lower()
    if encoding in {"16uc1", "mono16"}:
        depth_m = image.astype(np.float32) * float(depth_scale)
    elif encoding == "32fc1":
        depth_m = image.astype(np.float32)
    else:
        raise ValueError(f"Depth topic must use 16UC1 or 32FC1, got {message.encoding}")
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    np.save(path, depth_m.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="ROS 2 bag directory containing metadata.yaml and .db3")
    parser.add_argument("--output", type=Path, required=True, help="New output video folder")
    parser.add_argument("--rgb-topic", required=True)
    parser.add_argument("--depth-topic", required=True)
    parser.add_argument("--camera-info-topic", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="meters per depth unit for 16UC1 (default: 0.001)")
    parser.add_argument("--max-sync-ms", type=float, default=30.0)
    args = parser.parse_args()
    if args.depth_scale <= 0 or args.max_sync_ms < 0:
        parser.error("--depth-scale must be positive and --max-sync-ms non-negative")

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit("Source your ROS 2 environment and install rosbag2_py, rclpy, sensor_msgs: " + str(exc))

    bag = args.bag.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not bag.exists() or not (bag.is_dir() or bag.suffix.lower() == ".db3"):
        raise FileNotFoundError(f"Expected a ROS 2 bag directory or a .db3 file: {bag}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new folder: {output}")

    topic_types = {
        args.rgb_topic: "sensor_msgs/msg/Image",
        args.depth_topic: "sensor_msgs/msg/Image",
        args.camera_info_topic: "sensor_msgs/msg/CameraInfo",
    }
    classes = {topic: get_message(type_name) for topic, type_name in topic_types.items()}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = [topic for topic, type_name in topic_types.items() if available.get(topic) != type_name]
    if missing:
        raise ValueError("Missing topics or unexpected types: " + ", ".join(
            f"{topic} (found {available.get(topic, 'not present')})" for topic in missing
        ))

    output.mkdir(parents=True)
    raw_rgb, raw_depth = output / "_rgb_raw", output / "_depth_raw"
    raw_rgb.mkdir()
    raw_depth.mkdir()
    rgb_records, depth_records = [], []
    camera_k = None
    while reader.has_next():
        topic, serialized, bag_time = reader.read_next()
        if topic not in topic_types:
            continue
        message = deserialize_message(serialized, classes[topic])
        timestamp = stamp_ns(message, int(bag_time))
        if topic == args.rgb_topic:
            path = raw_rgb / f"{timestamp}.png"
            write_rgb(message, path)
            rgb_records.append((timestamp, path))
        elif topic == args.depth_topic:
            path = raw_depth / f"{timestamp}.npy"
            write_depth(message, path, args.depth_scale)
            depth_records.append((timestamp, path))
        elif camera_k is None:
            camera_k = np.asarray(message.k, dtype=np.float64).reshape(3, 3)

    if camera_k is None:
        raise RuntimeError("No CameraInfo message was found")
    if not rgb_records or not depth_records:
        raise RuntimeError(f"No frames found (RGB={len(rgb_records)}, depth={len(depth_records)})")

    rgb_records.sort(key=lambda row: row[0])
    depth_records.sort(key=lambda row: row[0])
    depth_times = [row[0] for row in depth_records]
    max_delta = int(args.max_sync_ms * 1_000_000)
    rgb_dir, depth_dir = output / "rgb", output / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    paired = []
    used_depth = set()
    for rgb_index, (rgb_time, rgb_path) in enumerate(rgb_records):
        position = bisect.bisect_left(depth_times, rgb_time)
        candidates = [i for i in (position - 1, position) if 0 <= i < len(depth_records)]
        if not candidates:
            continue
        depth_index = min(candidates, key=lambda i: abs(depth_times[i] - rgb_time))
        delta = abs(depth_times[depth_index] - rgb_time)
        if delta > max_delta or depth_index in used_depth:
            continue
        used_depth.add(depth_index)
        stem = f"{len(paired):06d}"
        rgb_out, depth_out = rgb_dir / f"{stem}.png", depth_dir / f"{stem}.npy"
        rgb_path.replace(rgb_out)
        depth_records[depth_index][1].replace(depth_out)
        paired.append({"frame": stem, "rgb_timestamp_ns": rgb_time,
                       "depth_timestamp_ns": depth_times[depth_index], "sync_delta_ms": delta / 1e6})

    for folder in (raw_rgb, raw_depth):
        for item in folder.iterdir():
            item.unlink()
        folder.rmdir()
    if not paired:
        raise RuntimeError("No RGB/depth pairs within --max-sync-ms; increase the tolerance")
    np.savetxt(output / "cam_K.txt", camera_k, fmt="%.10g")
    with (output / "conversion.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "bag": str(bag), "rgb_topic": args.rgb_topic, "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic, "depth_scale_16uc1_m": args.depth_scale,
            "max_sync_ms": args.max_sync_ms, "paired_frames": len(paired), "frames": paired,
        }, stream, indent=2)
    print(f"Converted {len(paired)} synchronized RGB-D frames to {output}", flush=True)


if __name__ == "__main__":
    main()
