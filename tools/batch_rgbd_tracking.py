#!/usr/bin/env python3
"""Run WAPR tracking for selected object/video pairs.

Expected layout:
  dataset/
    manifest.json
    objects/<object-name>/mesh.ply
    videos/<video-name>/
      rgb/<frame-stem>.png
      depth/<frame-stem>.npy   # float32 depth in meters
      cam_K.txt                # 3x3 intrinsics
      clicks.json              # {"object-name": [u, v], ...}

The click selects the intended instance in the first RGB frame. The first pose
is estimated by WAPRDet2D + WAPR; later poses are tracked with DINOv2 and the
published six-hypothesis WAPR/SAPR/WBPS tracking path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

from wapr import recipe
from wapr.bootstrap import ensure_optional
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.download_assets import check_and_fetch_pack
from wapr.estimator import prepare_mesh
from wapr.resources import outputs_dir
from wapr.view import save_pose_view


def load_follow_module():
    path = EXAMPLES / "16_follow_saved.py"
    spec = importlib.util.spec_from_file_location("wapr_follow_saved", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_names(value: str, available: list[str], option: str) -> list[str]:
    if value.strip().lower() == "all":
        return available
    names = [part.strip() for part in value.split(",") if part.strip()]
    resolved = []
    unknown = []
    for name in names:
        if name in available:
            resolved.append(name)
        elif name.isdigit() and 1 <= int(name) <= len(available):
            resolved.append(available[int(name) - 1])
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(f"{option}: unknown {', '.join(unknown)}; available names (or 1-based indices): {', '.join(available)}")
    if not resolved:
        raise SystemExit(f"{option}: select at least one name or use all")
    return resolved


def read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest.get("objects"), dict) or not isinstance(manifest.get("videos"), dict):
        raise ValueError("manifest.json must contain objects and videos maps")
    return manifest


def video_jobs(video_name: str, video_cfg: dict, dataset_root: Path, stride: int) -> list[dict]:
    folder = (dataset_root / video_cfg["path"]).resolve()
    rgb_dir, depth_dir = folder / "rgb", folder / "depth"
    K_path = folder / video_cfg.get("camera_file", "cam_K.txt")
    if not rgb_dir.is_dir() or not depth_dir.is_dir() or not K_path.is_file():
        raise FileNotFoundError(f"{video_name}: expected rgb/, depth/, and {K_path}")
    K = np.loadtxt(K_path, dtype=np.float32).reshape(3, 3)
    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    jobs = []
    for rgb_path in rgb_files:
        depth_path = depth_dir / (rgb_path.stem + ".npy")
        if depth_path.is_file():
            jobs.append({
                "id": rgb_path.stem,
                "name": video_name,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "K": K.tolist(),
            })
    jobs = jobs[::stride]
    if len(jobs) < 2:
        raise ValueError(f"{video_name}: need at least two matched RGB/depth frames after stride={stride}")
    return jobs


def load_click(video_folder: Path, object_name: str, object_cfg: dict) -> tuple[int, int]:
    click_file = video_folder / "clicks.json"
    if click_file.is_file():
        with click_file.open(encoding="utf-8") as stream:
            clicks = json.load(stream)
        value = clicks.get(object_name)
    else:
        value = object_cfg.get("click_uv")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"Missing click for {object_name} in {click_file}; add "
            f'{{"{object_name}": [u, v]}} to clicks.json'
        )
    return int(value[0]), int(value[1])


def select_detection(detections: list[dict], click_uv: tuple[int, int], image_shape: tuple[int, int]):
    from pycocotools import mask as mask_utils
    u, v = click_uv
    height, width = image_shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"click {click_uv} is outside first RGB frame ({width}x{height})")
    candidates = []
    for detection in detections:
        visible = np.asarray(mask_utils.decode(detection["mask"])) > 0
        if visible.shape != (height, width):
            raise ValueError("detector mask shape does not match RGB")
        if visible[v, u] and int(visible.sum()) >= 32:
            candidates.append((float(detection["score_2d"]), visible))
    if not candidates:
        raise RuntimeError(f"No WAPRDet2D mask covers click {click_uv}")
    return max(candidates, key=lambda item: item[0])[1]


def run_pair(args, follow, estimator, runtime, dino, manifest, dataset_root: Path,
             object_name: str, video_name: str, output_root: Path):
    object_cfg = manifest["objects"][object_name]
    video_cfg = manifest["videos"][video_name]
    mesh_path = (dataset_root / object_cfg["mesh"]).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    diameter_m = float(object_cfg["diameter_m"])
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError(f"{object_name}: diameter_m must be positive")

    jobs = video_jobs(video_name, video_cfg, dataset_root, args.stride)
    first_rgb_bgr = cv2.imread(jobs[0]["rgb"], cv2.IMREAD_COLOR)
    if first_rgb_bgr is None:
        raise FileNotFoundError(jobs[0]["rgb"])
    first_rgb = cv2.cvtColor(first_rgb_bgr, cv2.COLOR_BGR2RGB)
    first_depth = np.load(jobs[0]["depth"]).astype(np.float32)
    if first_depth.ndim == 3:
        first_depth = first_depth[..., 0]
    if first_depth.shape != first_rgb.shape[:2]:
        raise ValueError(f"{video_name}: RGB/depth dimensions differ in {jobs[0]['id']}")
    K = np.asarray(jobs[0]["K"], dtype=np.float32).reshape(3, 3)
    click = load_click((dataset_root / video_cfg["path"]).resolve(), object_name, object_cfg)

    raw_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(raw_mesh.vertices, dtype=np.float64).copy()
    mesh = prepare_mesh(raw_mesh)
    template = onboard_meshes({1: mesh}, device=args.device)
    detector = WAPRDet2D(template, device=args.device, backend=args.det_backend)
    detections, det_timing = detector.detect_many_categories_many_instances(first_rgb, profile=True)
    if not detections:
        raise RuntimeError(f"{object_name}/{video_name}: WAPRDet2D returned no detections")
    mask = select_detection(detections, click, first_rgb.shape[:2]).astype(np.uint8) * 255

    prepared = estimator.prepare_meshes([mesh])[0]
    init = estimator.estimate_one_category_one_instance(
        first_rgb, first_depth, K, prepared, diameter_m, mask=mask,
    )
    pose = np.asarray(init["pose_4x4"], dtype=np.float64).reshape(4, 4)

    pair_out = output_root / object_name / video_name
    pair_out.mkdir(parents=True, exist_ok=True)
    mask_path = pair_out / "init_mask.png"
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"could not write {mask_path}")
    jobs[0]["mask"] = str(mask_path)

    center = np.asarray(follow.center_from_mesh(prepared), dtype=np.float32).reshape(3)
    mesh_id = runtime.load_mesh_trimesh(prepared, name=f"{object_name}_{video_name}")
    rows, times_ms = follow.track_stream(
        estimator, runtime, dino, jobs, pose, prepared, center, diameter_m, vertices, mesh_id,
    )

    poses_path = pair_out / "poses.json"
    with poses_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    for job, row in zip(jobs, rows):
        bgr = cv2.imread(job["rgb"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(job["depth"]).astype(np.float32)
        frame_pose = np.asarray(row["pose"], dtype=np.float64).reshape(4, 4)
        preview_row = {
            "track_id": object_name,
            "mesh": prepared,
            "pose_4x4": frame_pose,
            "score_6d": 0.0,
            "score_2d": 1.0,
        }
        save_pose_view(rgb, [preview_row], str(pair_out / f"{job['id']}_pose.jpg"), K=np.asarray(job["K"]))
    report = {
        "object": object_name,
        "video": video_name,
        "mesh": str(mesh_path),
        "diameter_m": diameter_m,
        "click_uv": list(click),
        "frame_count": len(rows),
        "stride": args.stride,
        "device": args.device,
        "detector_backend": args.det_backend,
        "detector_timing": det_timing,
        "tracking_median_ms": float(np.median(times_ms)) if len(times_ms) else None,
        "tracking_p90_ms": float(np.percentile(times_ms, 90)) if len(times_ms) else None,
        "outputs": {"poses": str(poses_path), "init_mask": str(mask_path)},
    }
    with (pair_out / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"COMPLETE {object_name}/{video_name}: {len(rows)} frames -> {pair_out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="dataset root containing manifest.json")
    parser.add_argument("--objects", default="all", help="comma-separated object names, or all")
    parser.add_argument("--videos", default="all", help="comma-separated video names, or all")
    parser.add_argument("--all", action="store_true", help="run every object against every video (default)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-backend", choices=("trt", "torch"), default="trt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    dataset_root = args.root.expanduser().resolve()
    manifest = read_manifest(dataset_root / "manifest.json")
    object_names = parse_names("all" if args.all else args.objects, sorted(manifest["objects"]), "--objects")
    video_names = parse_names("all" if args.all else args.videos, sorted(manifest["videos"]), "--videos")
    output_root = args.output.expanduser().resolve() if args.output else Path(outputs_dir("18_batch_rgbd_tracking"))

    ensure_optional("det2d")
    ensure_optional("dinov2")
    check_and_fetch_pack("wapr_sapr_wbps")
    follow = load_follow_module()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    estimator = follow.WAPREstimator(device=args.device)
    device_index = torch.device(args.device).index or 0
    runtime = __import__("wapr.raster_fallback", fromlist=["NvRuntime"]).NvRuntime(device_index)
    dino = follow.tracker.load_tracker_dino(args.device)

    failures = []
    total = len(object_names) * len(video_names)
    index = 0
    for object_name in object_names:
        for video_name in video_names:
            index += 1
            print(f"[{index}/{total}] {object_name} x {video_name}", flush=True)
            try:
                run_pair(args, follow, estimator, runtime, dino, manifest, dataset_root,
                         object_name, video_name, output_root)
            except Exception as exc:
                failures.append({"object": object_name, "video": video_name, "error": repr(exc)})
                print(f"FAILED {object_name}/{video_name}: {exc}", flush=True)
    summary = {"completed": total - len(failures), "failed": failures, "output_root": str(output_root)}
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "batch_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if failures:
        raise SystemExit(f"{len(failures)} object/video pair(s) failed; see {output_root / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
