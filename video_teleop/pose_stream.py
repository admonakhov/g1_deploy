"""Read a video source and show people, a selected person and their 3D pose."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from urllib.request import urlopen

import cv2
import numpy as np

from video_teleop.pose_visibility import serializable_pose, without_face

ROOT = Path(__file__).resolve().parents[1]
MODEL_URLS = {
    "vitpose-s-coco.pth": (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
        "torch/coco/vitpose-s-coco.pth"
    ),
    "yolov8n.pt": (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/yolov8/yolov8n.pt"
    ),
    "pretrained_h36m_detectron_coco.bin": (
        "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
}
WINDOW = "Easy-ViTPose | Q: quit | TAB: next person"
PANEL_WIDTH, PANEL_HEIGHT = 480, 480
OUTPUT_SIZE = (PANEL_WIDTH * 2, PANEL_HEIGHT)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path or RTSP/HTTP URL")
    parser.add_argument("--model", type=Path, default=ROOT / "models/vitpose-s-coco.pth")
    parser.add_argument("--yolo", type=Path, default=ROOT / "models/yolov8n.pt")
    parser.add_argument("--model-name", choices=("s", "b", "l", "h"), default="s")
    parser.add_argument("--dataset", choices=("coco", "coco_25", "wholebody", "mpii", "aic"), default="coco")
    parser.add_argument("--pose-view", choices=("3d", "2d"), default="3d", help="Pose visualization; default: 3d")
    parser.add_argument("--pose-backend", choices=("mediapipe", "videopose3d"), default="mediapipe", help="3D estimator; default: MediaPipe Pose Heavy")
    parser.add_argument("--pose-model", type=Path, default=ROOT / "models/pose_landmarker_heavy.task", help="MediaPipe Pose Heavy task file")
    parser.add_argument("--model-3d", type=Path, default=ROOT / "models/pretrained_h36m_detectron_coco.bin", help="Trusted VideoPose3D checkpoint")
    parser.add_argument("--no-palms", action="store_true", help="Disable palm detection; wrist pitch will be held")
    parser.add_argument("--hand-model", type=Path, default=ROOT / "models/hand_landmarker.task")
    parser.add_argument("--hand-threshold", type=float, default=.5, help="Hand detection/presence confidence")
    parser.add_argument("--retarget", action="store_true", help="Stream GMR inverse kinematics to Unitree G1")
    parser.add_argument("--robot", choices=("unitree_g1",), default="unitree_g1")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--udp", help="Publish robot JSON packets to HOST:PORT")
    transport.add_argument("--web", help="Publish references to the motion bridge, e.g. http://127.0.0.1:8765")
    parser.add_argument("--udp-hz", type=float, default=50., help="Continuous UDP/HTTP reference frequency (default: 50 Hz)")
    parser.add_argument("--pose-timeout", type=float, default=.75, help="Use initial pose if no valid completed result arrives for this many seconds")
    parser.add_argument("--loss-grace", type=float, default=.35, help="Hold last valid reference across brief whole-pose losses (seconds)")
    parser.add_argument("--max-joint-speed", type=float, default=0.0, help="Optional per-joint reference speed cap in rad/s (0: disabled)")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 or mps; default: auto")
    parser.add_argument("--download-models", action="store_true", help="Download default models and exit")
    parser.add_argument("--threshold", type=float, default=0.5, help="Keypoint confidence threshold")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1), help="PyTorch CPU threads (default: up to 4)")
    parser.add_argument("--serial-inference", action="store_true", help="Process ViTPose, 3D body and palms sequentially, for comparison")
    parser.add_argument("--profile", action="store_true", help="Report FPS and mean stage times every 2 seconds")
    parser.add_argument("--output", type=Path, help="Save the visualization as an MP4 video")
    parser.add_argument("--json", type=Path, help="Write pose coordinates to JSONL")
    parser.add_argument("--no-display", action="store_true", help="Run without a GUI")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 <= args.hand_threshold <= 1:
        parser.error("--hand-threshold must be between 0 and 1")
    if args.yolo_size <= 0 or args.yolo_size % 32:
        parser.error("--yolo-size must be a positive multiple of 32")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.no_display and not (args.output or args.json or args.download_models or args.udp or args.web):
        parser.error("--no-display requires --output, --json, --udp or --web")
    if args.retarget and args.pose_view != "3d":
        parser.error("--retarget requires --pose-view 3d")
    if (args.udp or args.web) and not args.retarget:
        parser.error("--udp / --web requires --retarget")
    if not np.isfinite(args.udp_hz) or not 1 <= args.udp_hz <= 1000:
        parser.error("--udp-hz must be between 1 and 1000")
    if not np.isfinite(args.pose_timeout) or args.pose_timeout <= 0:
        parser.error("--pose-timeout must be finite and positive")
    if not np.isfinite(args.loss_grace) or not 0 <= args.loss_grace < args.pose_timeout:
        parser.error("--loss-grace must be nonnegative and shorter than --pose-timeout")
    if not np.isfinite(args.max_joint_speed) or args.max_joint_speed < 0:
        parser.error("--max-joint-speed must be finite and nonnegative")
    if args.pose_view == "3d" and args.dataset not in ("coco", "coco_25", "wholebody"):
        parser.error("3D requires coco, coco_25 or wholebody; use --pose-view 2d for this dataset")
    return args


def download_models() -> None:
    """Download to a temporary file so interrupted downloads are never loaded."""
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        target = directory / name
        if target.is_file() and target.stat().st_size:
            print(f"Already present: {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {name} ...", flush=True)
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved: {target}")


def fit_panel(image: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    """Letterbox an image without distorting its aspect ratio."""
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    available_height = height - 40
    scale = min(width / image.shape[1], available_height / image.shape[0])
    w = max(1, min(width, round(image.shape[1] * scale)))
    h = max(1, min(available_height, round(image.shape[0] * scale)))
    x, y = (width - w) // 2, 40 + (available_height - h) // 2
    panel[y:y + h, x:x + w] = cv2.resize(image, (w, h))
    cv2.putText(panel, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def body_overlay(frame, keypoints, boxes, joint_info, threshold):
    from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton

    image = frame.copy()
    for person_id, points in keypoints.items():
        image = draw_points_and_skeleton(
            image, without_face(points, joint_info["keypoints"]), joint_info["skeleton"],
            person_index=int(person_id), confidence_threshold=threshold,
        )
        if person_id in boxes:
            x1, y1, x2, y2 = boxes[person_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 210, 80), 1)
            cv2.putText(image, f"ID {person_id}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 210, 80), 1, cv2.LINE_AA)
    return image


def pose_image(points: np.ndarray, box: tuple, skeleton: list, threshold: float) -> np.ndarray:
    """Map Easy-ViTPose's (y, x, score) to a canvas using the same crop scale."""
    x1, y1, x2, y2 = box
    canvas = np.full((y2 - y1, x2 - x1, 3), 22, dtype=np.uint8)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= threshold)
    valid &= (points[:, 1] >= x1) & (points[:, 1] < x2)
    valid &= (points[:, 0] >= y1) & (points[:, 0] < y2)
    xy = np.zeros((len(points), 2), dtype=int)
    xy[valid] = np.rint(points[valid, :2][:, ::-1] - [x1, y1]).astype(int)
    thickness = max(2, round(max(canvas.shape[:2]) / 200))
    for a, b in skeleton:
        if valid[a] and valid[b]:
            cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), (50, 210, 255), thickness, cv2.LINE_AA)
    for index in np.flatnonzero(valid):
        center = tuple(xy[index])
        cv2.circle(canvas, center, thickness + 1, (100, 255, 100), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def pose_panel_mouse(event, x, y, flags, context):
    viewer, left = context
    inside = left <= x < left + PANEL_WIDTH and 0 <= y < PANEL_HEIGHT
    if inside or event == cv2.EVENT_LBUTTONUP:
        viewer.mouse(event, x - left, y, flags, None)
    elif viewer.drag is not None:
        viewer.mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)


def render(frame, overlay, keypoints, boxes, selected, skeleton, threshold, fps, view3d=None):
    pose=np.full((440,320,3),22,dtype=np.uint8)
    if view3d is not None:
        pose=view3d
    elif selected in boxes:
        pose=pose_image(keypoints[selected],boxes[selected],skeleton,threshold)
    video_panel=fit_panel(overlay,PANEL_WIDTH,PANEL_HEIGHT,f"Full frame | {fps:.1f} FPS | People: {len(boxes)}")
    pose_panel=fit_panel(pose,PANEL_WIDTH,PANEL_HEIGHT,"3D pose | drag to rotate" if view3d is not None else "2D pose | joint indices")
    return np.hstack([video_panel,pose_panel])


def _run_video(args, reader, publisher, stream_id):
    if args.download_models:
        download_models()
        return
    model_paths = [args.model, args.yolo]
    if args.pose_view == "3d":
        model_paths.append(args.pose_model if args.pose_backend == "mediapipe" else args.model_3d)
        if not args.no_palms:
            model_paths.append(args.hand_model)
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model missing: {path}. Run: ./uv run python -m video_teleop.pose_stream --download-models")
    # Keep library caches inside the project and allow inference on headless hosts.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.retarget and sys.platform.startswith("linux") and (args.no_display or not os.environ.get("DISPLAY")):
        os.environ.setdefault("MUJOCO_GL", "egl")
    if not args.no_display and sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("No display available. Use --no-display with --output or --json.")

    from easy_ViTPose import VitInference
    from easy_ViTPose.vit_utils.visualization import joints_dict
    import torch

    # Avoid excessive CPU thread overhead on small per-person inference batches.
    torch.set_num_threads(args.cpu_threads)
    model = VitInference(
        str(args.model), str(args.yolo), model_name=args.model_name,
        dataset=args.dataset, det_class="human", device=args.device,
        is_video=True, yolo_size=args.yolo_size,
    )
    joint_info = joints_dict()[args.dataset]
    from video_teleop.person_tracking import CroppedPersonInference, select_person
    frontend = CroppedPersonInference(model)
    lifter, viewer = None, None
    if args.pose_view == "3d":
        from video_teleop.pose3d import H36M_NAMES, Pose3DViewer, PoseLifter
        if args.pose_backend == "mediapipe":
            from video_teleop.mediapipe_pose import MediaPipePose
            lifter = MediaPipePose(args.pose_model, joint_info["keypoints"], args.threshold)
        else:
            lifter = PoseLifter(args.model_3d, model.device, joint_info["keypoints"], args.threshold)
        if args.output or not args.no_display:
            viewer = Pose3DViewer()
    source = int(args.source) if args.source.isdecimal() else args.source
    if args.output and isinstance(source, str) and "://" not in source:
        if Path(source).resolve() == args.output.resolve():
            raise ValueError("Input and output video paths must be different")
    with ExitStack() as stack:
        stack.callback(model.reset)
        if args.pose_view == "3d" and args.pose_backend == "mediapipe":
            stack.callback(lifter.close)
        if not args.no_display:
            stack.callback(cv2.destroyAllWindows)
        stream_fps = reader.fps
        is_live = reader.is_live
        palm_tracker = None
        if args.pose_view == "3d" and not args.no_palms:
            from video_teleop.palm_tracking import PalmTracker, draw_palms
            palm_tracker = PalmTracker(args.hand_model, joint_info["keypoints"], args.threshold, args.hand_threshold)
            stack.callback(palm_tracker.close)
        retargeter = None
        if args.retarget:
            from video_teleop.robot_retarget import StreamingRetargeter
            retargeter = StreamingRetargeter(max_joint_speed=args.max_joint_speed)
            if publisher:
                from video_teleop.teleop_stream import initial_reference
                publisher.set_initial(initial_reference(retargeter))
        writer = None
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), stream_fps, OUTPUT_SIZE)
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {args.output}")
        json_file = None
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            json_file = stack.enter_context(args.json.open("w", encoding="utf-8"))
        # Each detector gets at most one task, joined within the same frame.
        # Register after detector cleanup callbacks so shutdown precedes close().
        inference_workers = (stack.enter_context(ThreadPoolExecutor(max_workers=2, thread_name_prefix="pose-inference"))
                             if lifter and not args.serial_inference else None)
        from video_teleop.pipeline_profile import PipelineProfile
        from video_teleop.tracking_health import tracking_health
        profile = PipelineProfile(args.profile)
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, *OUTPUT_SIZE)
            if viewer:
                pose_left = PANEL_WIDTH
                cv2.setMouseCallback(WINDOW, pose_panel_mouse, (viewer, pose_left))
        selected, frame_index, fps = None, 0, 0.0
        previous_capture_time = None
        started = time.monotonic()
        last_udp_report, last_udp_report_frame, last_udp_sent = started, 0, 0
        valid_poses, failures = 0, Counter()
        print(f"Device: {model.device}. CPU threads: {torch.get_num_threads()}. 3D: {lifter.model_name if lifter else 'off'}. Press Q / Esc to quit, TAB to select another person.", flush=True)
        if args.retarget:
            if publisher:
                print(f"Deployment stream: {'HTTP' if args.web else 'UDP'} -> {args.web or args.udp} | {args.udp_hz:g} Hz | fallback: G1 initial stance", flush=True)
            else:
                print("Deployment stream: OFF (retargeting preview only). "
                      "Add --udp 127.0.0.1:9870 to send poses to the deployment controller.", flush=True)
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame, source_frame, captured_at, captured_unix = reader.read()
            if not ok:
                if publisher:
                    publisher.use_initial(reader.failure_reason)
                    selected,previous_capture_time=None,None
                    if lifter:lifter.reset()
                    if palm_tracker:palm_tracker.wrists_visible=False
                    if not wait_for_source(args,publisher,reader.failure_reason):break
                    continue
                if frame_index == 0:
                    raise RuntimeError("Video source opened but returned no frames")
                print("Video ended or the stream stopped returning frames.")
                break
            tick = time.perf_counter()
            boxes = profile.call("yolo", frontend.detect, frame)
            previous_selected = selected
            selected = select_person(boxes, selected, frame.shape[1], frame.shape[0])
            keypoints, processing_box, roi, roi_points = {}, None, None, None
            pending_pose, pending_palms = None, None
            prefetched_palms = False
            if selected is not None:
                processing_box = (0, 0, frame.shape[1], frame.shape[0])
                roi = frame
                # All three models see this exact ROI. Only image inference is
                # speculative; visibility/identity checks still use current 2D.
                if inference_workers:
                    if args.pose_backend == "mediapipe":
                        pending_pose = inference_workers.submit(profile.call, "pose3d", lifter.detect, roi)
                    # Previous visibility chooses timing only. The result is
                    # still gated by this frame's wrists after ViTPose completes.
                    if palm_tracker and selected == previous_selected and palm_tracker.wrists_visible:
                        pending_palms = inference_workers.submit(profile.call, "palms", palm_tracker.detect,
                            roi, (0, 0, roi.shape[1], roi.shape[0]))
                        prefetched_palms = True
                keypoints[selected] = profile.call("vitpose", frontend.pose, frame, boxes[selected])
                roi_points = keypoints[selected]
                if inference_workers and palm_tracker and pending_palms is None:
                    # First/reacquired wrists wait for current 2D visibility.
                    pending_palms = inference_workers.submit(profile.call, "palms", palm_tracker.update,
                        roi, roi_points, (0, 0, roi.shape[1], roi.shape[0]))
            palms = {}
            xyz = None
            if lifter:
                if args.pose_backend == "mediapipe":
                    if roi is None:
                        lifter.reset()
                    elif pending_pose is not None:
                        xyz = profile.call("associate3d", lifter.update, roi, roi_points, selected,
                            result=pending_pose.result())
                    else:
                        xyz = profile.call("pose3d", lifter.update, roi, roi_points, selected)
                else:
                    xyz = profile.call("pose3d", lifter.update, keypoints.get(selected), selected, frame.shape[1], frame.shape[0])
            if palm_tracker and roi is not None:
                if prefetched_palms:
                    palms = profile.call("associate_palms", palm_tracker.update, roi, roi_points,
                        (0, 0, roi.shape[1], roi.shape[0]), detection=pending_palms.result())
                elif pending_palms is not None:
                    palms = pending_palms.result()
                else:
                    palms = profile.call("palms", palm_tracker.update, roi, roi_points, (0, 0, roi.shape[1], roi.shape[0]))
                for observation in palms.values():
                    observation.image_xy += [processing_box[0], processing_box[1]]
            robot_packet, deployment_packet = None, None
            if retargeter:
                dt = (captured_at - previous_capture_time) if is_live and previous_capture_time is not None else 1 / stream_fps
                robot_packet = profile.call("gmr", retargeter.update, xyz, selected, dt, observed=lifter.observed,
                    palms={side: observation.offset_camera for side, observation in palms.items()})
                robot_packet.update({
                    "stream_id": stream_id,
                    "frame": frame_index, "source_frame": source_frame,
                    "capture_monotonic_s": captured_at,
                    "timestamp_unix_s": time.time(), "capture_unix_s": captured_unix,
                    "source_time_s": None if is_live else source_frame / stream_fps,
                    "latency_s": time.monotonic() - captured_at,
                    "processing_completed_monotonic_s": time.monotonic(),
                })
            health = tracking_health(frontend, lifter, selected, xyz, robot_packet)
            valid_poses += health["reason"] == "tracking"
            if health["reason"] != "tracking":
                failures[health["reason"]] += 1
            if robot_packet:
                robot_packet.update(tracking_reason=health["reason"], tracking_diagnostics=health)
                if publisher:
                    publisher.publish(robot_packet)
                    _, deployment_packet = publisher.snapshot()
            previous_capture_time = captured_at
            if json_file:
                record = {
                    "frame": frame_index,
                    "elapsed_s": time.monotonic() - started,
                    "dataset": args.dataset,
                    "tracking_diagnostics": health,
                    "coordinate_order": ["y", "x", "score"],
                    "joint_names": joint_info["keypoints"],
                    "selected_id": selected,
                    "processing_bbox_xyxy": processing_box,
                    "detected_people": {str(i): {"bbox_xyxy": box} for i, box in boxes.items()},
                    "people": {str(i): {"bbox_xyxy": boxes.get(int(i)),
                        "keypoints": serializable_pose(without_face(k, joint_info["keypoints"]))}
                        for i, k in keypoints.items()},
                }
                if lifter:
                    record["pose3d"] = {
                        "person_id": selected,
                        "model": lifter.model_name,
                        "joint_names": H36M_NAMES,
                        "coordinate_order": ["x", "y", "z"],
                        "axes": ["right", "down", "forward"],
                        "origin": "pelvis",
                        "units": "estimated_metres",
                        "temporal_mode": lifter.temporal_mode,
                        "observed_mask": lifter.observed.tolist(),
                        "keypoints": serializable_pose(xyz),
                    }
                    record["palms"] = {
                        side: None if side not in palms else palms[side].to_json(None if xyz is None else xyz[wrist])
                        for side, wrist in (("left", 13), ("right", 16))
                    }
                if robot_packet:
                    record["robot"] = robot_packet
                if deployment_packet:
                    record["robot_reference"] = deployment_packet
                json_file.write(json.dumps(record, allow_nan=False) + "\n")
                json_file.flush()
            if writer is not None or not args.no_display:
                view3d = profile.call("draw3d", viewer.render, xyz, selected, palms=palms) if viewer else None
                overlay = body_overlay(frame, keypoints, boxes, joint_info, args.threshold)
                if palms:
                    overlay = draw_palms(overlay, palms)
                visible_keypoints = {i: without_face(k, joint_info["keypoints"]) for i, k in keypoints.items()}
                result = render(frame, overlay, visible_keypoints, boxes, selected, joint_info["skeleton"], args.threshold, fps, view3d)
                if writer is not None:
                    writer.write(result)
                if not args.no_display:
                    cv2.imshow(WINDOW, result)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == 9 and boxes:
                        ids = sorted(boxes)
                        selected = ids[(ids.index(selected) + 1) % len(ids)]
            elapsed = time.perf_counter() - tick
            fps = 1 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed
            frame_index += 1
            profile.frame_done()
            if publisher and time.monotonic() - last_udp_report >= 2.:
                report_time = time.monotonic()
                sent, last_packet = publisher.snapshot()
                interval = report_time - last_udp_report
                publish_fps = (sent - last_udp_sent) / interval
                camera_fps = (frame_index - last_udp_report_frame) / interval
                valid_fps = valid_poses / interval
                failure_report = ",".join(f"{reason}:{count}" for reason, count in failures.most_common()) or "none"
                stream_status = last_packet['status'] if last_packet else 'starting'
                if last_packet and last_packet.get('tracking_state') == 'holding':
                    stream_status = 'holding'
                reason = f" ({last_packet['fallback_reason']})" if last_packet and stream_status == 'fallback' else ''
                print(f"Camera {'HTTP' if args.web else 'UDP'}: {publish_fps:.1f} Hz | camera={camera_fps:.1f} FPS | valid={valid_fps:.1f} FPS | pose={stream_status}{reason} | "
                      f"arms=L:{health['arm_joints']['left']}/R:{health['arm_joints']['right']} | failures={failure_report} | "
                      f"capture-to-pose={robot_packet['latency_s']:.3f} s | "
                      f"fallbacks={last_packet.get('fallback_entries', 0) if last_packet else 0} | "
                      f"holds={last_packet.get('holding_entries', 0) if last_packet else 0}", flush=True)
                last_udp_report, last_udp_report_frame, last_udp_sent = report_time, frame_index, sent
                valid_poses, failures = 0, Counter()
        profile.frame_done(force=True)
        print(f"Processed {frame_index} frames in {time.monotonic() - started:.1f} s.")


def wait_for_source(args, publisher, reason):
    """Keep the independent sender alive while waiting for video, including EOF."""
    publisher.snapshot()  # Surface transport errors instead of silently waiting.
    if not args.no_display:
        image=np.full((OUTPUT_SIZE[1],OUTPUT_SIZE[0],3),22,dtype=np.uint8)
        cv2.putText(image,"Video unavailable: "+reason,(20,55),cv2.FONT_HERSHEY_SIMPLEX,.7,(220,220,220),2)
        cv2.putText(image,"Sending G1 initial pose | Q / Esc: quit",(20,100),cv2.FONT_HERSHEY_SIMPLEX,.7,(80,230,80),2)
        cv2.imshow(WINDOW,image)
        key=cv2.waitKey(100)&0xff
        return key not in (27,ord("q")) and cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)>=1
    time.sleep(.1)
    return True


def run(args):
    if args.download_models:
        download_models()
        return
    source=int(args.source) if args.source.isdecimal() else args.source
    if args.output and isinstance(source,str) and "://" not in source and Path(source).resolve()==args.output.resolve():
        raise ValueError("Input and output video paths must be different")
    from video_teleop.stream_capture import RecoveringCapture
    from video_teleop.teleop_stream import ContinuousPosePublisher, UDPPublisher, HTTPPublisher, initial_reference
    stream_id=uuid.uuid4().hex
    with ExitStack() as stack:
        publisher=None
        if args.retarget and (args.udp or getattr(args,"web",None)):
            transport=HTTPPublisher(args.web) if getattr(args,"web",None) else UDPPublisher(args.udp)
            publisher=ContinuousPosePublisher(transport,initial_reference(),stream_id,
                hz=args.udp_hz,pose_timeout=args.pose_timeout,loss_grace=args.loss_grace)
            stack.callback(publisher.close)
            publisher.start()  # Before camera opening or expensive model initialization.
        reader=RecoveringCapture(source,latest=args.retarget)
        stack.callback(reader.close)
        if not args.no_display:
            if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("No display available. Use --no-display.")
            stack.callback(cv2.destroyAllWindows)
        announced=False
        while not reader.ready():
            if publisher is None:
                raise RuntimeError("Cannot open video source or read its first frame")
            publisher.use_initial(reader.failure_reason)
            if not announced:
                print("Video unavailable; sending G1 initial pose continuously. Retrying input; Ctrl+C to quit.",flush=True)
                announced=True
            if not wait_for_source(args,publisher,reader.failure_reason):return
        _run_video(args,reader,publisher,stream_id)


def main() -> int:
    try:
        run(arguments())
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, ImportError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
