"""Small execution adapter for unchanged TRACE sources and CPU/Intel XPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "TRACE"


def device_info(requested="auto"):
    import torch
    info = dict(torch=torch.__version__, requested=requested, device="cpu", name="CPU")
    if requested in ("auto", "xpu"):
        try:
            if torch.xpu.is_available():
                x = torch.ones(8, device="xpu")
                assert float((x @ x).cpu()) == 8.
                torch.xpu.synchronize()
                info.update(device="xpu", name=torch.xpu.get_device_name(0))
            else:
                info["fallback_reason"] = "Intel XPU is unavailable; using CPU"
        except Exception as error:
            info["fallback_reason"] = f"Intel XPU probe failed: {error}; using CPU"
    if requested not in ("auto", "cpu", "xpu"):
        raise ValueError("Device must be auto, cpu or xpu")
    return info


def stage(name, output):
    if name == "training":
        from .training_export import export_training_motion, DEFAULT_DATASET
        config = json.loads((output / "request.json").read_text())
        report = export_training_motion(
            output / "motion_final_qpos.npz",
            dataset_dir=config.get("training_dataset_dir", str(DEFAULT_DATASET)),
            clip_id=output.parent.name,
            metadata={k: config[k] for k in ("text", "seed", "steps", "cfg", "contact_refine", "planner_provider", "planner_model") if k in config})
        (output / "training_export.json").write_text(json.dumps(report, indent=2))
        print("WBC-mjlab training export: " + json.dumps(report), flush=True)
        return
    config = json.loads((output / "request.json").read_text())
    if name == "planner" and config.get("planner_provider", "local") in ("openai", "codex"):
        from .planner import PlannerError, predict_openai_plan
        provider=config["planner_provider"]
        try:
            if provider=="codex":
                from .codex_planner import predict_codex_plan
                predict_codex_plan(config,output)
            else:
                predict_openai_plan(config, output)
        except (PlannerError, ValueError) as error:
            (output / "planner_status.json").write_text(json.dumps(dict(failed=True, provider=provider, error=str(error))))
            print(provider + " planner: " + str(error), flush=True)
            raise SystemExit(1)
        return
    import torch
    sys.path.insert(0, str(TRACE))
    from trace_core import pipeline
    # Upstream language_dtype only distinguishes CUDA and CPU. XPU supports
    # BF16 for the language models; diffusion remains FP32 for compatibility.
    original = pipeline.language_dtype
    pipeline.language_dtype = lambda device: torch.bfloat16 if device.type == "xpu" else original(device)
    config = json.loads((output / "request.json").read_text())
    pipeline.run_stage(name, config, output)


def generate(request):
    import numpy as np
    sys.path.insert(0, str(TRACE))
    from .planner import check_motion_models
    from trace_core.references import validate_checkpoint, sha256
    request = json.loads(request.read_text())
    output = Path(request["output"])
    models = Path(request["models"])
    check_motion_models(models, request.get("planner_provider", "local"))
    checkpoint = models / "generator_boundary.pt"
    checkpoint_info = validate_checkpoint(checkpoint, "standing")
    output.mkdir(exist_ok=False, parents=True)
    device = device_info(request["device"])
    (output / "device.json").write_text(json.dumps(device, indent=2))
    pose = np.asarray(request["boundary_pose35"], dtype=np.float32)
    if pose.shape != (35,) or not np.isfinite(pose).all():
        raise ValueError("Invalid boundary snapshot")
    reference = output / "boundary_reference_pose35.npy"
    np.save(reference, np.stack([pose, pose]), allow_pickle=False)
    config = dict(text=request["text"], models=str(models), device=device["device"],
                  seed=request["seed"], steps=request["steps"], cfg=2.5, precision="fp32",
                  root_hz=3., joint_hz=4., filter_order=4, cpu_threads=request["cpu_threads"],
                  render=False, boundary="standing", boundary_frames=8,
                  contact_refine=request["contact_refine"], generator_checkpoint=str(checkpoint),
                  generation_stats=str(models / "generation_stats.npz"),
                  boundary_reference_path=str(reference), boundary_reference_sha256=sha256(reference))
    for key in ("planner_provider", "planner_model", "planner_timeout_s", "codex_bin", "codex_model"):
        if key in request:
            config[key] = request[key]
    config["training_dataset_dir"] = request.get("training_dataset_dir", str(ROOT / "motions"))
    (output / "checkpoint.json").write_text(json.dumps(checkpoint_info, indent=2))
    config_path = output / "request.json"
    config_path.write_text(json.dumps(config, indent=2))
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    began = time.monotonic()
    for name in ("planner", "embedding", "diffusion", "smooth", "contact", "training"):
        print(f"STAGE {name} device={config['device']}", flush=True)
        (output / "progress.json").write_text(json.dumps(dict(stage=name, device=config["device"])))
        command = [sys.executable, "-m", "motion_teleop.trace_worker", "--stage", name, "--output", str(output)]
        stage_env = dict(env)
        if name != "planner":
            stage_env.pop("OPENAI_API_KEY", None)
        result = subprocess.run(command, cwd=ROOT, env=stage_env)
        if result.returncode and config["device"] == "xpu" and not (name == "planner" and config.get("planner_provider") in ("openai", "codex")):
            print(f"Intel XPU failed at {name}; retrying this and following stages on CPU", flush=True)
            config["device"] = "cpu"
            config_path.write_text(json.dumps(config, indent=2))
            result = subprocess.run(command, cwd=ROOT, env=stage_env)
        result.check_returncode()
    result = dict(success=True, seconds=time.monotonic() - began, fps=20,
                  frames=int(np.load(output / "pose35_final.npy", allow_pickle=False).shape[0]),
                  device=config["device"], requested_device=request["device"],
                  final_qpos="motion_final_qpos.npz", text=request["text"])
    result["planner"] = dict(provider=config.get("planner_provider", "local"), model=config.get("planner_model") if config.get("planner_provider") in ("openai", "codex") else "Qwen3-4B-Instruct-2507 + TRACE LoRA")
    result["training_export"] = json.loads((output / "training_export.json").read_text())
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "xpu"), default="auto")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--stage", choices=("planner", "embedding", "diffusion", "smooth", "contact", "training"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(device_info(args.device)))
    elif args.stage:
        stage(args.stage, args.output)
    elif args.request:
        generate(args.request)
    else:
        parser.error("Use --probe or --request")


if __name__ == "__main__":
    main()
