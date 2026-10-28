"""One generation process at a time, isolated from WBC and the web event loop."""
import asyncio
import json
import os
from pathlib import Path
import signal
import time
import uuid

from .motion import MotionClip, boundary_pose
from .planner import PlannerSettings

ROOT = Path(__file__).resolve().parents[1]


class GenerationJobs:
    def __init__(self, bridge, directory, python, models, device="auto", cpu_threads=4, training_dataset_dir=None, planner=None):
        self.planner = planner or PlannerSettings()
        self._job_keys = {}
        self.bridge = bridge
        self.directory = Path(directory).resolve()
        # Keep the venv entry-point path: resolving its interpreter symlink
        # would execute system Python without TRACE's installed dependencies.
        self.python, self.models = Path(python).absolute(), Path(models).resolve()
        self.device, self.cpu_threads = device, cpu_threads
        self._clips = {}
        self.training_dataset_dir = Path(training_dataset_dir or ROOT / "motions").resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.task = self.process = None
        self.active_id = None
        for metadata in sorted(self.directory.glob("*/job.json")):
            try:
                job = json.loads(metadata.read_text())
                if job["state"] in ("queued", "running"):
                    job.update(state="interrupted", error="Server stopped before generation completed")
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                continue

    def save(self, job):
        path = self.directory / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        temporary.replace(path)

    def create(self, data):
        if not isinstance(data, dict):
            raise ValueError("Generation request must be a JSON object")
        if self.task and not self.task.done():
            raise ValueError("A motion is already being generated; wait or cancel it")
        text = data.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("Enter a motion description of 1..2000 characters")
        if not self.python.is_file():
            raise ValueError("TRACE environment is missing. Run bash motion_teleop/setup.sh")
        steps, seed = data.get("steps", 50), data.get("seed", 42)
        if type(steps) is not int or not 1 <= steps <= 100:
            raise ValueError("Diffusion steps must be 1..100")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in 0..2^32-1")
        refine = data.get("contact_refine", "heel_toe")
        if refine not in ("heel_toe", "none"):
            raise ValueError("Contact refinement must be heel_toe or none")
        planner_config, api_key = self.planner.snapshot()
        identifier = uuid.uuid4().hex
        folder = self.directory / identifier
        folder.mkdir()
        job = dict(id=identifier, text=text.strip(), state="queued", created=time.time(),
                   autoplay=data.get("autoplay", True) is True, epoch=self.bridge.epoch,
                   planner_provider=planner_config["planner_provider"], planner_model=planner_config["planner_model"])
        request = dict(text=job["text"], output=str(folder / "output"), models=str(self.models),
                       device=self.device, cpu_threads=self.cpu_threads, steps=steps, seed=seed,
                       contact_refine=refine, boundary_pose35=boundary_pose(self.bridge.last).tolist(),
                       training_dataset_dir=str(self.training_dataset_dir))
        request.update(planner_config)
        request_path = folder / "input.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2))
        self.jobs[identifier] = job
        self._job_keys[identifier] = api_key
        self.active_id = identifier
        self.save(job)
        self.task = asyncio.create_task(self.run(job, request_path))
        return job

    async def terminate(self):
        process = self.process
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=5.)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    async def cancel(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self._job_keys.pop(self.active_id, None)
            job = self.jobs[self.active_id]
            if job["state"] in ("queued", "running"):
                job.update(state="cancelled", finished=time.time())
                self.save(job)

    async def run(self, job, request):
        log_path = request.parent / "generation.log"
        job.update(state="running", started=time.time())
        self.save(job)
        try:
            environment=dict(os.environ, OMP_NUM_THREADS=str(self.cpu_threads), MKL_NUM_THREADS=str(self.cpu_threads))
            environment.pop("OPENAI_API_KEY", None)
            api_key=self._job_keys.pop(job["id"], "")
            if api_key:
                environment["OPENAI_API_KEY"]=api_key
            with log_path.open("w") as log:
                self.process = await asyncio.create_subprocess_exec(
                    str(self.python), "-u", "-m", "motion_teleop.trace_worker", "--request", str(request),
                    cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                    env=environment)
                code = await self.process.wait()
            if code:
                failure=request.parent/"output/planner_status.json"
                detail=json.loads(failure.read_text()) if failure.is_file() else {}
                if detail.get("provider") in ("openai", "codex") and detail.get("failed"):
                    raise RuntimeError(detail.get("error", "Remote planning failed"))
                raise RuntimeError("TRACE generation failed; see the job log")
            clip = await asyncio.to_thread(self.clip, job["id"])
            result = json.loads((request.parent / "output/result.json").read_text())
            job.update(state="ready", duration_s=clip.duration, frames=len(clip.qpos), result=result)
            if job["autoplay"] and self.bridge.epoch == job["epoch"]:
                self.bridge.play(clip, job["id"])
                job["autoplay_started"] = True
        except asyncio.CancelledError:
            await self.terminate()
            job.update(state="cancelled")
            raise
        except Exception as error:
            job.update(state="failed", error=str(error))
        finally:
            self._job_keys.pop(job["id"], None)
            job["finished"] = time.time()
            self.save(job)
            self.process = None

    def clip(self, identifier):
        if identifier not in self.jobs:
            raise ValueError("Unknown motion ID")
        if identifier not in self._clips:
            clip = MotionClip.load(self.directory / identifier / "output/motion_final_qpos.npz")
            clip.prepare_wbc()
            self._clips[identifier] = clip
        return self._clips[identifier]

    def status(self):
        values = []
        for item in sorted(self.jobs.values(), key=lambda v: v["created"], reverse=True)[:30]:
            job = dict(item)
            progress = self.directory / job["id"] / "output/progress.json"
            if job["state"] == "running" and progress.is_file():
                try:
                    job["progress"] = json.loads(progress.read_text())
                except ValueError:
                    pass
            values.append(job)
        return values
