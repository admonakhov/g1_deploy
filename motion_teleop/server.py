"""HTTP/WebSocket motion control and video/TRACE -> 50 Hz UDP references."""
import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import os
import time

from aiohttp import web, WSMsgType

from video_teleop.teleop_stream import UDPPublisher
from .bridge import ReferenceBridge
from .jobs import GenerationJobs, ROOT
from .planner import PlannerSettings, DEFAULT_MODEL
from .voice import VoiceRecognizer, DEFAULT_MODEL as DEFAULT_SPEECH_MODEL, MAX_AUDIO_BYTES

BRIDGE = web.AppKey("bridge", ReferenceBridge)
JOBS = web.AppKey("jobs", GenerationJobs)
VOICE = web.AppKey("voice", VoiceRecognizer)


def create_app(args, transport=None):
    bridge = ReferenceBridge(args.hz)
    planner=PlannerSettings(args.planner, args.openai_model, args.openai_timeout, codex_bin=args.codex_bin, codex_model=args.codex_model, codex_timeout=args.codex_timeout)
    jobs = GenerationJobs(bridge, args.jobs, args.trace_python, args.models, args.device, args.cpu_threads, args.training_dataset_dir, planner=planner)
    clients = set()
    sockets = set()
    voice = VoiceRecognizer(args.speech_model)

    def status():
        device_info = None
        probe_path = ROOT / ".cache/trace-device.json"
        if probe_path.is_file():
            try:
                device_info = json.loads(probe_path.read_text())
            except ValueError:
                pass
        return dict(bridge=bridge.status(), jobs=jobs.status(), device=args.device, udp_destination=args.udp,
                    device_info=device_info, planner=planner.status(), voice=voice.status(),
                    models_ready=all((Path(args.models) / name).is_file() for name in
                        (("generator_boundary.pt", "generation_stats.npz", "text_embedding/config.json") + (("planner_base/config.json", "planner_adapter/adapter_config.json") if planner.provider=="local" else ()))))

    @web.middleware
    async def errors(request, handler):
        # No CORS: state-changing browser requests must originate here. CLI
        # clients and the camera sender normally do not send an Origin header.
        origin = request.headers.get("Origin")
        if request.method != "GET" and origin and origin != f"{request.scheme}://{request.host}":
            return web.json_response({"error": "Cross-origin control is disabled"}, status=403)
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            return web.json_response({"error": str(error)}, status=400)

    app = web.Application(middlewares=[errors], client_max_size=MAX_AUDIO_BYTES)
    # State is exposed for offline integration tests, never as a robot API.
    app[BRIDGE], app[JOBS] = bridge, jobs
    app[VOICE] = voice

    async def index(request):
        return web.FileResponse(Path(__file__).with_name("web") / "index.html")

    async def font(request):
        return web.FileResponse(Path(__file__).with_name("web") / "fonts/Manrope.ttf")

    async def studio_script(request):
        return web.FileResponse(Path(__file__).with_name("web") / "studio.js")

    async def stylesheet(request):
        return web.FileResponse(Path(__file__).with_name("web") / "style.css")

    async def preview_script(request):
        return web.FileResponse(Path(__file__).with_name("web") / "preview.js")

    async def voice_script(request):
        return web.FileResponse(Path(__file__).with_name("web") / request.match_info["script"])

    async def transcribe(request):
        if request.content_type not in ("audio/wav", "audio/x-wav"):
            raise ValueError("Отправьте WAV-запись микрофона.")
        return web.json_response(await voice.transcribe(await request.read()))

    async def playback(request):
        data = await request.json()
        if not isinstance(data, dict) or data.get("action") not in ("pause", "resume"):
            raise ValueError("Use playback action pause or resume")
        getattr(bridge, data["action"])()
        return web.json_response(status())

    async def get_status(request):
        return web.json_response(status())

    async def source(request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Source request must be a JSON object")
        if data.get("source") == "trace":
            identifier = data.get("motion_id") or bridge.clip_id
            if identifier is None:
                identifier = next((j["id"] for j in jobs.status() if j["state"] == "ready"), None)
            if identifier not in jobs.jobs or jobs.jobs[identifier]["state"] != "ready":
                raise ValueError("Select a successfully generated motion")
            clip = await asyncio.to_thread(jobs.clip, identifier)
            bridge.play(clip, identifier)
        else:
            bridge.select(data["source"])
        return web.json_response(status())

    async def planner_settings(request):
        planner.update(await request.json())
        return web.json_response(status())

    async def codex_status(request):
        await asyncio.to_thread(planner.refresh_codex)
        return web.json_response(status())

    async def pose(request):
        bridge.receive(await request.json())
        return web.json_response({"accepted": True})

    async def generate(request):
        return web.json_response(jobs.create(await request.json()), status=202)

    async def cancel(request):
        await jobs.cancel()
        return web.json_response(status())

    async def log(request):
        identifier = request.match_info["identifier"]
        if identifier not in jobs.jobs:
            raise web.HTTPNotFound()
        path = jobs.directory / identifier / "generation.log"
        text = path.read_text(errors="replace")[-32000:] if path.is_file() else "Waiting for generation"
        return web.Response(text=text, content_type="text/plain")

    async def preview(request):
        identifier=request.match_info["identifier"]
        if identifier not in jobs.jobs or jobs.jobs[identifier]["state"] != "ready":
            raise web.HTTPNotFound(text="Motion is not ready")
        clip=await asyncio.to_thread(jobs.clip, identifier)
        return web.json_response(clip._preview)

    async def download_training(request):
        identifier = request.match_info["identifier"]
        if identifier not in jobs.jobs:
            raise web.HTTPNotFound()
        path = jobs.directory / identifier / "output/motion_wbc_mjlab.npz"
        if not path.is_file():
            raise web.HTTPNotFound(text="Training export is not ready")
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="trace_{identifier}_wbc_mjlab.npz"'})

    async def websocket(request):
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            raise web.HTTPForbidden(text="Cross-origin control is disabled")
        ws = web.WebSocketResponse(heartbeat=20., max_msg_size=256 * 1024)
        await ws.prepare(request)
        sockets.add(ws)
        # A maxsize=1 queue keeps slow web viewers from delaying robot output.
        queue = asyncio.Queue(maxsize=1)
        clients.add(queue)

        async def send():
            try:
                await ws.send_json({"type": "status", **status()})
                while not ws.closed:
                    await asyncio.wait_for(ws.send_json(await queue.get()), timeout=1.)
            finally:
                await ws.close()

        sender = asyncio.create_task(send())
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(message.data)
                        if not isinstance(data,dict) or data.get("type") != "pose":
                            raise ValueError("Send {type: pose, packet: ...}")
                        bridge.receive(data["packet"])
                    except (ValueError, TypeError, KeyError) as error:
                        await ws.send_json({"type": "error", "error": str(error)})
        finally:
            sockets.discard(ws)
            clients.discard(queue)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await ws.close()
        return ws

    async def output_loop(output):
        deadline = report = time.monotonic()
        count = 0
        while True:
            now = time.monotonic()
            packet = bridge.packet(now)
            try:
                output.publish(packet)
                bridge.sent += 1
                bridge.output_error = None
            except OSError as error:
                bridge.output_error = str(error)
            count += 1
            message = dict(type="pose", packet=packet)
            if now - report >= .5:
                bridge.output_hz = count / (now - report)
                message.update(status=status())
                count, report = 0, now
            for queue in tuple(clients):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(message)
            deadline += 1. / bridge.hz
            delay = deadline - time.monotonic()
            if delay < 0:
                deadline = time.monotonic()
            await asyncio.sleep(max(0., delay))

    async def lifecycle(app):
        output = transport or UDPPublisher(args.udp)
        task = asyncio.create_task(output_loop(output))
        yield
        await jobs.cancel()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        output.close()

    app.cleanup_ctx.append(lifecycle)
    async def shutdown(app):
        await asyncio.gather(*(ws.close(code=1001, message=b"Server restarting") for ws in tuple(sockets)), return_exceptions=True)
    app.on_shutdown.append(shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/preview.js", preview_script)
    app.router.add_get("/studio.js", studio_script)
    app.router.add_get("/{script:voice\\.js|voice-worklet\\.js}", voice_script)
    app.router.add_get("/style.css", stylesheet)
    app.router.add_get("/fonts/Manrope.ttf", font)
    app.router.add_get("/api/status", get_status)
    app.router.add_post("/api/source", source)
    app.router.add_post("/api/planner", planner_settings)
    app.router.add_post("/api/codex/status", codex_status)
    app.router.add_post("/api/pose", pose)
    app.router.add_post("/api/generate", generate)
    app.router.add_post("/api/cancel", cancel)
    app.router.add_post("/api/voice/transcribe", transcribe)
    app.router.add_post("/api/playback", playback)
    app.router.add_get("/api/jobs/{identifier}/log", log)
    app.router.add_get("/api/jobs/{identifier}/training.npz", download_training)
    app.router.add_get("/api/jobs/{identifier}/preview", preview)
    app.router.add_get("/ws/poses", websocket)
    return app


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--udp", default="127.0.0.1:9870")
    p.add_argument("--hz", type=float, default=50.)
    p.add_argument("--device", choices=("auto", "cpu", "xpu"), default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--planner", choices=("local", "openai", "codex"), default=os.environ.get("TRACE_PLANNER_PROVIDER", "local"))
    p.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--openai-timeout", type=float, default=90.)
    p.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN"))
    p.add_argument("--codex-model", default=os.environ.get("TRACE_CODEX_MODEL", ""))
    p.add_argument("--codex-timeout", type=float, default=180.)
    p.add_argument("--speech-model", type=Path, default=DEFAULT_SPEECH_MODEL)
    p.add_argument("--trace-python", type=Path, default=ROOT / ".cache/trace-venv/bin/python")
    p.add_argument("--models", type=Path, default=ROOT / "TRACE/models")
    p.add_argument("--jobs", type=Path, default=ROOT / ".cache/trace-jobs")
    p.add_argument("--training-dataset-dir", type=Path, default=ROOT / "motions")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 1 <= args.cpu_threads <= 8 or not 1 <= args.port <= 65535:
        p.error("Use 1..8 CPU threads and a valid HTTP port")
    print(f"G1 motion bridge: {args.hz:g} Hz -> UDP {args.udp}", flush=True)
    print("Connect camera with --web http://HOST:PORT; select Teleop or TeleopArms in the deploy", flush=True)
    web.run_app(create_app(args), host=args.host, port=args.port, shutdown_timeout=5.)


if __name__ == "__main__":
    main()
