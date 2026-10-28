# TRACE motion bridge for G1 deploy

Generate G1 motions from text and feed them into the same WBC Teleop / TeleopArms states as camera retargeting. The web bridge is the single UDP publisher; its UI selects video, TRACE, or the initial stance.

## Install

Run from the repository root:

```bash
bash motion_teleop/setup.sh auto
```

The installer uses `uv`. The web bridge uses the project environment with the `motion` extra. TRACE uses `.cache/trace-venv`, with its own PyTorch, Transformers, and NumPy versions. Camera inference dependencies remain independent. Official TRACE artifacts and the two Qwen 4B bases require approximately 17–18 GB of weights. TRACE artifacts are checked against the upstream release manifest.

On an Intel GPU, setup selects PyTorch XPU. The worker probes the device and falls back to CPU when unavailable. If a generation stage fails on XPU, that stage and subsequent stages retry on CPU. Use `bash motion_teleop/setup.sh cpu` for CPU-only installation. No driver changes are made.

```bash
.cache/trace-venv/bin/python -m motion_teleop.trace_worker --probe
```

The TRACE submodule stays unchanged. The worker adds process orchestration and XPU language-model dtype support around its original stages. Diffusion uses FP32; language models use BF16 on XPU and FP32 on CPU.

## Run

Rebuild the controller after updating its sources:

```bash
cmake -S deploy/g1 -B deploy/g1/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/g1/build -j2 -- --jobserver-style=pipe
deploy/g1/build/g1_ctrl --check-config
```

Start the simulator, controller, and web bridge:

```bash
# Terminal 1: controller, after starting unitree_mujoco
deploy/g1/build/g1_ctrl -n lo --domain 1

# Terminal 2: web bridge
./uv run --locked --extra motion python -m motion_teleop.server
```

Open **http://127.0.0.1:8765**. Initial references are published at 50 Hz to `127.0.0.1:9870` immediately. Select **R2+A** for whole-body WBC Teleop or **R2+X** for arms-only tracking.

Describe a motion, preferably in English, and click Generate. The original local pipeline runs:

```text
Text → language event planner → text embedding → diffusion
     → smoothing → heel/toe contact refinement → G1 qpos at 20 Hz
     → interpolation at 50 Hz → UDP → WBC
```

Generation completes a clip before playback; diffusion is not real-time. While generating, the selected video or initial reference continues. The UI shows progress and retains clips for replay. Automatic playback is optional. Selecting Video or Stance while generating cancels that job's automatic playback, while generation itself continues.

Start/end references use a snapshot of the active goal when generation starts. At clip end, the final pose is held with zero motion velocity. Select Stance to return to the initial WBC reference or Video to resume camera control. WBC receives the reference directly: no deployment EMA, joint speed cap, entry blend or height/velocity clamp.

### Choose a language planner: local Qwen or OpenAI API

Open **Planner** (`Планировщик`) in the web panel. Choose **Qwen + TRACE LoRA** for offline planning or **OpenAI API** for remote planning. The default API model is `gpt-4.1-mini`; the model field accepts another model that supports Responses API Structured Outputs. Enter your OpenAI API key in the password field and click Apply, or click Generate (which also applies the selected settings).

The key is kept only in server memory, never in job JSON, logs, motion files or browser local storage. Reloading the page preserves the server's key; restarting the server requires entering it again. **Remove key from memory** clears the credential for future jobs; an already-running generation uses its own captured settings. Do not put API keys in motion descriptions or model names.

Alternatively, set `OPENAI_API_KEY` in the server environment and start:

```bash
./uv run --locked --extra motion python -m motion_teleop.server \
  --planner openai --openai-model gpt-4.1-mini --device cpu
```

Optional defaults: `TRACE_PLANNER_PROVIDER=local|openai|codex`, `OPENAI_MODEL`, and `--openai-timeout 90` (seconds). The local planner remains the default; there are no API calls until OpenAI is selected and a generation is requested. API billing/access belongs to the supplied OpenAI API project.

Only the **motion description** is sent to `https://api.openai.com/v1/responses`; camera frames, pose snapshots and robot state stay local. Requests use `store: false` and a strict JSON schema. Named events are converted to the unchanged TRACE positional schema and validated with TRACE's actual compiler: duration, category ranges, strictly increasing 4-frame patch endpoints, complete coverage and boundary-compatible length. One retry is allowed for a semantically invalid plan; API connection/auth/quota errors and refusals fail explicitly without a silent switch to Qwen. An API failure does not change the selected reference stream or the active robot policy.

**Embedding, diffusion, contact processing, 50 Hz playback and training NPZ export remain local.** OpenAI planning skips the local Qwen planner and LoRA entirely, including the requirement for those planner files. The original Qwen embedding remains necessary because the motion generator was trained with its features. A valid JSON plan does not guarantee better motion quality; compare generated motions using the preview and simulator.

Each job records `planner_provider`/`planner_model`; the accepted plan is saved as `output/plan.json` and `output/plan.npz`. `output/planner_status.json` records validation, API model, attempts, timing and token usage without credentials. The same public planner provenance is included in the training NPZ export metadata.

The adapter uses the documented [OpenAI Responses API Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and the [GPT-4.1 mini model](https://developers.openai.com/api/docs/models/gpt-4.1-mini). The real API requires a key and model access; local tests use simulated API replies and never spend API credits.

### Codex subscription without an API key

Choose **Codex · ChatGPT subscription** in the planner selector. This backend uses the official **Codex CLI's saved ChatGPT login**, with the subscription's available usage and model limits. It does not read/copy OAuth tokens and does not require `OPENAI_API_KEY`. The API-key backend is still available as a separate choice.

Install the official Codex CLI separately and authenticate once. The adapter looks for `codex` on `PATH`, and can also detect the desktop app's bundled Linux CLI:

```bash
codex login
codex login status
```

Choose **Sign in with ChatGPT**, then select **Check Codex sign-in** in the panel. The application checks the authentication method and refuses an API-key login for this backend. A successful status probe confirms the cached login; actual generation checks subscription/model access.

```bash
./uv run --locked --extra motion python -m motion_teleop.server \
  --planner codex --device cpu
```

Optional settings: `--codex-bin /path/to/codex` (or `CODEX_BIN`), `--codex-model MODEL` (or `TRACE_CODEX_MODEL`) and `--codex-timeout 180`. An empty model field lets Codex CLI select its default model. Changing the planner does not change the selected video/TRACE reference source or start playback.

Each request runs `codex exec` in a fresh temporary directory with a JSON output schema, read-only sandbox, no approval prompts, tools/plugins/apps/hooks disabled, user configuration/project documents excluded, and ephemeral session storage. The application strips API-key/access-token environment overrides and lets the official client manage its saved ChatGPT authentication and refresh. No token values or raw CLI authentication output are put into jobs, logs or NPZ. Cancellation terminates the generation process group, including the CLI; failures do not switch to API-key billing.

Codex returns a plan, not motor commands. TRACE performs the same timeline validation, local embedding, diffusion, contact processing and training NPZ export as before. The web panel's 3D preview is local and independent of deployment playback. This integration is intended for a private local Motion Studio instance.

The CLI executable, authentication files, configuration and conversation/session history are personal local dependencies and must not be committed. The repository ignores Codex home directories, common CLI distribution names and credential files. Keep the CLI's normal user data outside this checkout, including when setting `CODEX_HOME`. Only the Python adapter and its credential-free tests belong in the repository.

Official references: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode), [Codex ChatGPT authentication](https://learn.chatgpt.com/docs/auth).

### Dashboard design

The light dashboard takes visual inspiration from the [Ronas IT banking application concept](https://ronasit.com/design-projects/banking-web-application-design-digital-financial-services/): rounded white panels, a pale mint workspace, teal/blue reference cards and a three-column desktop layout. The library, generation/preview workspace and stream/planner controls remain functional. At narrower widths the library moves below the workspace; mobile uses a single column. Search and ready/in-progress filters use actual job data, and the frequency chart uses measured stream updates.

Manrope is bundled locally in `web/fonts/` under its included SIL Open Font License, so the interface does not need an external font service.

### Russian voice commands

Click **Voice command** (`Голосовая команда`), allow microphone access, and speak Russian. Recording ends after approximately 1.3 seconds of silence following speech, after a second click, or at the 20-second limit. **Cancel recording** or Escape discards the recording. The microphone is closed after each utterance; there is no background listening.

With **Execute voice commands immediately** enabled, a description such as **«помаши правой рукой»** or **«сделай два шага назад»** starts the normal TRACE generation with the selected planner, steps, seed and contact settings. The existing **Send to deployment after generation** checkbox controls automatic playback. Disable immediate execution to use voice as dictation: inspect/edit the recognized text, then click Generate. The last recognized phrase and action/error remain visible.

Supported control phrases (complete utterances; longer descriptions are passed to the motion planner):

- **«отмени генерацию»** / **«отмена»**: cancel generation, leave the current reference source selected.
- **«включи видео»** / **«включи камеру»**: select the video reference source.
- **«стойка»** / **«исходная стойка»**: select the initial WBC reference.
- **«повтори движение»**: play the latest successfully generated trajectory.
- **«стоп»** / **«пауза»**: cancel generation and pause the active TRACE trajectory at its current reference pose, with zero reference linear/angular velocities. WBC remains active. This is a reference playback command, not a motor emergency stop; it does not pause live video.
- **«продолжи»**: resume the paused TRACE trajectory without restarting its timeline. The stream panel also has manual Pause/Resume buttons.

Recognition uses the official [Vosk](https://alphacephei.com/vosk/install) engine and [vosk-model-small-ru-0.22](https://alphacephei.com/vosk/models), on CPU in a separate low-priority, single-threaded worker. It does not change camera or policy inference. Audio stays between the browser and this server and is not saved; only the recognized motion description follows the chosen planner's existing local/API/Codex behavior. The small model prioritizes low resource use; check the displayed transcription, particularly with noise or unusual motion names.

The regular setup script installs voice support. To add it to an existing installation:

```bash
./uv sync --locked --extra motion
./uv run --locked --extra motion python -m motion_teleop.setup_voice
```

The installer downloads the approximately 45 MB model from the official Vosk site, verifies its pinned SHA-256 and places it under `.cache/voice/`. Subsequent recognition is offline. `--speech-model /path/to/vosk-model` selects another compatible Russian model. No microphone opens during installation/server startup. Browser microphone capture requires **localhost or HTTPS** and Web Audio support; use Chrome/Chromium if an embedded browser blocks microphone access.

The local API **POST `/api/voice/transcribe`** accepts `audio/wav` (16-bit PCM, mono, 16 kHz, 0.2–20 seconds) and returns `{text, command, language, engine}`. Transcription alone does not generate or change references; the UI executes recognized commands through the existing APIs. **POST `/api/playback`** with `{ "action": "pause" }` or `{ "action": "resume" }` controls TRACE playback. Cross-origin writes are blocked, as for the other control endpoints.

### Preview a generated trajectory

In **Your motions**, select **3D**. The panel renders the G1 link skeleton from the training kinematics at 50 Hz. Play/pause, seek, reset and 0.25×/0.5×/1×/2× playback are local browser controls. Drag to orbit and scroll to zoom. This shows the reference, not a closed-loop robot simulation. Preview does not change the selected source or send the clip to deployment. Use **Send to deployment** (`В деплой`) for actual streaming.

### Camera or video

Route the camera through the bridge with `--web` in place of direct `--udp`:

```bash
./uv run --locked --extra motion python -m video_teleop.pose_stream \
  --source 0 --retarget --web http://127.0.0.1:8765 --no-display
```

Replace `0` with a file path or RTSP/HTTP video URL. Camera inference keeps its existing backend. The HTTP sender reconnects after a bridge restart. Missing camera joints stay `null` with a false observed mask.

When using the bridge, stop any direct camera UDP sender targeting the same deployment port. Both sources must pass through the one selected bridge output.

### Addresses and rate

```bash
./uv run --locked --extra motion python -m motion_teleop.server \
  --host 127.0.0.1 --port 8765 --udp 127.0.0.1:9870 --hz 50 \
  --device auto --cpu-threads 4
```

The deployment policies run at 50 Hz. Generation uses four CPU threads by default to leave time for camera inference and control. The server listens only on localhost by default; use an appropriate listening address for another computer.

## Web API

- `GET /api/status`: source, measured output frequency, jobs, and progress.
- `POST /api/planner`: `{ "provider": "local|openai|codex", "model": "gpt-4.1-mini", "codex_model": "optional CLI model", "api_key": "optional API credential" }`; keep the key out of recorded shell commands. An empty/missing `api_key` keeps the existing key; `{ "clear_api_key": true }` clears it. Responses expose only `api_key_configured`.
- `POST /api/codex/status`: refresh the public Codex CLI availability/ChatGPT sign-in status; never returns credentials.
- `POST /api/generate`: `{ "text": "Wave the right hand, then return to standing.", "steps": 50, "seed": 42, "autoplay": true, "contact_refine": "heel_toe" }`.
- `POST /api/source`: `{ "source": "video" }`, `{ "source": "idle" }`, or `{ "source": "trace", "motion_id": "JOB_ID" }`.
- `POST /api/cancel`: cancel generation; reference output continues.
- `POST /api/pose`: a `gmr-g1-v1` pose packet.
- `GET /api/jobs/JOB_ID/log`: generation log.
- `GET /api/jobs/JOB_ID/preview`: named body positions, tree edges and timestamps for local 3D playback; does not change deployment source.
- `GET /api/jobs/JOB_ID/training.npz`: download the completed wbc-mjlab training clip.
- `WS /ws/poses`: output packets at the configured frequency, with status twice per second. Messages are `{ "type": "pose", "packet": {...} }`; incoming messages of the same form update the video/reference source.

Slow web viewers receive the latest packet without accumulating a queue or delaying UDP output. Generation uses separate processes; cancellation terminates its process group. One generation runs at a time to bound memory use.

## Motion and WBC contract

TRACE's `motion_final_qpos.npz` contains named G1 joints, xyz position in metres, wxyz quaternion, joint angles in radians, and FPS. The adapter maps joints by name, aligns initial planar position/heading, linearly interpolates translation/joints, and uses quaternion SLERP. Playback follows monotonic wall time.

Generated packets set `reference_kind: "motion"` and carry a `wbc_reference` block (`schema: "wbc-torso-v1"`, `body_name: "torso_link"`). It includes `position_world`, `quaternion_wxyz`, `linear_velocity_world` and `angular_velocity_world`, computed using the exact same 50 Hz FK and derivatives as the training NPZ. The receiver interpolates this anchor and converts all velocity/gravity components to the reference torso frame. This includes vertical and waist-induced motion, without re-estimating velocity from filtered height. Legacy packets without this block use native torso FK and root/joint derivatives. A tracked camera base now contributes its measured translation and rotation derivatives. TeleopArms deliberately replaces all leg/waist/root goals with the WBC stance and ignores the generated torso block.

See [the deployment audit](../deploy/WBC_REFERENCE_AUDIT.md) for all corrected mismatches and remaining constraints.

WBC tracks joints, torso height, gravity, and base velocity; it does not track an absolute world position or guarantee execution of arbitrary generated motions. Validate new motions in MuJoCo before physical execution.

When no poses arrive, the controller stays in its teleop state indefinitely with the initial reference active. Actual receiver/model errors retain Walk recovery. DDS state checks and Passive controls are unchanged.

Models are stored in ignored `TRACE/models/`; job inputs, logs, and clips in `.cache/trace-jobs/`. Completed training exports are collected in ignored `motions/`. Restarting the server preserves completed clips and marks unfinished jobs as interrupted. See the root README's [private data guidance](../README.md#private-data-and-git) before sharing a checkout.

## Export mocaps for wbc-mjlab training

Every successful generation also writes **`motion_wbc_mjlab.npz`** after smoothing and contact refinement. This is a kinematic reference for learning, not a recording of WBC's motor commands. The original 20 Hz TRACE export remains available for playback.

The training export is resampled to **50 Hz**, matching the training control step of 0.02 s. It contains `fps` (shape `[1]`), `robot`, `joint_names`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`. Joint arrays are `[T,29]`; body arrays include all 30 G1 bodies in training-asset order, excluding MuJoCo world. Quaternions use **wxyz**, positions are metres, joint angles radians, and velocities are world-frame values at **link origins**, not centres of mass. Names and provenance are stored as Unicode/numeric arrays, readable without pickle.

FK uses the exact G1 joint/body/inertial tree from `wbc-mjlab`'s `robots/g1/xmls/g1.xml`, bundled without rendering assets in `assets/g1_training_kinematics.xml`. Source revision and hashes are in `assets/SOURCE.json`. Export preserves source translation and heading and uses central differences with one-sided clip boundaries, matching the training motion library. No extra dynamics rollout, clipping, height debias, or smoothing is added.

Files are saved in both locations:

```text
.cache/trace-jobs/JOB_ID/output/motion_wbc_mjlab.npz
motions/npz/JOB_ID.npz
motions/metadata/JOB_ID.json
```

The dataset copy is outside `.cache/` so clearing generation caches does not remove the training collection. Every successful generation automatically adds its NPZ to `motions/npz/`; the entire `motions/` folder is ignored by Git. JSON metadata records the prompt, seed, generation settings, source hash, and robot kinematics hash. A completed job has a **Download training NPZ** button in the web library (`Скачать NPZ для обучения`). Intermediate planner/diffusion NPZ files stay in the ignored job cache.

To change the collection folder:

```bash
./uv run --locked --extra motion python -m motion_teleop.server \
  --training-dataset-dir /path/to/trace_dataset
```

To convert previously generated jobs without running TRACE again:

```bash
./uv run --locked --extra motion python -m motion_teleop.training_export \
  --jobs-dir .cache/trace-jobs
```

Or export one existing TRACE clip:

```bash
./uv run --locked --extra motion python -m motion_teleop.training_export \
  --source /path/to/motion_final_qpos.npz --clip-id my_motion
```

Load the collection directly in the training repository; no copy or conversion inside it is required:

```bash
cd /path/to/wbc-mjlab
uv run wbc-mjlab-train --task Wbc-G1 \
  --dataset-path /path/to/g1_deploy/motions
```

For fine-tuning, select the task and observation/action configuration of your existing checkpoint and add the repository's resume options: `--agent.resume True --agent.load-run YOUR_RUN --agent.load-checkpoint YOUR_CHECKPOINT.pt`. A single exported file also works with `--motion-file /path/to/clip.npz`. The NPZ loader uses the environment control step, so retain 50 Hz / 0.02 s when using these exports.

## Checks without a robot

```bash
./uv run --locked --extra motion python -m unittest discover -s motion_teleop/tests -v
ctest --test-dir deploy/g1/build --output-on-failure
```

Checks cover 20→50 Hz resampling, rotations, named joints, missing camera joints, source selection, HTTP/WebSocket, and the real native interpolator/WBC observation contract. They do not initialize DDS.

Upstream: [TRACE](https://github.com/Multimodal-Perception-Action/TRACE), Apache-2.0. Original sources and model manifest remain in the submodule.
