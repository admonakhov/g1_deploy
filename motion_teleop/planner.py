"""TRACE event plans from OpenAI Responses API; no local LLM or torch import."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
API_URL = "https://api.openai.com/v1/responses"
EVENT_FIELDS = ("end", "left_support", "right_support", "root_vertical", "root_height", "left_hand", "right_hand")


class PlannerError(RuntimeError):
    """A user-facing error that contains no API credentials or HTTP error body."""


class PlannerSettings:
    """Server-side settings; credentials are deliberately absent from status/files."""
    def __init__(self, provider="local", model=DEFAULT_MODEL, timeout=90., api_key=None, codex_bin=None, codex_model="", codex_timeout=180.):
        self.provider, self.model = "local", DEFAULT_MODEL
        self._api_key = os.environ.get("OPENAI_API_KEY", "").strip() if api_key is None else api_key
        if not math.isfinite(timeout) or not 1 <= timeout <= 300:
            raise ValueError("OpenAI timeout must be 1..300 seconds")
        self.timeout = float(timeout)
        if not math.isfinite(codex_timeout) or not 1 <= codex_timeout <= 600:
            raise ValueError("Codex timeout must be 1..600 seconds")
        self.codex_bin, self.codex_model, self.codex_timeout = codex_bin, "", float(codex_timeout)
        from .codex_planner import find_codex
        executable=find_codex(codex_bin)
        self.codex_status=dict(available=bool(executable),authenticated=None,checked=False,binary=executable,
                               message="Check sign-in to use Codex subscription" if executable else "Codex CLI is not installed")
        self.update(dict(provider=provider, model=model, api_key=self._api_key, codex_model=codex_model))

    def update(self, data):
        if not isinstance(data, dict) or set(data) - {"provider", "model", "api_key", "clear_api_key", "codex_model"}:
            raise ValueError("Expected planner provider, model and optional API key")
        provider, model = data.get("provider", self.provider), data.get("model", self.model)
        if provider not in ("local", "openai", "codex"):
            raise ValueError("Planner must be local, openai or codex")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", model):
            raise ValueError("Invalid OpenAI model name")
        codex_model=data.get("codex_model",self.codex_model)
        if not isinstance(codex_model,str) or (codex_model and not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}",codex_model)):
            raise ValueError("Invalid Codex model name")
        if type(data.get("clear_api_key", False)) is not bool:
            raise ValueError("clear_api_key must be boolean")
        key = data.get("api_key", "")
        if not isinstance(key, str) or len(key) > 4096 or any(c.isspace() for c in key):
            raise ValueError("API key must be a single non-whitespace value")
        if key and data.get("clear_api_key"):
            raise ValueError("Cannot set and clear the API key together")
        self.provider, self.model, self.codex_model = provider, model, codex_model
        if key:
            self._api_key = key
        elif data.get("clear_api_key"):
            self._api_key = ""

    def status(self):
        return dict(provider=self.provider, model=self.model, api_key_configured=bool(self._api_key),
                    timeout_s=self.timeout,codex_model=self.codex_model,codex=self.codex_status)

    def refresh_codex(self):
        from .codex_planner import probe_codex
        self.codex_status=probe_codex(self.codex_bin)
        return self.codex_status

    def snapshot(self):
        if self.provider == "openai" and not self._api_key:
            raise ValueError("OpenAI API key is missing. Enter it in the planner panel or set OPENAI_API_KEY before starting the server.")
        if self.provider == "codex":
            from .codex_planner import find_codex
            binary=find_codex(self.codex_bin)
            if not binary:
                raise ValueError("Codex CLI is missing. Install it or set --codex-bin")
            return dict(planner_provider="codex",planner_model=self.codex_model or "Codex CLI default",
                        codex_model=self.codex_model,codex_bin=binary,planner_timeout_s=self.codex_timeout), ""
        return dict(planner_provider=self.provider, planner_model=self.model if self.provider=="openai" else "Qwen3-4B-Instruct-2507 + TRACE LoRA", planner_timeout_s=self.timeout), self._api_key if self.provider == "openai" else ""


def _trace_compile():
    trace = str(ROOT / "TRACE")
    if trace not in sys.path:
        sys.path.insert(0, trace)
    from trace_core.llm_compile import canonical_parse, compile_response
    from trace_core.llm_plan import SYSTEM
    return canonical_parse, compile_response, SYSTEM


def plan_schema(min_frames=20):
    properties = {"end": dict(type="integer", minimum=1, maximum=128)}
    for field, maximum in zip(EVENT_FIELDS[1:], (1, 1, 2, 2, 3, 3)):
        properties[field] = dict(type="integer", enum=list(range(-1, maximum + 1)))
    return dict(type="object", additionalProperties=False, required=["frames", "events"], properties=dict(
        frames=dict(type="integer", minimum=min_frames, maximum=512),
        events=dict(type="array", minItems=1, maxItems=128, items=dict(type="object", additionalProperties=False,
                    required=list(EVENT_FIELDS), properties=properties))))


def compile_api_plan(value, min_frames=20):
    """Convert typed API events to unchanged TRACE schema, then run its validator."""
    if not isinstance(value, dict) or set(value) != {"frames", "events"} or not isinstance(value["events"], list):
        raise ValueError("Expected frames and events")
    rows=[]
    for event in value["events"]:
        if not isinstance(event, dict) or set(event) != set(EVENT_FIELDS):
            raise ValueError("Each event must have exactly the seven named TRACE fields")
        rows.append([event[k] for k in EVENT_FIELDS])
    canonical_parse, compile_response, _ = _trace_compile()
    native = dict(frames=value["frames"], events=rows)
    obj, canonicalized = canonical_parse(json.dumps(native, allow_nan=False))
    if obj["frames"] < min_frames:
        raise ValueError("Motion is too short for the configured boundary frames")
    probability, known, status = compile_response(json.dumps(obj))
    if status["failed"]:
        raise ValueError("TRACE rejected the motion event timeline")
    status["canonicalized"] |= canonicalized
    return obj, probability, known, status


def response_text(response):
    if not isinstance(response, dict):
        raise PlannerError("OpenAI returned an invalid response envelope")
    texts=[]
    output=response.get("output", [])
    if not isinstance(output, list):
        raise PlannerError("OpenAI returned an invalid output list")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content=item.get("content", [])
        if not isinstance(content, list):
            raise PlannerError("OpenAI returned an invalid message")
        for part in content:
            if not isinstance(part, dict):
                raise PlannerError("OpenAI returned an invalid content block")
            if part.get("type") == "refusal":
                raise PlannerError("OpenAI declined this motion request; change the description")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if response.get("status") != "completed":
        raise PlannerError("OpenAI response was incomplete; no motion plan was accepted")
    if not texts:
        raise PlannerError("OpenAI returned no motion plan")
    return "".join(texts)


def _request(body, api_key, timeout):
    if not api_key:
        raise PlannerError("OpenAI API key is missing")
    request = Request(API_URL, data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise PlannerError("OpenAI response exceeded the planner response limit")
        return json.loads(raw)
    except HTTPError as error:
        messages={401:"API key was rejected",403:"access denied for this project or region",404:"model or endpoint is unavailable",429:"rate limit or API quota exceeded"}
        raise PlannerError(f"OpenAI HTTP {error.code}: {messages.get(error.code, 'request failed')}") from None
    except (URLError, TimeoutError, OSError):
        raise PlannerError("OpenAI connection failed or timed out; the current reference stream continues") from None
    except (ValueError, UnicodeError):
        raise PlannerError("OpenAI returned an invalid JSON response") from None


def predict_openai_plan(config, output):
    """Write the exact plan.json/plan.npz consumed by upstream TRACE diffusion."""
    import numpy as np
    output = Path(output)
    _, _, system = _trace_compile()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = config.get("planner_model", DEFAULT_MODEL)
    settings = PlannerSettings("openai", model, float(config.get("planner_timeout_s", 90.)), api_key=key)
    settings.snapshot()  # Validate credentials without saving them.
    minimum = 2 * int(config.get("boundary_frames", 8)) + 4
    instructions = (system + "\nUse named event objects matching the supplied JSON schema instead of positional event rows. "
        f"Choose {minimum}..512 frames at 20 FPS. Every end is an exclusive 4-frame patch index, NOT a frame index. "
        "The last end must equal ceil(frames/4); all ends must strictly increase. "
        "Merge adjacent identical states. Preserve the requested body side, movement order and duration. "
        "Root height means pelvis height above ground, not head or torso height. Neutral G1 standing pelvis height is about 0.8 m: root_height=1. Unless a crouch, jump or travel is requested, keep root_vertical=1 and both feet in stable support. "
        "Describe one physically coherent G1 realization. You receive a motion description only; do not output prose.")
    body = dict(model=model, instructions=instructions, input=config["text"], store=False, max_output_tokens=8192,
                text=dict(format=dict(type="json_schema", name="trace_motion_events", strict=True, schema=plan_schema(minimum))))
    began=time.monotonic()
    for attempt in range(2):
        response=_request(body, key, settings.timeout)
        text=response_text(response)
        try:
            value=json.loads(text)
            obj, probability, known, status=compile_api_plan(value, minimum)
            break
        except (ValueError, TypeError, KeyError, IndexError):
            if attempt:
                raise PlannerError("OpenAI plan failed TRACE timeline validation twice; no motion was generated") from None
            body=deepcopy(body)
            body["input"]=config["text"] + "\nYour previous plan failed timeline validation. Check frame count, strictly increasing patch ends, complete coverage and category ranges. Return a corrected plan."
    # Store only the accepted, canonical TRACE representation. No headers/key are serialized.
    (output/"planner_response.txt").write_text(json.dumps(obj, ensure_ascii=False)+"\n")
    (output/"plan.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
    np.savez(output/"plan.npz", events=probability, known=known, frames=obj["frames"])
    usage=response.get("usage") or {}
    status.update(provider="openai", model=model, response_id=response.get("id"), attempts=attempt+1,
                  seconds=time.monotonic()-began,
                  usage={k:usage[k] for k in ("input_tokens","output_tokens","total_tokens") if k in usage})
    (output/"planner_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n")
    print(f"OpenAI plan accepted: {obj['frames']} frames, {len(obj['events'])} events, model={model}", flush=True)


def check_motion_models(models, provider):
    """The remote planner must not require local Qwen planner assets."""
    models=Path(models)
    if provider not in ("local", "openai", "codex"):
        raise ValueError("Unknown language planner")
    if provider == "local":
        if str(ROOT/"TRACE") not in sys.path:
            sys.path.insert(0,str(ROOT/"TRACE"))
        from run import check_models
        check_models(models,"standing",custom_both_references=True)
        return
    required=[models/"generator_boundary.pt",models/"generation_stats.npz"]
    required.extend(models/"text_embedding"/name for name in ("config.json","tokenizer.json","tokenizer_config.json","model.safetensors.index.json"))
    index=models/"text_embedding/model.safetensors.index.json"
    if index.is_file():
        required.extend(models/"text_embedding"/name for name in set(json.loads(index.read_text())["weight_map"].values()))
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("TRACE motion/embedding model files are missing: " + ", ".join(missing))
