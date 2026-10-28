"""Bounded offline Russian speech recognition, isolated from the output loop."""
import asyncio
from importlib.util import find_spec
import io
import json
from pathlib import Path
import re
import sys
import wave

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / ".cache/voice/vosk-model-small-ru-0.22"
MAX_SECONDS = 20
MAX_AUDIO_BYTES = 16000 * 2 * MAX_SECONDS + 4096


def command_for(text):
    """Exact control phrases only; longer descriptions stay motion requests."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Не удалось распознать речь. Повторите команду ближе к микрофону.")
    normalized = re.sub(r"[^а-яa-z0-9 ]", " ", text.lower().replace("ё", "е"))
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"^(?:робот|джи один) ", "", normalized)
    normalized = re.sub(r"^пожалуйста | пожалуйста$", "", normalized)
    if not normalized:
        raise ValueError("Не удалось распознать речь. Повторите команду.")
    phrases = {
        "cancel": ("отмена", "отмени генерацию", "отменить генерацию", "отмени создание движения"),
        "pause": ("стоп", "остановись", "останови движение", "пауза"),
        "resume": ("продолжи", "продолжай", "продолжи движение", "продолжить движение"),
        "video": ("видео", "камера", "включи видео", "включи камеру", "режим видео", "режим камеры"),
        "idle": ("стойка", "исходная стойка", "исходная поза", "включи стойку", "вернись в исходную стойку"),
        "replay": ("повтори", "повтори движение", "повтори последнее движение", "включи последнее движение"),
    }
    for command, alternatives in phrases.items():
        if normalized in alternatives:
            return command
    return "generate"


def validate_audio(data):
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise ValueError("Запись должна длиться от 0.2 до 20 секунд.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("Нужен WAV: PCM 16 бит, 16 кГц, моно.")
            frames = audio.getnframes()
            if not 3200 <= frames <= 16000 * MAX_SECONDS or len(audio.readframes(frames)) != frames * 2:
                raise ValueError("Запись должна длиться от 0.2 до 20 секунд и не быть обрезанной.")
    except (wave.Error, EOFError) as error:
        raise ValueError("Не удалось прочитать WAV-запись.") from error


class VoiceRecognizer:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = Path(model).absolute()
        self.busy = False

    def status(self):
        installed = find_spec("vosk") is not None
        ready = all((self.model / name).is_file() for name in ("am/final.mdl", "conf/model.conf", "graph/HCLr.fst", "graph/Gr.fst"))
        return dict(available=installed and ready, installed=installed, model_ready=ready,
                    language="ru", engine="Vosk", max_seconds=MAX_SECONDS, busy=self.busy)

    async def transcribe(self, data):
        validate_audio(data)
        if self.busy:
            raise ValueError("Распознавание уже выполняется. Дождитесь результата.")
        if not self.status()["available"]:
            raise ValueError("Установите голосовой ввод: ./uv run --locked --extra motion python -m motion_teleop.setup_voice")
        self.busy = True
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "motion_teleop.voice_worker", "--model", str(self.model),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            output, _ = await asyncio.wait_for(process.communicate(data), timeout=30.)
            if process.returncode:
                raise ValueError("Ошибка распознавания. Проверьте установку русской модели Vosk.")
            result = json.loads(output)
            text = result["text"].strip()
            return dict(text=text, command=command_for(text), language="ru", engine="Vosk")
        except asyncio.TimeoutError as error:
            raise ValueError("Распознавание заняло слишком много времени. Повторите короткую команду.") from error
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()
            self.busy = False
