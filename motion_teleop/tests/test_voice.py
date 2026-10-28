import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import wave

from aiohttp.test_utils import TestClient, TestServer

from motion_teleop.server import create_app, parser, VOICE, BRIDGE
from motion_teleop.voice import command_for, validate_audio, VoiceRecognizer, MAX_AUDIO_BYTES


def recording(seconds=.5, rate=16000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * int(rate * seconds) * channels)
    return output.getvalue()


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    def test_commands_and_descriptions_cannot_accidentally_match_controls(self):
        for text, expected in (("Робот, стоп!", "pause"), ("ПРОДОЛЖАЙ", "resume"),
                               ("отмени генерацию", "cancel"), ("включи камеру", "video"),
                               ("исходная стойка", "idle"), ("повтори последнее движение", "replay"),
                               ("пожалуйста, включи видео", "video"),
                               ("подними правую руку", "generate"),
                               ("сделай шаг, стоп и повернись", "generate"),
                               ("вернись в стойку и помаши", "generate")):
            self.assertEqual(command_for(text), expected, text)
        for invalid in (None, "", "!!!", "x" * 2001):
            with self.assertRaises(ValueError):command_for(invalid)

    def test_audio_is_bounded_and_requires_actual_pcm_frames(self):
        validate_audio(recording())
        validate_audio(recording(20))
        for invalid in (b"", b"not wav", recording(.1), recording(21), recording(rate=48000),
                        recording(channels=2), recording()[:-30], b"x" * (MAX_AUDIO_BYTES+1)):
            with self.assertRaises(ValueError):validate_audio(invalid)

    async def test_subprocess_receives_only_audio_and_returns_command(self):
        speech = VoiceRecognizer()
        process = AsyncMock(returncode=0)
        process.communicate.return_value=(json.dumps({"text":"подними правую руку"}).encode(), b"")
        with patch.object(speech, "status", return_value={"available":True}), patch("asyncio.create_subprocess_exec", return_value=process) as start:
            result = await speech.transcribe(recording())
        self.assertEqual(result["command"], "generate")
        self.assertEqual(result["language"], "ru")
        self.assertIn("motion_teleop.voice_worker", start.call_args.args)
        self.assertEqual(process.communicate.call_args.args[0], recording())
        self.assertFalse(speech.busy)

    async def test_missing_model_busy_silence_failure_and_timeout(self):
        speech = VoiceRecognizer("/nonexistent/voice-model")
        self.assertFalse(speech.status()["available"])
        with self.assertRaisesRegex(ValueError, "Установите"):await speech.transcribe(recording())
        speech.busy = True
        with self.assertRaisesRegex(ValueError, "уже"):await speech.transcribe(recording())
        speech.busy = False
        for code, reply in ((0,b'{"text":""}'), (1,b"")):
            process=AsyncMock(returncode=code)
            process.communicate.return_value=(reply,b"")
            with patch.object(speech, "status", return_value={"available":True}), patch("asyncio.create_subprocess_exec", return_value=process):
                with self.assertRaises(ValueError):await speech.transcribe(recording())
            self.assertFalse(speech.busy)
        process=AsyncMock(returncode=None)
        process.kill=lambda: setattr(process, "returncode", -9)
        process.communicate.side_effect=[asyncio.TimeoutError(), (b"",b"")]
        with patch.object(speech, "status", return_value={"available":True}), patch("asyncio.create_subprocess_exec", return_value=process):
            with self.assertRaisesRegex(ValueError, "слишком много времени"):await speech.transcribe(recording())
        self.assertEqual(process.returncode, -9)
        self.assertFalse(speech.busy)

    async def test_http_transcription_is_read_only_and_does_not_block_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args=parser().parse_args(["--jobs",folder,"--models",folder])
            class Output:
                count=0
                def publish(self, packet):self.count+=1
                def close(self):pass
            output=Output()
            app=create_app(args,output)
            async with TestClient(TestServer(app)) as client:
                async def transcribe(data):
                    validate_audio(data)
                    await asyncio.sleep(.25)
                    return dict(text="включи видео", command="video", language="ru", engine="Vosk")
                before=app[BRIDGE].epoch
                with patch.object(app[VOICE], "transcribe", side_effect=transcribe):
                    response=await client.post("/api/voice/transcribe",data=recording(),headers={"Content-Type":"audio/wav"})
                self.assertEqual(response.status,200)
                self.assertEqual((await response.json())["command"],"video")
                self.assertEqual(app[BRIDGE].epoch,before)
                self.assertGreaterEqual(output.count,10)
                self.assertEqual((await client.post("/api/voice/transcribe",json={})).status,400)
                self.assertEqual((await client.post("/api/voice/transcribe",data=recording(),headers={"Content-Type":"audio/wav","Origin":"https://other.test"})).status,403)
                self.assertEqual((await client.post("/api/voice/transcribe",data=b"x"*(MAX_AUDIO_BYTES+1),headers={"Content-Type":"audio/wav"})).status,413)


if __name__ == "__main__":unittest.main()
