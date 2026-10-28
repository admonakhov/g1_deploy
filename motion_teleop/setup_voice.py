"""Download the official lightweight Russian Vosk model for offline speech."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from .voice import DEFAULT_MODEL

URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
SHA256 = "961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4"


def main():
    parent = DEFAULT_MODEL.parent
    parent.mkdir(parents=True, exist_ok=True)
    if (DEFAULT_MODEL / "am/final.mdl").is_file():
        print(f"Russian voice model is installed: {DEFAULT_MODEL}")
        return
    archive = parent / (DEFAULT_MODEL.name + ".zip")
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        print("Downloading the official Russian Vosk model (~45 MB)…", flush=True)
        temporary = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(URL, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise ValueError("Vosk model checksum mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = Path(temporary, member.filename).resolve()
                if not target.is_relative_to(Path(temporary).resolve()) or member.file_size > 128 * 1024 * 1024:
                    raise ValueError("Invalid model archive path or size")
            bundle.extractall(temporary)
        source = Path(temporary) / DEFAULT_MODEL.name
        if not (source / "am/final.mdl").is_file():
            raise ValueError("The Vosk archive does not contain the expected model")
        shutil.move(str(source), str(DEFAULT_MODEL))
    print(f"Russian voice input is ready: {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
