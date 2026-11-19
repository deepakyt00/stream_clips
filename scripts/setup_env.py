#!/usr/bin/env python3
"""One-time setup for stream-clips. Uses only the Python standard library, so it runs
before anything is installed. Safe to run again. Ends with the word READY.

    macOS:    python3 scripts/setup_env.py
    Windows:  python  scripts\\setup_env.py
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
# STREAMCLIPS_ENV moves the Python env and models off the home drive (e.g. D:\StreamClipsEnv).
CUSTOM_ENV = os.environ.get("STREAMCLIPS_ENV")
ENV_DIR = Path(CUSTOM_ENV) if CUSTOM_ENV else Path.home() / ".streamclips"
VENV = ENV_DIR / "venv"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
MODELS = ENV_DIR / "models"
DATA = Path(os.environ.get("STREAMCLIPS_HOME", Path.home() / "StreamClips"))
HERE = Path(__file__).resolve().parent

PACKAGES = ["faster-whisper", "opencv-python-headless", "pillow", "numpy", "yt-dlp[default]"]
FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def good(msg):
    print(f"  [ok]   {msg}", flush=True)


def bad(msg, fix):
    print(f"  [MISSING] {msg}\n          fix: {fix}", flush=True)


def hint(mac, win, linux):
    return mac if IS_MAC else win if IS_WIN else linux


def existing(path):
    """The nearest folder that exists at or above `path` (to measure its drive)."""
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def child_env():
    """With STREAMCLIPS_ENV set, keep the download caches on that drive too."""
    env = dict(os.environ)
    if CUSTOM_ENV:
        cache = ENV_DIR / "cache"
        env.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
        env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
        env.setdefault("HF_HOME", str(cache / "huggingface"))
    return env


def check_tools():
    missing = False
    print("Checking tools")
    if sys.version_info < (3, 9):
        bad(f"Python {sys.version.split()[0]} is too old (need 3.9+)",
            hint("brew install python@3.12", "winget install Python.Python.3.12", "install python3.12"))
        missing = True
    else:
        good(f"Python {sys.version.split()[0]}")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            good(tool)
        else:
            bad(f"{tool} not found",
                hint("brew install ffmpeg", "winget install Gyan.FFmpeg  (then open a new terminal)",
                     "sudo apt install ffmpeg"))
            missing = True
    if shutil.which("deno") or shutil.which("node") or shutil.which("bun"):
        good("JavaScript runtime (YouTube downloads need one)")
    else:
        bad("no deno or node (YouTube downloads need one)",
            hint("brew install deno", "winget install DenoLand.Deno", "curl -fsSL https://deno.land/install.sh | sh"))
        missing = True
    brains = [b for b in ("claude", "codex", "gemini", "opencode") if shutil.which(b)]
    if brains:
        good("AI CLI for picking moments: " + ", ".join(brains))
    else:
        print("  [note] no claude/codex/gemini/opencode CLI on PATH: moments will be picked by the "
              "agent itself (--brain agent)")
    where = existing(ENV_DIR)
    free = shutil.disk_usage(where).free / 1e9
    if free < 3:
        bad(f"only {free:.1f} GB free on the drive of {ENV_DIR}",
            "free up space (setup needs ~1 GB, clipping ~0.15 GB per stream hour), or install on "
            "another drive by setting STREAMCLIPS_ENV, e.g. " + hint(
                "export STREAMCLIPS_ENV=/Volumes/Big/streamclips-env",
                'setx STREAMCLIPS_ENV "D:\\StreamClipsEnv"  (then open a new terminal)',
                "export STREAMCLIPS_ENV=/mnt/big/streamclips-env"))
        missing = True
    else:
        good(f"disk: {free:.0f} GB free for {ENV_DIR}")
    return not missing


def make_venv():
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    if VENV_PY.exists():
        good("private Python env exists")
        return
    print(f"Creating a private Python env in {ENV_DIR} (keeps your system Python clean)...")
    uv = shutil.which("uv")
    if uv:
        for version in ("3.12", "3.11", None):
            cmd = [uv, "venv", "-q", str(VENV)] + (["--python", version] if version else [])
            if subprocess.run(cmd, env=child_env()).returncode == 0 and VENV_PY.exists():
                break
    if not VENV_PY.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    good("private Python env created")


def install_packages():
    print("Installing Python packages (Whisper, OpenCV, Pillow, yt-dlp)... this takes a minute")
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "-q", "--python", str(VENV_PY), "-U", *PACKAGES]
    else:
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q", "-U", "pip"], check=True,
                       env=child_env())
        cmd = [str(VENV_PY), "-m", "pip", "install", "-q", "-U", *PACKAGES]
    subprocess.run(cmd, check=True, env=child_env())
    good("Python packages")


def download_models():
    print("Downloading Whisper speech models (tiny + base, ~220 MB, first time only)...")
    code = (
        "import sys\n"
        "from faster_whisper import WhisperModel\n"
        "for n in ('tiny', 'base'):\n"
        "    WhisperModel(n, device='cpu', compute_type='int8', download_root=sys.argv[1])\n"
    )
    (MODELS / "whisper").mkdir(parents=True, exist_ok=True)
    subprocess.run([str(VENV_PY), "-c", code, str(MODELS / "whisper")], check=True, env=child_env())
    good("Whisper models (tiny = scanning the stream, base = captions)")
    face = MODELS / "face_yunet.onnx"
    if not face.exists() or face.stat().st_size < 10000:
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, face)
        except Exception as exc:  # noqa: BLE001
            print(f"  [note] face model download failed ({exc}); a simpler face finder will be used")
    if face.exists() and face.stat().st_size > 10000:
        good("face finder (OpenCV YuNet)")


def make_data_dir():
    DATA.mkdir(parents=True, exist_ok=True)
    watch = DATA / "watchlist.txt"
    if not watch.exists():
        shutil.copy(HERE.parent / "watchlist.txt", watch)
    good(f"output folder: {DATA}")
    good(f"watchlist: {watch}")


def main():
    print("stream-clips setup\n")
    if not check_tools():
        print("\nInstall the MISSING items above, open a new terminal, and run setup again.")
        sys.exit(1)
    make_venv()
    install_packages()
    download_models()
    make_data_dir()
    launcher = HERE / "streamclips.py"
    py = "python" if IS_WIN else "python3"
    print(f"\nREADY. Try:  {py} \"{launcher}\" find --topic \"GTA 6\"")


if __name__ == "__main__":
    main()
