# Shared transcription, translation and subtitle helpers for the Colab notebooks.

import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests


@dataclass(frozen=True)
class TranslationConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    max_workers: int = 5
    max_retries: int = 2
    timeout: float = 30
    retry_delay: float = 1

    def __post_init__(self):
        if not self.api_key.strip() or "*" in self.api_key:
            raise ValueError("请填写有效的翻译 API Key。")
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or "*" in url.netloc:
            raise ValueError("BASE_URL 必须是完整的 HTTP(S) API 地址。")
        if url.query or url.fragment or url.username or url.password:
            raise ValueError("BASE_URL 不能包含查询参数、片段或登录凭据。")
        if not self.model.strip():
            raise ValueError("请填写翻译模型名称。")
        for name, minimum in (("max_workers", 1), ("max_retries", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} 必须是 >= {minimum} 的整数。")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout 必须是有限正数。")
        if not math.isfinite(self.retry_delay) or self.retry_delay < 0:
            raise ValueError("retry_delay 必须是有限非负数。")


@dataclass
class TranslationResult:
    translations: dict[int, str]
    failed_indices: list[int]
    total: int


class TranslationAPIError(RuntimeError):
    """A permanent API error that requires changing configuration or the request."""


class _RetryableError(Exception):
    def __init__(self, delay=0):
        self.delay = delay


def resolve_audio_path(folder, filename):
    """Resolve exactly the requested file; never rename it or choose a namesake."""
    if (
        not filename
        or filename in {".", ".."}
        or any(c in filename for c in "/\\")
        or Path(filename).name != filename
    ):
        raise ValueError("AUDIO_FILENAME 只能填写文件名，目录请填写在 DRIVE_FOLDER 中。")
    path = (Path(folder).expanduser() / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到音频文件：{path}")
    return path


def _text(value):
    return " ".join(value.split())


def validate_segments(data):
    """Reject malformed Whisper output before calling a paid translation API."""
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Whisper JSON 缺少 segments 列表。")
    segments = []
    for i, segment in enumerate(data["segments"]):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError(f"字幕片段 {i} 缺少有效文本。")
        start, end = segment.get("start"), segment.get("end")
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)):
            raise ValueError(f"字幕片段 {i} 的时间戳无效。")
        if start < 0 or end < start:
            raise ValueError(f"字幕片段 {i} 的起止时间无效。")
        segments.append({"start": start, "end": end, "text": _text(segment["text"])})
    return segments


def transcribe_audio(audio_path, model="turbo"):
    """Run Whisper in an isolated directory and only read a successful run's JSON."""
    source = Path(audio_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"不是音频文件：{source}")
    with tempfile.TemporaryDirectory(prefix="audio2text-") as folder:
        workdir = Path(folder)
        local_audio = workdir / ("audio" + source.suffix)
        shutil.copy2(source, local_audio)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "whisper",
                str(local_audio),
                "--model",
                model,
                "--language",
                "English",
                "--output_format",
                "json",
                "--output_dir",
                str(workdir),
                "--verbose",
                "False",
            ],
            check=True,
        )
        with (workdir / "audio.json").open(encoding="utf-8") as stream:
            return validate_segments(json.load(stream))


def _timestamp_units(seconds, scale):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("字幕时间戳必须是有限非负数。")
    # Round the entire timestamp first so 59.999 becomes the next minute.
    return int(round(seconds * scale))


def seconds_to_lrc(seconds):
    minutes, remainder = divmod(_timestamp_units(seconds, 100), 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]"


def seconds_to_srt(seconds):
    seconds, milliseconds = divmod(_timestamp_units(seconds, 1000), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def build_lrc(segments, translations=None):
    translations = translations or {}
    lines = ["[ti:Auto Generated]", "[ar:Whisper]", ""]
    for i, segment in enumerate(segments):
        english = _text(segment["text"])
        if not english:
            continue
        timestamp = seconds_to_lrc(segment["start"])
        lines.append(timestamp + english)
        chinese = _text(translations.get(i, ""))
        if chinese:
            lines.append(timestamp + chinese)
    return "\n".join(lines) + "\n"


def build_srt(segments, translations=None):
    translations = translations or {}
    blocks = []
    for i, segment in enumerate(segments):
        english = _text(segment["text"])
        if not english:
            continue
        text = english
        chinese = _text(translations.get(i, ""))
        if chinese:
            text += "\n" + chinese
        start = seconds_to_srt(segment["start"])
        end = seconds_to_srt(segment["end"])
        blocks.append(f"{len(blocks) + 1}\n{start} --> {end}\n{text}\n\n")
    return "".join(blocks)


def _retry_after(value):
    if not value:
        return 0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0
    return min(60, max(0, delay)) if math.isfinite(delay) else 0


def _request_translation(session, text, config):
    with session.post(
        config.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json={
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate the following subtitle to Simplified Chinese. "
                        "Output only the translation, without explanations."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
        },
        timeout=config.timeout,
        allow_redirects=False,
    ) as response:
        status = response.status_code
        if status in {408, 429} or 500 <= status < 600:
            raise _RetryableError(_retry_after(response.headers.get("Retry-After")))
        if not 200 <= status < 300:
            raise TranslationAPIError(f"翻译 API 返回 HTTP {status}，请检查 API Key、地址、模型和配额。")
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty translation")
        except (ValueError, KeyError, IndexError, TypeError):
            raise _RetryableError() from None
        return _text(content)


def translate_segments(segments, config=None, *, output_chinese=True, progress=None):
    pending = {i: _text(s["text"]) for i, s in enumerate(segments) if s["text"].strip()}
    if not output_chinese or not pending:
        return TranslationResult({}, [], len(pending))
    if config is None:
        raise ValueError("启用中文输出时需要 TranslationConfig。")

    stop = threading.Event()
    local = threading.local()
    sessions = []
    session_lock = threading.Lock()

    def translate(text):
        if stop.is_set():
            return None
        if not hasattr(local, "session"):
            local.session = requests.Session()
            with session_lock:
                sessions.append(local.session)
        for attempt in range(config.max_retries + 1):
            if stop.is_set():
                return None
            try:
                return _request_translation(local.session, text, config)
            except TranslationAPIError:
                stop.set()
                raise
            except (requests.RequestException, _RetryableError) as error:
                if attempt == config.max_retries:
                    return None
                delay = max(
                    min(60, config.retry_delay * 2 ** min(attempt, 10)),
                    getattr(error, "delay", 0),
                )
                if stop.wait(delay):
                    return None

    translations = {}
    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(translate, text): i for i, text in pending.items()}
            try:
                for finished, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    if result:
                        translations[futures[future]] = result
                    if progress:
                        progress(finished, len(pending))
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
    finally:
        for session in sessions:
            session.close()

    failed = [i for i in pending if i not in translations]
    return TranslationResult(translations, failed, len(pending))


def write_text_atomic(path, content):
    """Replace the destination only after the complete UTF-8 file has been written."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".audio2text-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_subtitles(
    audio_path, segments, subtitle_format="lrc", config=None, *, output_chinese=True, progress=None
):
    """Save subtitles alongside the audio and report untranslated segments."""
    if subtitle_format not in {"lrc", "srt"}:
        raise ValueError("字幕格式只能是 lrc 或 srt。")
    segments = validate_segments({"segments": segments})
    source = Path(audio_path)
    destination = source.with_suffix("." + subtitle_format)
    if destination.resolve() == source.resolve():
        raise ValueError("字幕输出路径不能与音频路径相同。")
    if subtitle_format == "srt":
        write_text_atomic(source.with_name(source.stem + "_en.srt"), build_srt(segments))
    result = translate_segments(segments, config, output_chinese=output_chinese, progress=progress)
    render = build_lrc if subtitle_format == "lrc" else build_srt
    write_text_atomic(destination, render(segments, result.translations))
    return destination, result
