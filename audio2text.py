# Shared transcription, translation and subtitle helpers for the Colab notebooks.

import hashlib
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
    reused: int = 0


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


def _audio_identity(source):
    stat = source.stat()
    return {"name": source.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_progress(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError):
        raise ValueError(f"进度文件损坏，请备份后删除再重试：{path}") from None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"进度文件格式不支持，请备份后删除再重试：{path}")
    return data


def transcribe_audio(audio_path, model="turbo", *, save_progress=True):
    """Run Whisper in an isolated directory and only read a successful run's JSON."""
    source = Path(audio_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"不是音频文件：{source}")
    identity = _audio_identity(source)
    cache_path = source.with_name(source.name + ".transcript.json")
    cached = _read_progress(cache_path) if save_progress else None
    if cached and cached.get("audio") == identity and cached.get("model") == model:
        return validate_segments(cached)
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
            segments = validate_segments(json.load(stream))
    if save_progress:
        write_text_atomic(
            cache_path,
            json.dumps(
                {
                    "version": 1,
                    "audio": identity,
                    "model": model,
                    "segments": segments,
                },
                ensure_ascii=False,
            ),
        )
    return segments


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


def translate_segments(
    segments, config=None, *, output_chinese=True, progress=None, initial_translations=None, on_success=None
):
    texts = {i: _text(s["text"]) for i, s in enumerate(segments) if s["text"].strip()}
    if not output_chinese or not texts:
        return TranslationResult({}, [], len(texts))
    translations = {}
    for i, value in (initial_translations or {}).items():
        if type(i) is not int or i not in texts or not isinstance(value, str) or not value.strip():
            raise ValueError("已有翻译包含无效的片段索引或文本。")
        translations[i] = _text(value)
    reused = len(translations)
    pending = {i: text for i, text in texts.items() if i not in translations}
    if not pending:
        return TranslationResult(translations, [], len(texts), reused)
    if config is None:
        raise ValueError("启用中文输出时需要 TranslationConfig。")

    stop = threading.Event()
    local = threading.local()
    sessions = []
    session_lock = threading.Lock()
    result_lock = threading.Lock()

    def translate(index, text):
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
                result = _request_translation(local.session, text, config)
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
                continue
            # Persist in the worker so successful in-flight requests also survive
            # a failure or interruption in the thread collecting results.
            with result_lock:
                translations[index] = result
                if on_success:
                    try:
                        on_success(index, result)
                    except BaseException:
                        stop.set()
                        raise
            return result

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(translate, i, text): i for i, text in pending.items()}
            try:
                for finished, future in enumerate(as_completed(futures), 1):
                    future.result()
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
    return TranslationResult(translations, failed, len(texts), reused)


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


def _import_existing_translations(path, segments, subtitle_format):
    """Migrate old exports only when every English cue and timestamp matches."""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    cues = [(i, s) for i, s in enumerate(segments) if s["text"]]
    translations = {}
    if subtitle_format == "srt":
        blocks = [block for block in content.strip().split("\n\n") if block.strip()]
        if len(blocks) != len(cues):
            return {}
        for number, ((i, segment), block) in enumerate(zip(cues, blocks), 1):
            lines = block.splitlines()
            timing = f"{seconds_to_srt(segment['start'])} --> {seconds_to_srt(segment['end'])}"
            if len(lines) not in (3, 4) or lines[:3] != [str(number), timing, segment["text"]]:
                return {}
            if len(lines) == 4 and lines[3].strip():
                translations[i] = _text(lines[3])
    else:
        lines = [
            line for line in content.splitlines() if line.strip() and not line.startswith(("[ti:", "[ar:"))
        ]
        expected = [seconds_to_lrc(s["start"]) + s["text"] for _, s in cues]
        cursor = 0
        for position, (i, segment) in enumerate(cues):
            if cursor >= len(lines) or lines[cursor] != expected[position]:
                return {}
            cursor += 1
            timestamp = seconds_to_lrc(segment["start"])
            next_english = expected[position + 1] if position + 1 < len(expected) else None
            if cursor < len(lines) and lines[cursor].startswith(timestamp) and lines[cursor] != next_english:
                chinese = _text(lines[cursor][len(timestamp) :])
                if not chinese:
                    return {}
                translations[i] = chinese
                cursor += 1
        if cursor != len(lines):
            return {}
    return translations


def _load_translation_progress(path, identity, signature, segments):
    data = _read_progress(path)
    if data is None:
        return None
    if data.get("audio") != identity or data.get("transcript") != signature:
        return {}
    values = data.get("translations")
    if not isinstance(values, dict):
        raise ValueError(f"翻译进度无效，请备份后删除再重试：{path}")
    translations = {}
    for key, value in values.items():
        if not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"翻译进度中的片段索引无效：{path}")
        index = int(key)
        if (
            index >= len(segments)
            or not segments[index]["text"]
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(f"翻译进度中的片段或文本无效：{path}")
        translations[index] = _text(value)
    return translations


def export_subtitles(
    audio_path,
    segments,
    subtitle_format="lrc",
    config=None,
    *,
    output_chinese=True,
    progress=None,
    save_progress=True,
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
    saved = {}
    save_translation = None
    if output_chinese and save_progress:
        identity = _audio_identity(source)
        signature = hashlib.sha256(
            json.dumps(
                segments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path = source.with_name(source.name + ".translations.json")
        saved = _load_translation_progress(cache_path, identity, signature, segments)
        if saved is None:
            saved = _import_existing_translations(destination, segments, subtitle_format)

        def persist_progress():
            write_text_atomic(
                cache_path,
                json.dumps(
                    {
                        "version": 1,
                        "audio": identity,
                        "transcript": signature,
                        "translations": {str(i): text for i, text in saved.items()},
                    },
                    ensure_ascii=False,
                ),
            )

        def save_translation(index, text):
            saved[index] = text
            persist_progress()

        # Verify storage is writable before making requests, and persist any
        # translations recovered from an older LRC/SRT file immediately.
        persist_progress()
    result = translate_segments(
        segments,
        config,
        output_chinese=output_chinese,
        progress=progress,
        initial_translations=saved,
        on_success=save_translation,
    )
    render = build_lrc if subtitle_format == "lrc" else build_srt
    write_text_atomic(destination, render(segments, result.translations))
    return destination, result
