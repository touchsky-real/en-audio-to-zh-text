import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import nbformat

import audio2text as app


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = [
    {"start": 59.9996, "end": 62.25, "text": " Hello\nworld. "},
    {"start": 62.25, "end": 63, "text": " "},
    {"start": 63, "end": 65, "text": "Goodbye."},
]


def config(**overrides):
    values = dict(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model="test-model",
        max_workers=1,
        max_retries=2,
        retry_delay=0,
        batch_size=1,
        requests_per_minute=0,
    )
    values.update(overrides)
    return app.TranslationConfig(**values)


def response(content="你好", *, status=200, payload=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = (
        payload
        if payload is not None
        else {
            "choices": [{"message": {"content": content}}],
        }
    )
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class FakeClock:
    """Advance timed waits without delaying tests or changing thread startup."""

    def __init__(self):
        self.now = 100.0
        self.waits = []
        self.original_wait = threading.Event.wait

    def wait(self, event, timeout=None):
        if timeout is None:
            return self.original_wait(event)
        self.waits.append(timeout)
        self.now += timeout
        return event.is_set()

    @contextlib.contextmanager
    def installed(self):
        with (
            patch.object(app.time, "monotonic", side_effect=lambda: self.now),
            patch.object(threading.Event, "wait", autospec=True, side_effect=self.wait),
        ):
            yield self


def fake_process(output=b"", return_code=0):
    process = Mock()
    process.stdout = io.BytesIO(output)
    process.wait.return_value = return_code
    process.poll.return_value = return_code
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    return process


def fake_whisper(command, **kwargs):
    if kwargs.get("stdout") != subprocess.PIPE or kwargs.get("stderr") != subprocess.STDOUT:
        raise AssertionError("Whisper progress must be captured for streaming")
    workdir = Path(command[command.index("--output_dir") + 1])
    (workdir / "audio.json").write_text(json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    return fake_process(b"\r100%|##########| 100/100 [00:01<00:00, 100frames/s]\n")


class SubtitleTests(unittest.TestCase):
    def test_lrc_rounds_across_minute_and_hour(self):
        for seconds, expected in [
            (0, "[00:00.00]"),
            (59.999, "[01:00.00]"),
            (3599.999, "[60:00.00]"),
            (3600.12, "[60:00.12]"),
        ]:
            with self.subTest(seconds=seconds):
                self.assertEqual(app.seconds_to_lrc(seconds), expected)

    def test_srt_rounds_across_minute_and_hour(self):
        self.assertEqual(app.seconds_to_srt(59.9996), "00:01:00,000")
        self.assertEqual(app.seconds_to_srt(3599.9996), "01:00:00,000")
        self.assertEqual(app.seconds_to_srt(3661.123), "01:01:01,123")

    def test_invalid_timestamps(self):
        for value in [-1, float("nan"), float("inf")]:
            for formatter in [app.seconds_to_lrc, app.seconds_to_srt]:
                with self.subTest(value=value, formatter=formatter.__name__):
                    with self.assertRaises(ValueError):
                        formatter(value)

    def test_srt_contiguous_indices_and_multiline_translation(self):
        content = app.build_srt(SEGMENTS, {0: "你好\n\n世界"})
        self.assertEqual(
            content,
            (
                "1\n00:01:00,000 --> 00:01:02,250\nHello world.\n你好 世界\n\n"
                "2\n00:01:03,000 --> 00:01:05,000\nGoodbye.\n\n"
            ),
        )

    def test_lrc_preserves_english_when_translation_is_missing(self):
        content = app.build_lrc(SEGMENTS, {0: "你好"})
        self.assertIn("[01:00.00]Hello world.\n[01:00.00]你好\n", content)
        self.assertIn("[01:03.00]Goodbye.\n", content)
        self.assertNotIn("[01:02.25]", content)

    def test_segment_validation(self):
        for data in [
            {},
            {"segments": None},
            {"segments": [None]},
            {"segments": [{"text": "x", "start": 0, "end": -1}]},
            {"segments": [{"text": "x", "start": True, "end": 1}]},
            {"segments": [{"text": "x", "start": 0, "end": float("inf")}]},
            {"segments": [{"text": None, "start": 0, "end": 1}]},
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                app.validate_segments(data)
        self.assertEqual(app.validate_segments({"segments": []}), [])
        self.assertEqual(app.validate_segments({"segments": SEGMENTS})[0]["text"], "Hello world.")


class FileTests(unittest.TestCase):
    def test_original_filename_wins_over_sanitized_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "episode (1).mp3"
            collision = Path(folder) / "episode_1.mp3"
            source.write_bytes(b"original")
            collision.write_bytes(b"other")
            self.assertEqual(app.resolve_audio_path(folder, source.name), source.resolve())
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(collision.read_bytes(), b"other")
            with self.assertRaises(FileNotFoundError):
                app.resolve_audio_path(folder, "missing.mp3")

    def test_filename_cannot_select_another_directory(self):
        for filename in ["", ".", "..", "../a.mp3", "sub/a.mp3", "sub\\a.mp3"]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                app.resolve_audio_path(".", filename)

    def test_transcription_uses_isolated_copy_and_argument_list(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "中文 $(example) & podcast.mp3"
            source.write_bytes(b"audio")
            stale = Path(folder) / "audio.json"
            stale.write_text("stale", encoding="utf-8")
            with patch.object(app.subprocess, "Popen", side_effect=fake_whisper) as run:
                result = app.transcribe_audio(source)
            self.assertEqual(result, app.validate_segments({"segments": SEGMENTS}))
            command = run.call_args.args[0]
            local_audio = Path(command[command.index("whisper") + 1])
            self.assertIsInstance(command, list)
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertNotEqual(local_audio.parent, source.parent)
            self.assertFalse(local_audio.exists())
            self.assertEqual(source.read_bytes(), b"audio")
            self.assertEqual(stale.read_text(), "stale")

    def test_failed_transcription_does_not_read_existing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "podcast.mp3"
            source.write_bytes(b"audio")
            source.with_suffix(".json").write_text(json.dumps({"segments": SEGMENTS}))
            with patch.object(
                app.subprocess, "Popen", side_effect=subprocess.CalledProcessError(1, "whisper")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    app.transcribe_audio(source)

    def test_success_exit_without_json_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "podcast.mp3"
            source.touch()
            with patch.object(app.subprocess, "Popen", return_value=fake_process()):
                with self.assertRaises(FileNotFoundError):
                    app.transcribe_audio(source)

    def test_atomic_write_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "podcast.lrc"
            destination.write_text("old", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("unavailable")):
                with self.assertRaises(OSError):
                    app.write_text_atomic(destination, "new")
            self.assertEqual(destination.read_text(), "old")
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_export_english_srt_and_preserve_original_filename(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app.requests, "Session") as session:
            source = Path(folder) / "a podcast.mp3"
            source.write_bytes(b"audio")
            destination, result = app.export_subtitles(source, SEGMENTS, "srt", output_chinese=False)
            self.assertEqual(destination.name, "a podcast.srt")
            self.assertEqual(destination.read_bytes(), (Path(folder) / "a podcast_en.srt").read_bytes())
            self.assertNotIn(b"\r", destination.read_bytes())
            self.assertEqual(source.read_bytes(), b"audio")
            self.assertEqual(result.failed_indices, [])
            session.assert_not_called()

    def test_partial_translation_saved_with_failure_report(self):
        partial = app.TranslationResult({0: "你好"}, [2], 2)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "podcast.mp3"
            source.touch()
            with patch.object(app, "translate_segments", return_value=partial):
                destination, result = app.export_subtitles(source, SEGMENTS)
            self.assertIn("Goodbye.", destination.read_text(encoding="utf-8"))
            self.assertEqual(result.failed_indices, [2])

    def test_permanent_api_error_keeps_english_srt_and_old_final(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "podcast.mp3"
            source.touch()
            destination = source.with_suffix(".srt")
            destination.write_text("old", encoding="utf-8")
            with patch.object(app, "translate_segments", side_effect=app.TranslationAPIError("HTTP 401")):
                with self.assertRaises(app.TranslationAPIError):
                    app.export_subtitles(source, SEGMENTS, "srt")
            self.assertEqual(destination.read_text(), "old")
            self.assertIn("Hello world.", (Path(folder) / "podcast_en.srt").read_text())

    def test_invalid_export_does_not_touch_files_or_api(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app.requests, "Session") as session:
            source = Path(folder) / "podcast.lrc"
            source.write_text("original")
            with self.assertRaises(ValueError):
                app.export_subtitles(source, SEGMENTS)
            with self.assertRaises(ValueError):
                app.export_subtitles(source, SEGMENTS, "bad")
            self.assertEqual(source.read_text(), "original")
            session.assert_not_called()


class TranslationTests(unittest.TestCase):
    def test_config_validation_and_key_not_in_repr(self):
        for overrides in [
            {"api_key": ""},
            {"api_key": "***"},
            {"model": " "},
            {"base_url": "missing-scheme"},
            {"base_url": "ftp://example.com"},
            {"base_url": "https://example.com/?key=secret"},
            {"base_url": "https://user:secret@example.com"},
            {"max_workers": 0},
            {"max_workers": True},
            {"max_retries": -1},
            {"max_retries": 1.5},
            {"timeout": 0},
            {"timeout": float("nan")},
            {"retry_delay": -1},
            {"batch_size": 0},
            {"batch_size": True},
            {"batch_size": 1.5},
            {"batch_max_chars": 0},
            {"batch_max_chars": True},
            {"requests_per_minute": -1},
            {"requests_per_minute": float("nan")},
            {"requests_per_minute": float("inf")},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides)
        self.assertNotIn("test-key", repr(config()))

    def test_english_only_and_empty_segments_skip_all_api_work(self):
        with patch.object(app.requests, "Session") as session:
            result = app.translate_segments(SEGMENTS, output_chinese=False)
            self.assertEqual(result.failed_indices, [])
            self.assertEqual(app.translate_segments([]).total, 0)
            self.assertEqual(app.translate_segments([SEGMENTS[1]]).total, 0)
            session.assert_not_called()

    def test_chinese_requires_config(self):
        with self.assertRaises(ValueError):
            app.translate_segments(SEGMENTS)

    def test_retries_and_session_reuse(self):
        session = Mock()
        session.post.side_effect = [requests.Timeout(), response(status=429), response(), response("再见")]
        progress = Mock()
        with patch.object(app.requests, "Session", return_value=session) as factory:
            result = app.translate_segments(SEGMENTS, config(), progress=progress)
        self.assertEqual(result.translations, {0: "你好", 2: "再见"})
        self.assertEqual(result.failed_indices, [])
        self.assertEqual(session.post.call_count, 4)
        factory.assert_called_once()
        session.close.assert_called_once()
        self.assertEqual(session.post.call_args.args[0], "https://example.com/v1/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["timeout"], 30)
        progress.assert_any_call(2, 2)

    def test_retry_limit_is_extra_attempts(self):
        for retries in [0, 1, 2]:
            with self.subTest(retries=retries):
                session = Mock()
                session.post.side_effect = requests.ConnectionError()
                with patch.object(app.requests, "Session", return_value=session):
                    result = app.translate_segments(SEGMENTS[:1], config(max_retries=retries))
                self.assertEqual(session.post.call_count, retries + 1)
                self.assertEqual(result.failed_indices, [0])
                session.close.assert_called_once()

    def test_permanent_errors_stop_queue_without_retries(self):
        for status in [301, 400, 401, 403, 404, 422]:
            with self.subTest(status=status):
                session = Mock()
                session.post.return_value = response(status=status)
                with patch.object(app.requests, "Session", return_value=session):
                    with self.assertRaisesRegex(app.TranslationAPIError, f"HTTP {status}"):
                        app.translate_segments(SEGMENTS * 10, config())
                session.post.assert_called_once()
                session.close.assert_called_once()

    def test_empty_and_malformed_responses_are_retried(self):
        for invalid in [
            response(content=" "),
            response(content=None),
            response(payload={}),
            response(payload={"choices": []}),
            response(payload=[]),
            response(payload={"choices": [{"message": {"content": ["bad"]}}]}),
        ]:
            session = Mock()
            session.post.side_effect = [invalid, response()]
            with patch.object(app.requests, "Session", return_value=session):
                result = app.translate_segments(SEGMENTS[:1], config())
            self.assertEqual(result.translations, {0: "你好"})
            self.assertEqual(session.post.call_count, 2)

    def test_server_error_honors_retry_after(self):
        session = Mock()
        session.post.side_effect = [response(status=503, headers={"Retry-After": "2"}), response()]
        clock = FakeClock()
        with patch.object(app.requests, "Session", return_value=session), clock.installed():
            app.translate_segments(SEGMENTS[:1], config())
        self.assertIn(2, clock.waits)

    def test_retry_after_honors_long_delays_and_handles_invalid_values(self):
        for value, expected in [
            ("3", 3),
            ("999", 999),
            ("-4", 0),
            ("invalid", 0),
            ("nan", 0),
            ("inf", 0),
            (None, 0),
            ("Wed, 21 Oct 2015 07:28:00 GMT", 0),
        ]:
            with self.subTest(value=value):
                self.assertEqual(app._retry_after(value), expected)

    def test_concurrent_completion_keeps_segment_order(self):
        second_finished = threading.Event()

        def translate(session, text, configuration):
            if text.startswith("Hello"):
                self.assertTrue(second_finished.wait(5))
                return "你好"
            second_finished.set()
            return "再见"

        with patch.object(app, "_request_translation", side_effect=translate):
            with patch.object(app.requests, "Session", side_effect=lambda: Mock()):
                result = app.translate_segments(SEGMENTS, config(max_workers=2))
        content = app.build_srt(SEGMENTS, result.translations)
        self.assertLess(content.index("你好"), content.index("再见"))


@contextlib.contextmanager
def notebook_session(subtitle_format, audio_path=None, *, configure=True, prepare=True):
    notebook = json.loads((ROOT / f"audio2{subtitle_format}.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    module = types.ModuleType("notebook_test")
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        contextlib.redirect_stdout(io.StringIO()) as output,
    ):
        if configure:
            with patch.dict("os.environ", {"TRANSLATION_API_KEY": "test-key"}):
                exec(cells["config"], module.__dict__)
        if audio_path is not None:
            module.__dict__.update(
                DRIVE_FOLDER=str(audio_path.parent),
                AUDIO_FILENAME=audio_path.name,
                MAX_WORKERS=1,
                MAX_RETRIES=0,
                BATCH_SIZE=1,
                REQUESTS_PER_MINUTE=0,
                RETRY_DELAY=0,
            )
        if prepare:
            colab = types.ModuleType("google.colab")
            colab.drive = Mock()
            with (
                patch.dict(sys.modules, {"google.colab": colab}),
                patch.object(subprocess, "run"),
                patch("shutil.which", return_value="ffmpeg"),
            ):
                exec(cells["setup"], module.__dict__)
        yield cells, module, output


class NotebookTests(unittest.TestCase):
    def test_generated_notebooks_have_four_steps_and_valid_code(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/build_notebooks.py"), "--check"], check=True)
        for path in ROOT.glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            nbformat.validate(notebook)
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(
                [cell["id"] for cell in code_cells], ["config", "setup", "transcribe", "translate"]
            )
            for cell in code_cells:
                compile("".join(cell["source"]), str(path) + ":" + cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_run_without_configuration_explains_how_to_recover(self):
        for subtitle_format in ["lrc", "srt"]:
            with (
                self.subTest(format=subtitle_format),
                notebook_session(subtitle_format, configure=False) as (cells, module, _),
            ):
                with patch.object(subprocess, "Popen") as run, patch.object(requests, "Session") as session:
                    for step in ["transcribe", "translate"]:
                        with self.assertRaisesRegex(RuntimeError, "请先运行 Step 1"):
                            exec(cells[step], module.__dict__)
                    run.assert_not_called()
                    session.assert_not_called()

    def test_steps_without_environment_explain_how_to_recover(self):
        for entry in ["lrc", "srt"]:
            with self.subTest(entry=entry), notebook_session(entry, prepare=False) as (cells, module, _):
                for step in ["transcribe", "translate"]:
                    with self.assertRaisesRegex(RuntimeError, "Step 2"):
                        exec(cells[step], module.__dict__)

    def test_translation_without_transcript_never_runs_whisper(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "podcast.mp3"
            audio.touch()
            for entry in ["lrc", "srt"]:
                with self.subTest(entry=entry), notebook_session(entry, audio) as (cells, module, _):
                    with (
                        patch.object(subprocess, "Popen") as run,
                        patch.object(requests, "Session") as session,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "Step 3"):
                            exec(cells["translate"], module.__dict__)
                    run.assert_not_called()
                    session.assert_not_called()
                    self.assertFalse(audio.with_suffix(".lrc").exists())

    def test_disabled_saving_uses_memory_only_in_current_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "podcast.mp3"
            audio.touch()
            with notebook_session("lrc", audio) as (cells, module, _):
                module.SAVE_PROGRESS = False
                module.OUTPUT_CHINESE = False
                with patch.object(subprocess, "Popen", side_effect=fake_whisper):
                    exec(cells["transcribe"], module.__dict__)
                with patch.object(subprocess, "Popen") as run:
                    exec(cells["translate"], module.__dict__)
                    run.assert_not_called()
                self.assertTrue(audio.with_suffix(".lrc").is_file())
                self.assertFalse(list(audio.parent.glob("*.json")))
                module.WHISPER_MODEL = "tiny"
                with self.assertRaisesRegex(RuntimeError, "Step 3"):
                    exec(cells["translate"], module.__dict__)
            with notebook_session("lrc", audio) as (cells, module, _):
                module.SAVE_PROGRESS = False
                with patch.object(subprocess, "Popen") as run:
                    with self.assertRaisesRegex(RuntimeError, "Step 3"):
                        exec(cells["translate"], module.__dict__)
                    run.assert_not_called()

    def test_changed_model_does_not_load_old_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "podcast.mp3"
            audio.touch()
            with notebook_session("lrc", audio) as (cells, module, _):
                with patch.object(subprocess, "Popen", side_effect=fake_whisper):
                    exec(cells["transcribe"], module.__dict__)
                module.WHISPER_MODEL = "tiny"
                with patch.object(subprocess, "Popen") as run, patch.object(requests, "Session") as session:
                    with self.assertRaisesRegex(RuntimeError, "Step 3"):
                        exec(cells["translate"], module.__dict__)
                    run.assert_not_called()
                    session.assert_not_called()

    def test_run_loads_helpers_and_failed_rerun_preserves_output(self):
        for subtitle_format in ["lrc", "srt"]:
            with self.subTest(format=subtitle_format), tempfile.TemporaryDirectory() as folder:
                audio = Path(folder) / "podcast (test).mp3"
                audio.write_bytes(b"audio")
                with notebook_session(subtitle_format, audio) as (cells, module, _):
                    namespace = module.__dict__
                    self.assertIs(namespace["SAVE_PROGRESS"], True)
                    namespace["OUTPUT_FORMAT"] = subtitle_format
                    namespace["OUTPUT_CHINESE"] = False
                    with patch.object(subprocess, "Popen", side_effect=fake_whisper) as run:
                        with patch.object(requests, "Session") as session:
                            exec(cells["transcribe"], namespace)
                            exec(cells["translate"], namespace)
                            exec(cells["transcribe"], namespace)
                            exec(cells["translate"], namespace)
                            session.assert_not_called()
                        run.assert_called_once()
                    output_path = audio.with_suffix("." + subtitle_format)
                    saved = output_path.read_bytes()
                    self.assertIn(b"Hello world.", saved)
                    audio.write_bytes(b"changed audio")
                    with patch.object(
                        subprocess, "run", side_effect=subprocess.CalledProcessError(1, "whisper")
                    ):
                        with self.assertRaises(subprocess.CalledProcessError):
                            exec(cells["transcribe"], namespace)
                        self.assertIsNone(namespace["transcription"])
                        with self.assertRaisesRegex(RuntimeError, "Step 3"):
                            exec(cells["translate"], namespace)
                    self.assertEqual(output_path.read_bytes(), saved)

    def test_changed_filename_uses_new_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.mp3"
            second = Path(folder) / "second.mp3"
            first.touch()
            second.touch()
            with notebook_session("lrc", first) as (cells, module, _):
                module.OUTPUT_CHINESE = False
                with patch.object(subprocess, "Popen", side_effect=fake_whisper) as run:
                    exec(cells["transcribe"], module.__dict__)
                    exec(cells["translate"], module.__dict__)
                    module.AUDIO_FILENAME = second.name
                    with self.assertRaisesRegex(RuntimeError, "Step 3"):
                        exec(cells["translate"], module.__dict__)
                    exec(cells["transcribe"], module.__dict__)
                    exec(cells["translate"], module.__dict__)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(module.output_path, second.with_suffix(".lrc").resolve())
                self.assertTrue(first.with_suffix(".lrc").is_file())

    def test_default_lrc_and_switching_format_reuse_saved_progress(self):
        for entry in ["lrc", "srt"]:
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as folder:
                audio = Path(folder) / "podcast.mp3"
                audio.touch()
                with notebook_session(entry, audio) as (cells, module, _):
                    self.assertEqual(module.OUTPUT_FORMAT, "lrc")
                    session = Mock()
                    session.post.return_value = response()
                    with (
                        patch.object(subprocess, "Popen", side_effect=fake_whisper) as transcribe,
                        patch.object(requests, "Session", return_value=session) as sessions,
                    ):
                        exec(cells["transcribe"], module.__dict__)
                        exec(cells["translate"], module.__dict__)
                        self.assertEqual(module.output_path.suffix, ".lrc")
                        module.OUTPUT_FORMAT = "srt"
                        exec(cells["translate"], module.__dict__)
                    transcribe.assert_called_once()
                    sessions.assert_called_once()
                    self.assertEqual(session.post.call_count, 2)
                    self.assertEqual(module.result.reused, 2)
                    self.assertEqual(module.output_path.suffix, ".srt")
                    for filename in ["podcast.lrc", "podcast.srt", "podcast_en.srt"]:
                        self.assertTrue((audio.parent / filename).is_file())

    def test_invalid_format_stops_before_transcription_or_translation(self):
        for entry in ["lrc", "srt"]:
            for invalid in ["vtt", "SRT", "", None]:
                with self.subTest(entry=entry, format=invalid), notebook_session(entry) as (cells, module, _):
                    module.OUTPUT_FORMAT = invalid
                    with (
                        patch.object(subprocess, "Popen") as run,
                        patch.object(requests, "Session") as session,
                    ):
                        with self.assertRaisesRegex(ValueError, "OUTPUT_FORMAT"):
                            exec(cells["translate"], module.__dict__)
                        run.assert_not_called()
                        session.assert_not_called()

    def test_auto_disconnect_requires_successful_complete_export(self):
        for subtitle_format in ["lrc", "srt"]:
            for outcome in ["complete", "partial", "api_error", "save_error"]:
                with (
                    self.subTest(format=subtitle_format, outcome=outcome),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    audio = Path(folder) / "podcast.mp3"
                    audio.touch()
                    runtime = Mock()
                    colab = types.ModuleType("google.colab")
                    colab.runtime = runtime
                    session = Mock()
                    session.post.side_effect = (
                        [response(), requests.Timeout()] if outcome == "partial" else None
                    )
                    session.post.return_value = response(status=401 if outcome == "api_error" else 200)
                    original_replace = Path.replace
                    # Windows TEMP may use an 8.3 alias such as RUNNER~1, while
                    # the notebook resolves the source to its full path.
                    expected_output = audio.with_suffix("." + subtitle_format).resolve()

                    def replace(path, target):
                        if outcome == "save_error" and Path(target).resolve() == expected_output:
                            raise OSError("save failed")
                        return original_replace(path, target)

                    with notebook_session(subtitle_format, audio) as (cells, module, _):
                        module.AUTO_DISCONNECT = True
                        module.OUTPUT_FORMAT = subtitle_format
                        with (
                            patch.dict(sys.modules, {"google.colab": colab}),
                            patch("time.sleep"),
                            patch.object(subprocess, "Popen", side_effect=fake_whisper),
                            patch.object(requests, "Session", return_value=session),
                            patch.object(Path, "replace", replace),
                        ):
                            exec(cells["transcribe"], module.__dict__)
                            runtime.unassign.assert_not_called()
                            session.post.assert_not_called()
                            if outcome in {"api_error", "save_error"}:
                                with self.assertRaises(RuntimeError if outcome == "api_error" else OSError):
                                    exec(cells["translate"], module.__dict__)
                            else:
                                exec(cells["translate"], module.__dict__)
                        self.assertEqual(runtime.unassign.call_count, int(outcome == "complete"))


if __name__ == "__main__":
    unittest.main()
