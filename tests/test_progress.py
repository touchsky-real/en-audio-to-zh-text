import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import audio2text as app
from test_audio2text import SEGMENTS, config, fake_whisper, notebook_session, response


def translate_partially(session, text, configuration):
    if text == "Goodbye.":
        raise requests.Timeout()
    return "你好"


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "podcast.mp3"
        self.source.write_bytes(b"audio")
        self.cache = self.source.with_name(self.source.name + ".translations.json")
        self.transcript = self.source.with_name(self.source.name + ".transcript.json")

    def export(self, subtitle_format="lrc", **kwargs):
        return app.export_subtitles(self.source, SEGMENTS, subtitle_format, config(max_retries=0), **kwargs)

    def test_rerun_only_requests_missing_segments_and_reuses_across_formats(self):
        with patch.object(app, "_request_translation", side_effect=translate_partially):
            _, first = self.export()
        self.assertEqual(first.translations, {0: "你好"})
        self.assertEqual(first.failed_indices, [2])
        with patch.object(app, "_request_translation", return_value="再见") as request:
            _, second = self.export()
        request.assert_called_once()
        self.assertEqual(request.call_args.args[1], "Goodbye.")
        self.assertEqual(second.translations, {0: "你好", 2: "再见"})
        self.assertEqual(second.reused, 1)
        self.assertEqual(second.total, 2)
        self.assertEqual(second.failed_indices, [])
        with patch.object(app.requests, "Session") as factory:
            destination, third = self.export("srt")
        factory.assert_not_called()
        self.assertEqual(third.reused, 2)
        self.assertIn("你好", destination.read_text(encoding="utf-8"))
        self.assertNotIn("test-key", self.cache.read_text(encoding="utf-8"))

    def test_subsequent_failures_do_not_lose_previous_successes(self):
        with patch.object(app, "_request_translation", side_effect=translate_partially):
            self.export()
        with patch.object(app, "_request_translation", side_effect=requests.Timeout()) as request:
            path, result = self.export()
        request.assert_called_once()
        self.assertEqual(result.translations, {0: "你好"})
        self.assertEqual(result.reused, 1)
        self.assertIn("你好", path.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(self.cache.read_text(encoding="utf-8"))["translations"], {"0": "你好"})

    def test_permanent_error_preserves_already_completed_requests(self):
        with patch.object(app, "_request_translation", side_effect=["你好", app.TranslationAPIError("401")]):
            with self.assertRaises(app.TranslationAPIError):
                self.export()
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertEqual(saved["translations"], {"0": "你好"})
        with patch.object(app, "_request_translation", return_value="再见") as request:
            _, result = self.export()
        request.assert_called_once()
        self.assertEqual(result.reused, 1)

    def test_interruption_preserves_successes_from_inflight_requests(self):
        entered = threading.Event()
        release = threading.Event()

        def translate(session, text, configuration):
            if text == "Goodbye.":
                entered.set()
                self.assertTrue(release.wait(5))
                return "再见"
            self.assertTrue(entered.wait(5))
            return "你好"

        def interrupt(finished, total):
            release.set()
            raise KeyboardInterrupt()

        with patch.object(app, "_request_translation", side_effect=translate):
            with self.assertRaises(KeyboardInterrupt):
                app.export_subtitles(self.source, SEGMENTS, config=config(max_workers=2), progress=interrupt)
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertEqual(saved["translations"], {"0": "你好", "2": "再见"})

    def test_old_lrc_and_srt_exports_are_migrated(self):
        for subtitle_format, render in [("lrc", app.build_lrc), ("srt", app.build_srt)]:
            with self.subTest(format=subtitle_format):
                self.cache.unlink(missing_ok=True)
                self.source.with_suffix("." + subtitle_format).write_text(
                    render(SEGMENTS, {0: "你好"}),
                    encoding="utf-8",
                )
                with patch.object(app, "_request_translation", return_value="再见") as request:
                    _, result = self.export(subtitle_format)
                request.assert_called_once()
                self.assertEqual(request.call_args.args[1], "Goodbye.")
                self.assertEqual(result.reused, 1)
                self.assertEqual(result.translations, {0: "你好", 2: "再见"})

    def test_mismatched_old_subtitles_are_not_imported(self):
        for subtitle_format, render in [("lrc", app.build_lrc), ("srt", app.build_srt)]:
            with self.subTest(format=subtitle_format):
                self.cache.unlink(missing_ok=True)
                changed = [dict(s) for s in SEGMENTS]
                changed[0]["text"] = "Different episode."
                self.source.with_suffix("." + subtitle_format).write_text(
                    render(changed, {0: "另一集"}),
                    encoding="utf-8",
                )
                with patch.object(app, "_request_translation", return_value="新翻译") as request:
                    _, result = self.export(subtitle_format)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(result.reused, 0)

    def test_cache_is_invalidated_when_audio_or_transcript_changes(self):
        with patch.object(app, "_request_translation", return_value="你好"):
            self.export()
        self.source.write_bytes(b"different audio")
        with patch.object(app, "_request_translation", return_value="新翻译") as request:
            _, result = self.export()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result.reused, 0)
        changed = [dict(s) for s in SEGMENTS]
        changed[0]["start"] = 59
        with patch.object(app, "_request_translation", return_value="新翻译") as request:
            _, result = app.export_subtitles(self.source, changed, config=config())
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result.reused, 0)

    def test_corrupt_cache_is_not_overwritten(self):
        for invalid in ["{", "[]", '{"version": 99}']:
            with self.subTest(invalid=invalid):
                self.cache.write_text(invalid, encoding="utf-8")
                with patch.object(app.requests, "Session") as factory:
                    with self.assertRaises(ValueError):
                        self.export()
                factory.assert_not_called()
                self.assertEqual(self.cache.read_text(encoding="utf-8"), invalid)

    def test_invalid_cached_entries_are_rejected(self):
        with patch.object(app, "_request_translation", return_value="你好"):
            self.export()
        data = json.loads(self.cache.read_text(encoding="utf-8"))
        for translations in [
            {"999": "bad"},
            {"-1": "bad"},
            {"01": "bad"},
            {"1": "blank cue"},
            {"0": None},
            {"0": " "},
        ]:
            with self.subTest(translations=translations):
                data["translations"] = translations
                self.cache.write_text(json.dumps(data), encoding="utf-8")
                with patch.object(app.requests, "Session") as factory:
                    with self.assertRaises(ValueError):
                        self.export()
                factory.assert_not_called()

    def test_disabled_saving_never_reads_or_writes_cache(self):
        self.cache.write_text("invalid cache", encoding="utf-8")
        self.source.with_suffix(".lrc").write_text(app.build_lrc(SEGMENTS, {0: "旧翻译"}), encoding="utf-8")
        with patch.object(app, "_request_translation", return_value="新翻译") as request:
            _, result = self.export(save_progress=False)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result.reused, 0)
        self.assertEqual(self.cache.read_text(), "invalid cache")
        self.cache.unlink()
        with patch.object(app, "_request_translation", return_value="新翻译"):
            self.export(save_progress=False)
        self.assertFalse(self.cache.exists())

    def test_english_only_does_not_read_or_clear_translation_progress(self):
        self.cache.write_text("existing progress", encoding="utf-8")
        with patch.object(app.requests, "Session") as factory:
            _, result = self.export(output_chinese=False)
        factory.assert_not_called()
        self.assertEqual(result.translations, {})
        self.assertEqual(self.cache.read_text(), "existing progress")

    def test_cache_write_failure_stops_before_paid_requests(self):
        with patch.object(app, "write_text_atomic", side_effect=OSError("Drive unavailable")):
            with patch.object(app.requests, "Session") as factory:
                with self.assertRaises(OSError):
                    self.export()
        factory.assert_not_called()

    def test_transcript_is_restored_and_invalidated_on_source_or_model_change(self):
        with patch.object(app.subprocess, "run", side_effect=fake_whisper) as run:
            first = app.transcribe_audio(self.source)
            second = app.transcribe_audio(self.source)
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)
            app.transcribe_audio(self.source, model="tiny")
            self.assertEqual(run.call_count, 2)
            self.source.write_bytes(b"changed audio")
            app.transcribe_audio(self.source, model="tiny")
            self.assertEqual(run.call_count, 3)

    def test_disabling_progress_does_not_read_or_write_transcript(self):
        with patch.object(app.subprocess, "run", side_effect=fake_whisper):
            app.transcribe_audio(self.source, save_progress=False)
        self.assertFalse(self.transcript.exists())
        self.transcript.write_text("previous progress", encoding="utf-8")
        with patch.object(app.subprocess, "run", side_effect=fake_whisper) as run:
            app.transcribe_audio(self.source, save_progress=False)
        run.assert_called_once()
        self.assertEqual(self.transcript.read_text(), "previous progress")

    def test_notebook_resumes_after_restart_without_retranscribing(self):
        for subtitle_format in ["lrc", "srt"]:
            with self.subTest(format=subtitle_format):
                self.cache.unlink(missing_ok=True)
                self.transcript.unlink(missing_ok=True)
                self.source.with_suffix("." + subtitle_format).unlink(missing_ok=True)
                for attempt in [0, 1]:
                    with notebook_session(subtitle_format, self.source) as (cells, module, _):
                        module.OUTPUT_FORMAT = subtitle_format
                        session = Mock()
                        session.post.side_effect = (
                            [response("你好"), requests.Timeout()] if attempt == 0 else None
                        )
                        session.post.return_value = response("再见")
                        with (
                            patch.object(subprocess, "run", side_effect=fake_whisper) as run,
                            patch.object(requests, "Session", return_value=session),
                        ):
                            exec(cells["run"], module.__dict__)
                        self.assertEqual(run.call_count, 1 if attempt == 0 else 0)
                        self.assertEqual(session.post.call_count, 2 if attempt == 0 else 1)
                        self.assertEqual(module.result.reused, attempt)


if __name__ == "__main__":
    unittest.main()
