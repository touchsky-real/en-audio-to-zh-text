import json
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import audio2text as app
from test_audio2text import FakeClock, SEGMENTS, config, fake_whisper, notebook_session, response


def segments(count):
    return [{"start": i, "end": i + 1, "text": f"Sentence {i}."} for i in range(count)]


def batch_response(indices):
    # Return entries out of order to exercise ID-based alignment.
    return response(json.dumps([{"id": i, "text": f"译文{i}"} for i in reversed(list(indices))]))


class TranslationBatchTests(unittest.TestCase):
    def test_default_batching_reduces_300_requests_to_20(self):
        settings = app.TranslationConfig("test-key", "https://example.com/v1", "test-model",
                                         max_workers=1, requests_per_minute=0)
        session = Mock()
        sent = []

        def post(*args, **kwargs):
            items = json.loads(kwargs["json"]["messages"][-1]["content"])
            sent.append(items)
            return batch_response(item["id"] for item in items)

        session.post.side_effect = post
        progress = Mock()
        with patch.object(app.requests, "Session", return_value=session):
            result = app.translate_segments(segments(300), settings, progress=progress)
        self.assertEqual(session.post.call_count, 20)
        self.assertTrue(all(len(items) == 15 for items in sent))
        self.assertEqual(result.translations, {i: f"译文{i}" for i in range(300)})
        self.assertEqual(result.failed_indices, [])
        progress.assert_called_with(300, 300)
        session.close.assert_called_once()
        subtitle = app.build_srt(segments(300), result.translations)
        self.assertTrue(subtitle.startswith("1\n00:00:00,000 --> 00:00:01,000\nSentence 0.\n译文0\n"))
        self.assertIn("300\n00:04:59,000 --> 00:05:00,000\nSentence 299.\n译文299\n", subtitle)

    def test_character_boundary_oversized_cue_and_tail(self):
        texts = {0: "aaaaa", 2: "bbbbb", 3: "c", 4: "d" * 11, 5: "ee", 6: "ff", 7: "gg", 8: "h"}
        chunks = list(app._translation_chunks(texts, config(batch_size=3, batch_max_chars=10)))
        self.assertEqual([list(chunk) for chunk in chunks], [[0, 2], [3], [4], [5, 6, 7], [8]])
        self.assertEqual({i: text for chunk in chunks for i, text in chunk.items()}, texts)

    def test_skips_blanks_and_cached_cues_and_keeps_original_ids(self):
        source = SEGMENTS + [{"start": 65, "end": 66, "text": "Already done."}]
        session = Mock()
        session.post.return_value = batch_response([0, 2])
        saved, progress = Mock(), Mock()
        with patch.object(app.requests, "Session", return_value=session):
            result = app.translate_segments(source, config(batch_size=15), initial_translations={3: "已完成"},
                                            on_success=saved, progress=progress)
        sent = json.loads(session.post.call_args.kwargs["json"]["messages"][-1]["content"])
        self.assertEqual(sent, [{"id": 0, "text": "Hello world."}, {"id": 2, "text": "Goodbye."}])
        self.assertEqual(result.translations, {0: "译文0", 2: "译文2", 3: "已完成"})
        self.assertEqual((result.reused, result.total), (1, 3))
        self.assertEqual(saved.call_count, 2)
        progress.assert_called_with(2, 2)

    def test_malformed_batches_are_retried_without_saving_wrong_alignment(self):
        valid = [{"id": 0, "text": "你好"}, {"id": 2, "text": "再见"}]
        invalid = [
            "not JSON", "{}", "[]", '["你好", "再见"]', json.dumps(valid[:1]),
            json.dumps(valid + [valid[0]]), json.dumps(valid + [{"id": 99, "text": "extra"}]),
        ]
        for index, text in [(True, "x"), (0.0, "x"), ("0", "x"), (0, " "), (0, None), (0, ["x"])]:
            invalid.append(json.dumps([{"id": index, "text": text}, valid[1]]))
        for content in invalid:
            with self.subTest(content=content):
                session = Mock()
                session.post.side_effect = [response(content), response("```json\n" + json.dumps(valid) + "\n```")]
                saved = Mock()
                with patch.object(app.requests, "Session", return_value=session):
                    result = app.translate_segments(SEGMENTS, config(batch_size=15), on_success=saved)
                self.assertEqual(session.post.call_count, 2)
                self.assertEqual(result.translations, {0: "你好", 2: "再见"})
                self.assertEqual(saved.call_count, 2)
                self.assertEqual(result.failures, {})

    def test_exhausted_batch_reports_each_cue_without_single_request_fallback(self):
        session = Mock()
        session.post.return_value = response("[]")
        failed, progress = Mock(), Mock()
        with patch.object(app.requests, "Session", return_value=session):
            result = app.translate_segments(SEGMENTS, config(batch_size=15, max_retries=1),
                                            on_failure=failed, progress=progress)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(result.failed_indices, [0, 2])
        self.assertEqual(result.translations, {})
        for index in (0, 2):
            self.assertEqual(result.failures[index].attempts, 2)
            failed.assert_any_call(index, result.failures[index])
        progress.assert_called_with(2, 2)

    def test_concurrent_chunks_complete_out_of_order(self):
        second_finished = threading.Event()
        completed = []

        def post(*args, **kwargs):
            ids = [item["id"] for item in json.loads(kwargs["json"]["messages"][-1]["content"])]
            if ids == [0, 1]:
                self.assertTrue(second_finished.wait(5))
            return batch_response(ids)

        def saved(index, text):
            completed.append(index)
            if index == 3:
                second_finished.set()

        def session_factory():
            session = Mock()
            session.post.side_effect = post
            return session

        with patch.object(app.requests, "Session", side_effect=session_factory):
            result = app.translate_segments(segments(4), config(batch_size=2, max_workers=2), on_success=saved)
        self.assertEqual(completed, [2, 3, 0, 1])
        self.assertEqual(result.translations, {i: f"译文{i}" for i in range(4)})

    def test_resume_after_permanent_error_reuses_completed_batch_across_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "episode.mp3"
            source.touch()
            session = Mock()
            session.post.side_effect = [batch_response([0, 1]), response(status=401)]
            with patch.object(app.requests, "Session", return_value=session):
                with self.assertRaises(app.TranslationAPIError):
                    app.export_subtitles(source, segments(4), config=config(batch_size=2))
            cache = json.loads(source.with_name(source.name + ".translations.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["translations"], {"0": "译文0", "1": "译文1"})
            session.post.side_effect = None
            session.post.return_value = batch_response([2, 3])
            session.post.reset_mock()
            with patch.object(app.requests, "Session", return_value=session):
                path, result = app.export_subtitles(source, segments(4), "srt", config(batch_size=15))
            session.post.assert_called_once()
            sent = json.loads(session.post.call_args.kwargs["json"]["messages"][-1]["content"])
            self.assertEqual([item["id"] for item in sent], [2, 3])
            self.assertEqual(result.reused, 2)
            self.assertIn("译文3", path.read_text(encoding="utf-8"))

    def test_both_notebooks_send_batches(self):
        for entry in ("lrc", "srt"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "episode.mp3"
                source.touch()
                with notebook_session(entry, source) as (cells, module, output):
                    module.BATCH_SIZE = 15
                    session = Mock()
                    session.post.return_value = batch_response([0, 2])
                    with (
                        patch.object(subprocess, "Popen", side_effect=fake_whisper),
                        patch.object(requests, "Session", return_value=session),
                    ):
                        exec(cells["transcribe"], module.__dict__)
                        exec(cells["translate"], module.__dict__)
                    session.post.assert_called_once()
                    self.assertEqual(module.result.translations, {0: "译文0", 2: "译文2"})
                    self.assertIn("本轮已处理：2/2（成功 2，失败 0）", output.getvalue())


class RequestPacingTests(unittest.TestCase):
    def test_exponential_delays_jitter_cap_and_retry_after_floor(self):
        settings = config(retry_delay=1)
        with patch.object(app.random, "uniform", side_effect=lambda low, high: high):
            delays = [app._backoff_delay(i, settings, requests.Timeout()) for i in range(8)]
            self.assertEqual(delays, [1.25, 2.5, 5, 10, 20, 40, 60, 60])
            self.assertEqual(app._backoff_delay(0, settings, app._RetryableError("busy", delay=120)), 120)
        with patch.object(app, "datetime") as date:
            date.now.return_value = datetime(2026, 10, 7, tzinfo=timezone.utc)
            self.assertEqual(app._retry_after("Wed, 07 Oct 2026 00:02:00 GMT"), 120)

    def test_rate_limit_applies_to_retries_and_new_chunks(self):
        clock = FakeClock()
        sent_at = []
        outcomes = iter([response(status=500), response(), response()])

        def post(*args, **kwargs):
            sent_at.append(clock.now)
            return next(outcomes)

        session = Mock()
        session.post.side_effect = post
        with clock.installed(), patch.object(app.requests, "Session", return_value=session):
            result = app.translate_segments(SEGMENTS, config(requests_per_minute=30))
        self.assertEqual(sent_at, [100, 102, 104])
        self.assertEqual(result.failed_indices, [])

    def test_exhausted_429_or_retry_after_slows_down_remaining_queue(self):
        for status, headers, earliest in [(429, {}, 102), (429, {"Retry-After": "120"}, 220),
                                         (503, {"Retry-After": "120"}, 220)]:
            with self.subTest(status=status, headers=headers):
                clock = FakeClock()
                sent_at = []
                outcomes = iter([response(status=status, headers=headers), response()])

                def post(*args, **kwargs):
                    sent_at.append(clock.now)
                    return next(outcomes)

                session = Mock()
                session.post.side_effect = post
                with (
                    clock.installed(), patch.object(app.requests, "Session", return_value=session),
                    patch.object(app.random, "uniform", return_value=0),
                ):
                    result = app.translate_segments(SEGMENTS, config(max_retries=0, retry_delay=1))
                self.assertEqual(sent_at, [100, earliest])
                self.assertEqual(result.failed_indices, [0])
                self.assertEqual(result.translations, {2: "你好"})

    def test_shared_pacer_spaces_concurrent_workers(self):
        clock = FakeClock()
        pacer = app._RequestPacer(config(requests_per_minute=30))
        stop = threading.Event()
        entered = threading.Barrier(4)

        def acquire():
            entered.wait()
            return pacer.acquire(stop)

        # The barrier uses Condition.wait, so FakeClock affects only pacing waits.
        with clock.installed(), ThreadPoolExecutor(max_workers=3) as executor:
            jobs = [executor.submit(acquire) for _ in range(3)]
            entered.wait(timeout=5)
            self.assertEqual([job.result(timeout=5) for job in jobs], [0, 0, 0])
        self.assertGreaterEqual(clock.now, 104)

    def test_repeated_429_extends_cooldown_and_stale_success_cannot_undo_it(self):
        clock = FakeClock()
        stop = threading.Event()
        pacer = app._RequestPacer(config(requests_per_minute=30, retry_delay=1))
        with clock.installed():
            stale = pacer.acquire(stop)
            error = app._RetryableError("limited", status_code=429)
            pacer.defer(error, 1)
            pacer.defer(error, 1)
            self.assertEqual(pacer.interval, 8)
            pacer.succeeded(stale)
            self.assertEqual(pacer.interval, 8)
            current = pacer.acquire(stop)
            self.assertEqual(clock.now, 108)
            pacer.succeeded(current)
            self.assertAlmostEqual(pacer.interval, 7.2)
            for _ in range(30):
                pacer.succeeded(current)
            self.assertEqual(pacer.interval, 2)

    def test_stop_interrupts_long_cooldown(self):
        pacer = app._RequestPacer(config())
        pacer.defer(app._RetryableError("limited", status_code=429, delay=600), 600)
        stop = threading.Event()
        waiting = threading.Event()
        original_wait = stop.wait

        def wait(timeout):
            waiting.set()
            return original_wait(timeout)

        with patch.object(stop, "wait", side_effect=wait), ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(pacer.acquire, stop)
            try:
                self.assertTrue(waiting.wait(5))
            finally:
                stop.set()
            self.assertIsNone(job.result(timeout=5))


if __name__ == "__main__":
    unittest.main()
