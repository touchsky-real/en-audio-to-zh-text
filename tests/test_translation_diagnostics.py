import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import wait
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import audio2text as app
from test_audio2text import SEGMENTS, config, fake_whisper, notebook_session, response


class TranslationDiagnosticsTests(unittest.TestCase):
    def test_exhausted_requests_report_cause_attempts_and_http_status(self):
        invalid_json = response()
        invalid_json.json.side_effect = ValueError("private response test-key")
        cases = [
            (response(status=429), "请求受限", 429),
            (response(status=408), "服务端请求超时", 408),
            (response(status=500), "服务端错误", 500),
            (response(status=503), "服务端错误", 503),
            (invalid_json, "响应不是有效 JSON", 200),
            (response(payload={}), "缺少 choices[0].message.content 文本", 200),
            (response(content=None), "返回格式异常", 200),
            (response(content=" \n"), "返回翻译为空", 200),
            (requests.Timeout("private endpoint test-key"), "请求超时", None),
            (requests.ConnectionError("private endpoint test-key"), "网络连接失败", None),
            (requests.exceptions.SSLError("private endpoint test-key"), "TLS/SSL", None),
            (requests.RequestException("private endpoint test-key"), "网络请求异常", None),
        ]
        for outcome, reason, status in cases:
            with self.subTest(reason=reason, status=status):
                session = Mock()
                session.post.side_effect = [outcome] * 3
                events = []
                with patch.object(app.requests, "Session", return_value=session):
                    result = app.translate_segments(
                        SEGMENTS[:1], config(),
                        on_failure=lambda i, failure: events.append(("failure", i, failure)),
                        progress=lambda done, total: events.append(("progress", done, total)),
                    )
                self.assertEqual(result.failed_indices, [0])
                self.assertEqual(result.translations, {})
                failure = result.failures[0]
                self.assertIn(reason, failure.reason)
                self.assertEqual(failure.status_code, status)
                self.assertEqual(failure.attempts, 3)
                self.assertEqual(session.post.call_count, 3)
                self.assertEqual(events, [("failure", 0, failure), ("progress", 1, 1)])
                if status is not None:
                    self.assertIn(f"HTTP {status}", failure.description)
                self.assertNotIn("test-key", repr(failure))
                self.assertNotIn("private", repr(failure))

    def test_summary_uses_last_error_and_excludes_recovered_requests(self):
        session = Mock()
        session.post.side_effect = [response(status=429), response(status=503), requests.Timeout(), response()]
        callback = Mock()
        with patch.object(app.requests, "Session", return_value=session):
            result = app.translate_segments(SEGMENTS, config(max_retries=1), on_failure=callback)
        self.assertEqual(result.translations, {2: "你好"})
        self.assertEqual(result.failed_indices, [0])
        self.assertEqual(set(result.failures), {0})
        self.assertEqual(result.failures[0].status_code, 503)
        self.assertEqual(result.failures[0].attempts, 2)
        callback.assert_called_once_with(0, result.failures[0])

    def test_concurrent_diagnostics_keep_original_segment_indices(self):
        second_finished = threading.Event()

        def translate(session, text, configuration):
            if text.startswith("Hello"):
                self.assertTrue(second_finished.wait(5))
                raise requests.Timeout()
            second_finished.set()
            raise requests.ConnectionError()

        reported = {}
        with (
            patch.object(app, "_request_translation", side_effect=translate),
            patch.object(app.requests, "Session", side_effect=lambda: Mock()),
        ):
            result = app.translate_segments(
                SEGMENTS, config(max_workers=2, max_retries=0),
                on_failure=lambda i, failure: reported.update({i: failure}),
            )
        self.assertEqual(result.failed_indices, [0, 2])
        self.assertEqual(reported, result.failures)
        self.assertIn("请求超时", result.failures[0].reason)
        self.assertIn("网络连接失败", result.failures[2].reason)

    def test_permanent_error_explains_authentication_without_response_body(self):
        session = Mock()
        session.post.return_value = response(
            status=401, payload={"error": {"message": "Your key is test-key; private subtitle"}}
        )
        with patch.object(app.requests, "Session", return_value=session):
            with self.assertRaises(app.TranslationAPIError) as caught:
                app.translate_segments(SEGMENTS, config())
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertIn("身份验证失败", str(caught.exception))
        self.assertNotIn("test-key", str(caught.exception))
        self.assertNotIn("private subtitle", str(caught.exception))
        session.post.assert_called_once()

    def test_jobs_skipped_after_permanent_error_are_not_counted_as_successes(self):
        def skipped_first(futures):
            wait(futures)
            # A completed, skipped job may be collected before the failed job.
            yield from reversed(list(futures))

        session = Mock()
        session.post.return_value = response(status=401)
        progress = Mock()
        with (
            patch.object(app.requests, "Session", return_value=session),
            patch.object(app, "as_completed", side_effect=skipped_first),
        ):
            with self.assertRaises(app.TranslationAPIError):
                app.translate_segments(SEGMENTS, config(), progress=progress)
        progress.assert_not_called()
        session.post.assert_called_once()

    def test_notebooks_show_failures_and_resume_without_stale_diagnostics(self):
        for entry in ["lrc", "srt"]:
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "episode.mp3"
                source.touch()
                with notebook_session(entry, source) as (cells, module, output):
                    module.OUTPUT_FORMAT = entry
                    session = Mock()
                    session.post.side_effect = [response("你好"), requests.Timeout("test-key")]
                    with (
                        patch.object(subprocess, "Popen", side_effect=fake_whisper),
                        patch.object(requests, "Session", return_value=session),
                    ):
                        exec(cells["transcribe"], module.__dict__)
                        exec(cells["translate"], module.__dict__)
                    log = output.getvalue()
                    self.assertIn("翻译失败：片段 3，已尝试 1 次", log)
                    self.assertIn("请求超时（REQUEST_TIMEOUT=30 秒）", log)
                    self.assertIn("本轮已处理：2/2（成功 1，失败 1）", log)
                    self.assertIn("翻译覆盖率：1/2，复用 0 条", log)
                    self.assertIn("请求超时（REQUEST_TIMEOUT=30 秒）：1 条", log)
                    self.assertNotIn("test-key", log)
                    self.assertIn("Goodbye.", module.output_path.read_text(encoding="utf-8"))
                    output.seek(0)
                    output.truncate(0)
                    session.reset_mock()
                    session.post.side_effect = None
                    session.post.return_value = response("再见")
                    with patch.object(requests, "Session", return_value=session):
                        exec(cells["translate"], module.__dict__)
                    session.post.assert_called_once()
                    self.assertEqual(session.post.call_args.kwargs["json"]["messages"][-1]["content"], "Goodbye.")
                    self.assertIn("本轮已处理：1/1（成功 1，失败 0）", output.getvalue())
                    self.assertIn("翻译覆盖率：2/2，复用 1 条", output.getvalue())
                    self.assertNotIn("失败原因汇总", output.getvalue())
                    self.assertEqual(module.result.failures, {})


if __name__ == "__main__":
    unittest.main()
