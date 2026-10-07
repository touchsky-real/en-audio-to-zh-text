import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import audio2text as app
from test_audio2text import fake_process, fake_whisper


class TranscriptionOutputTests(unittest.TestCase):
    def test_progress_without_newline_is_visible_before_child_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            acknowledgement = Path(folder) / "seen.txt"

            class Capture(io.StringIO):
                def write(self, value):
                    count = super().write(value)
                    if "50%" in self.getvalue():
                        acknowledgement.touch()
                    return count

            # The child only finishes once the parent has displayed its partial
            # progress. Buffering until a newline or process exit fails this test.
            child = """
import sys
import time
from pathlib import Path
sys.stderr.write('\\r50%|#####     | 50/100 [00:01<00:01, 50frames/s]')
sys.stderr.flush()
deadline = time.monotonic() + 5
while not Path(sys.argv[1]).exists():
    if time.monotonic() > deadline:
        sys.exit(3)
    time.sleep(0.01)
sys.stderr.write('\\r100%|##########| 100/100 [00:02<00:00, 50frames/s]\\n')
sys.stderr.flush()
"""
            with contextlib.redirect_stdout(Capture()) as output:
                app._run_whisper([sys.executable, "-u", "-c", child, str(acknowledgement)])
            self.assertIn("\r50%", output.getvalue())
            self.assertIn("\r100%", output.getvalue())

    def test_utf8_split_between_chunks_is_preserved(self):
        content = "\r识别进度：50%|█████     |"
        encoded = content.encode("utf-8")
        process = fake_process()
        process.stdout = Mock()
        process.stdout.read1.side_effect = [encoded[:2], encoded[2:7], encoded[7:], b""]
        with patch.object(app.subprocess, "Popen", return_value=process):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                app._run_whisper(["whisper"])
        self.assertEqual(output.getvalue(), content + "\n")

    def test_nonzero_exit_keeps_diagnostics_and_raises(self):
        child = "import sys; sys.stderr.write('decoder failed\\n'); sys.exit(7)"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(subprocess.CalledProcessError) as error:
                app._run_whisper([sys.executable, "-u", "-c", child])
        self.assertEqual(error.exception.returncode, 7)
        self.assertIn("decoder failed", output.getvalue())
        self.assertNotIn("100%", output.getvalue())

    def test_interrupt_stops_child_and_escalates_if_needed(self):
        for needs_kill in [False, True]:
            with self.subTest(needs_kill=needs_kill):
                process = fake_process()
                process.stdout = Mock()
                process.stdout.read1.side_effect = KeyboardInterrupt()
                process.poll.return_value = None
                if needs_kill:
                    process.wait.side_effect = [subprocess.TimeoutExpired("whisper", 5), 0]
                with patch.object(app.subprocess, "Popen", return_value=process):
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                        app._run_whisper(["whisper"])
                process.terminate.assert_called_once()
                process.wait.assert_any_call(timeout=5)
                self.assertEqual(process.kill.call_count, int(needs_kill))

    def test_restored_transcript_reports_skip_without_progress_bar(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "podcast.mp3"
            audio.touch()
            with patch.object(app.subprocess, "Popen", side_effect=fake_whisper):
                with contextlib.redirect_stdout(io.StringIO()):
                    app.transcribe_audio(audio)
            with patch.object(app.subprocess, "Popen") as process:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    app.transcribe_audio(audio)
            process.assert_not_called()
            self.assertIn("跳过音频识别", output.getvalue())
            self.assertNotIn("100%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
