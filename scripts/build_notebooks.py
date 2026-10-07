"""Generate standalone Colab notebooks from the tested shared module."""

import argparse
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def code_cell(cell_id, source):
    metadata = {"id": cell_id}
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build_notebook(notebook_name):
    dependencies = [
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    cells = [
        {
            "cell_type": "markdown",
            "id": "intro",
            "metadata": {"id": "intro"},
            "source": [
                "# 英文音频 → 中英双语字幕（LRC / SRT）\n",
                "先填写下方必要参数，再依次运行：**1. 配置参数 → 2. 准备环境 → 3. 转录 → 4. 翻译与保存**。建议选择 GPU 运行时。\n",
                "字幕保存到音频所在目录。默认保存进度，补译时只需重跑 Step 4；有已保存转录时，重启后运行 Step 1、2、4 即可。\n",
                '默认生成 LRC 歌词；在 Step 1 设置 `OUTPUT_FORMAT = "srt"` 可生成 SRT 字幕。\n',
                "设置 `OUTPUT_CHINESE = False` 仅输出英文；`SAVE_PROGRESS = False` 关闭进度保存与恢复。\n",
            ],
        }
    ]
    cells.append(
        code_cell(
            "config",
            """
        # [Step 1] 配置参数：先填写或确认下方必要参数
        import os
        from getpass import getpass

        # 必要参数：音频文件和目录必须正确；翻译接口仅在输出中文时需要
        AUDIO_FILENAME = "example.mp3"  # 必填：Drive 中的文件名，包含扩展名
        DRIVE_FOLDER = "/content/drive/MyDrive/podcast"  # 音频所在目录，按准备说明上传可不改
        API_KEY = os.environ.get("TRANSLATION_API_KEY", "")  # 留空时运行本步骤会提示隐藏输入
        BASE_URL = os.environ.get("TRANSLATION_BASE_URL", "https://api.openai.com/v1")  # API 根地址
        MODEL_NAME = os.environ.get("TRANSLATION_MODEL", "gpt-4.1-mini")  # 接口支持的翻译模型

        # 可选参数：一般保持默认即可
        OUTPUT_FORMAT = "lrc"  # "lrc"：歌词（默认）；"srt"：字幕
        OUTPUT_CHINESE = True  # False：仅输出英文，不需要翻译接口
        SAVE_PROGRESS = True  # 保存并恢复进度；False：不读写进度文件，每次重新处理
        AUTO_DISCONNECT = False  # True：仅在全部成功保存后自动断开
        MAX_WORKERS = 3
        MAX_RETRIES = 2  # 首次请求失败后的额外尝试次数；0 表示不重试
        REQUEST_TIMEOUT = 30  # 每次翻译请求的超时秒数
        WHISPER_MODEL = "turbo"

        if OUTPUT_CHINESE and not API_KEY:
            API_KEY = getpass("请输入翻译 API Key（输入不会显示）：")
        print(f"配置完成：{AUDIO_FILENAME}")
    """,
        )
    )
    core = (ROOT / "audio2text.py").read_text(encoding="utf-8")
    cells.append(
        code_cell(
            "setup",
            dedent(f"""
        # [Step 2] 准备环境并加载处理函数（直接运行，无需修改）
        _helpers_ready = False
        import shutil
        import subprocess
        import sys
        from google.colab import drive

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *{dependencies!r}],
            check=True,
        )
        if shutil.which("ffmpeg") is None:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
        drive.mount("/content/drive")
    """)
            + core
            + '\n_helpers_ready = True\nprint("环境准备完成。")\n',
        )
    )
    cells.append(
        code_cell(
            "transcribe",
            """
        # [Step 3] 转录音频（完成后再运行 Step 4，或切换到 CPU 运行时）
        transcription = None  # 新任务失败时不能误用上次结果

        def run_transcription():
            import time

            if not globals().get("_helpers_ready", False):
                raise RuntimeError("请先成功运行 Step 2 准备环境。")
            required = ("DRIVE_FOLDER", "AUDIO_FILENAME", "WHISPER_MODEL", "SAVE_PROGRESS")
            if any(name not in globals() for name in required):
                raise RuntimeError("请先运行 Step 1 配置参数。")
            source = resolve_audio_path(DRIVE_FOLDER, AUDIO_FILENAME)
            started = time.monotonic()
            segments = transcribe_audio(source, model=WHISPER_MODEL, save_progress=SAVE_PROGRESS)
            print(f"转录完成：{len(segments)} 个片段，耗时 {time.monotonic() - started:.1f} 秒。")
            if SAVE_PROGRESS:
                print("转录进度已保存。可直接运行 Step 4，或切换 CPU 后运行 Step 1、2、4。")
            else:
                print("进度保存已关闭，请在当前运行时执行 Step 4；重启后需要重新转录。")
            if not any(segment["text"] for segment in segments):
                print("未识别到有效文本，请检查音频内容。")
            return {"path": source, "audio": _audio_identity(source), "model": WHISPER_MODEL,
                    "segments": segments}

        transcription = run_transcription()
        """,
        )
    )
    cells.append(
        code_cell(
            "translate",
            """
        # [Step 4] 翻译并保存字幕（只补失败片段，不执行转录）
        def run_translation():
            import time

            if not globals().get("_helpers_ready", False):
                raise RuntimeError("请先成功运行 Step 2 准备环境。")
            required = (
                "DRIVE_FOLDER", "AUDIO_FILENAME", "WHISPER_MODEL", "OUTPUT_CHINESE",
                "API_KEY", "BASE_URL", "MODEL_NAME", "MAX_WORKERS", "MAX_RETRIES",
                "REQUEST_TIMEOUT", "SAVE_PROGRESS", "AUTO_DISCONNECT", "OUTPUT_FORMAT",
            )
            if any(name not in globals() for name in required):
                raise RuntimeError("请先运行 Step 1 配置参数。")
            if OUTPUT_FORMAT not in ("lrc", "srt"):
                raise ValueError('OUTPUT_FORMAT 只能填写 "lrc" 或 "srt"。请修改 Step 1 后重新运行。')
            source = resolve_audio_path(DRIVE_FOLDER, AUDIO_FILENAME)
            previous = globals().get("transcription")
            segments = None
            if (previous and previous["path"] == source
                    and previous["audio"] == _audio_identity(source) and previous["model"] == WHISPER_MODEL):
                segments = previous["segments"]
            elif SAVE_PROGRESS:
                segments = load_transcript(source, model=WHISPER_MODEL)
            if segments is None:
                raise RuntimeError("找不到当前音频和模型对应的转录结果，请先成功运行 Step 3。")
            translation_config = None
            if OUTPUT_CHINESE:
                translation_config = TranslationConfig(
                    api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME,
                    max_workers=MAX_WORKERS, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT,
                )

            failure_counts = {}

            def show_failure(index, failure):
                description = failure.description
                failure_counts[description] = failure_counts.get(description, 0) + 1
                print(
                    f"翻译失败：片段 {index + 1}，已尝试 {failure.attempts} 次；"
                    f"最后原因：{description}", flush=True,
                )

            def show_progress(finished, total):
                if finished % 20 == 0 or finished == total:
                    failed = sum(failure_counts.values())
                    print(
                        f"本轮已处理：{finished}/{total}（成功 {finished - failed}，失败 {failed}）",
                        flush=True,
                    )

            started = time.monotonic()
            output_path, result = export_subtitles(
                source, segments, subtitle_format=OUTPUT_FORMAT,
                config=translation_config, output_chinese=OUTPUT_CHINESE,
                progress=show_progress, on_failure=show_failure, save_progress=SAVE_PROGRESS,
            )
            print(f"字幕已保存：{output_path}，耗时 {time.monotonic() - started:.1f} 秒。")
            if OUTPUT_CHINESE:
                print(f"翻译覆盖率：{len(result.translations)}/{result.total}，复用 {result.reused} 条。")
            else:
                print("已导出英文字幕，未调用翻译 API。")
            if not SAVE_PROGRESS:
                print("进度保存已关闭，再次运行 Step 4 将重新翻译全部片段。")
            if result.failed_indices:
                print(f"仍有 {len(result.failed_indices)} 条翻译失败，已保留英文。")
                print("失败原因汇总（按每条片段最后一次错误统计）：")
                for description, count in sorted(failure_counts.items()):
                    print(f"  {description}：{count} 条")
                if SAVE_PROGRESS:
                    print("按上述原因排查后重跑 Step 4，即可继续补译；修改参数后先重跑 Step 1。")
            elif AUTO_DISCONNECT:
                from google.colab import runtime
                print("字幕已完整保存，5 秒后断开运行时。")
                time.sleep(5)
                runtime.unassign()
            return output_path, result

        output_path, result = run_translation()
        """,
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": notebook_name, "gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing files")
    args = parser.parse_args()
    outdated = []
    for notebook_name in ("audio2lrc.ipynb", "audio2srt.ipynb"):
        path = ROOT / notebook_name
        content = json.dumps(build_notebook(notebook_name), ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                outdated.append(path.name)
        else:
            path.write_text(content, encoding="utf-8")
    if outdated:
        parser.exit(
            1, "Outdated notebooks: " + ", ".join(outdated) + "\nRun python scripts/build_notebooks.py\n"
        )


if __name__ == "__main__":
    main()
